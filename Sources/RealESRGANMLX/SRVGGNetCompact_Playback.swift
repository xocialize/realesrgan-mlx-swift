// SRVGGNetCompact_Playback.swift
//
// Role: Concrete `PlaybackTier` backed by the vendored SRVGGNetCompact
//       MLX-Swift port (Task #28). Three variants — general, generalWDN,
//       anime — each load from a distinct `Resources/realesr_*_x4.safetensors`
//       bundle.
//
// Plan ref: Forge-CodingPlan-v1.0.md §C (playback A/B baseline) / §C.5
// ADR:      Docs/ADRs/0006-phase-c-unfreeze-efrlfn.md §"Ship criterion"
// Upstream: https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause)
// Weights:  Resources/realesr_{general,general_wdn,anime}_x4.safetensors
//
// Tile shape: 64 / 8 — matches the upstream `realesr-general-x4v3`
// `inference_realesrgan_video.py` training/inference tile size. Same
// 1:8 ratio as EfRLFN's 128/16 (per plan §D.2 convention) — just at a
// finer grid because SRVGGNetCompact's tiny body (~0.6–1.2 M params)
// runs natively on small tiles and a coarser grid would lose more to
// the seam-blend overlap than it gains in per-tile throughput.
//
// Lazy init: weights load on first `upscale(_:)` behind an `NSLock` so
// the protocol-conformance / surface-query path stays free.

import CoreVideo
import Foundation
import MLX
import MLXNN

/// SRVGGNetCompact playback tier — MLX-Swift, BSD-3-Clause.
///
/// Marked `@unchecked Sendable` because the MLX module + lock are mutated
/// across the `async` boundary; the lock guards the only crossing point.
public final class SRVGGNetCompact_Playback: PlaybackTier, @unchecked Sendable {

    /// Vendored SRVGGNetCompact variant.
    public enum Variant: String, Sendable {
        case general        // realesr_general_x4.safetensors    (num_conv=32, ~1.21 M)
        case generalWDN     // realesr_general_wdn_x4.safetensors (num_conv=32, ~1.21 M)
        case anime          // realesr_anime_x4.safetensors      (num_conv=16, ~0.62 M)

        /// Resource stem (no extension) of the vendored safetensors file.
        public var safetensorsName: String {
            switch self {
            case .general:    return "realesr_general_x4"
            case .generalWDN: return "realesr_general_wdn_x4"
            case .anime:      return "realesr_anime_x4"
            }
        }

        /// Stable `PlaybackTier.name` identifier per variant.
        var tierName: String {
            switch self {
            case .general:    return "srvggnet-general-x4"
            case .generalWDN: return "srvggnet-general-wdn-x4"
            case .anime:      return "srvggnet-anime-x4"
            }
        }
    }

    // MARK: - PlaybackTier surface

    public let name: String
    public let scaleFactor: Int
    public let inputTileSize: Int = 64
    public let tileOverlap: Int = 8

    public var inputResolution: (width: Int, height: Int) {
        (inputTileSize, inputTileSize)
    }

    public var outputResolution: (width: Int, height: Int) {
        (inputTileSize * scaleFactor, inputTileSize * scaleFactor)
    }

    /// The vendored variant this instance wraps. Exposed for tests / logs.
    public let variant: Variant

    // MARK: - Internals

    private let model: SRVGGNetCompact
    private let tileProcessor: MLXTileProcessor
    private let weightsURL: URL
    private let loadLock = NSLock()
    private var weightsLoaded = false

    /// `mx.compile`-traced forward, built lazily once weights are loaded.
    /// Same rationale as `EfRLFN_Playback`: kept identical so the Phase C.4
    /// A/B compares the two backends under the same inference strategy.
    private var compiledForward: (@Sendable (MLXArray) -> MLXArray)?

    /// Whole-frame fast-path ceiling. SRVGGNetCompact is fully convolutional
    /// (no internal downsampling), so it shares EfRLFN's whole-frame
    /// treatment; frames above 1080p fall back to 64-px tiling.
    private let wholeFrameMaxPixels = 1920 * 1080

    // MARK: - Init

    /// Initialise an SRVGGNetCompact playback tier.
    ///
    /// The three vendored v3 checkpoints all ship at x4; the architecture
    /// itself supports any integer scale ≥ 1, but mismatched weight + scale
    /// pairs would fail at safetensors load time. We pin scale=4 here for
    /// the same forward-compat reason `EfRLFN_Playback` does.
    public init(variant: Variant) throws {
        self.variant = variant
        self.name = variant.tierName
        self.scaleFactor = 4

        guard let url = Bundle.module.url(
            forResource: variant.safetensorsName,
            withExtension: "safetensors"
        ) else {
            throw PlaybackTierError.weightsNotFound(variant.safetensorsName)
        }
        self.weightsURL = url

        switch variant {
        case .general:    self.model = SRVGGNetCompact.general()
        case .generalWDN: self.model = SRVGGNetCompact.generalWDN()
        case .anime:      self.model = SRVGGNetCompact.anime()
        }

        self.tileProcessor = MLXTileProcessor(
            tileSize: inputTileSize,
            overlap: tileOverlap,
            scale: scaleFactor
        )
    }

    // MARK: - PlaybackTier impl

    public func upscale(_ buffer: CVPixelBuffer) async throws -> CVPixelBuffer {
        let run = try ensureReady()
        do {
            return try tileProcessor.processAdaptive(
                buffer,
                wholeFrameMaxPixels: wholeFrameMaxPixels
            ) { tile in
                let y = run(tile)
                MLX.eval(y)
                return y
            }
        } catch let err as PlaybackTierError {
            throw err
        } catch {
            throw PlaybackTierError.inferenceError(String(describing: error))
        }
    }

    // MARK: - Weights + compile

    /// Load weights (once) and build the compiled forward (once). See
    /// `EfRLFN_Playback.ensureReady` for the rationale; kept identical so the
    /// A/B runs both backends through the same inference strategy.
    private func ensureReady() throws -> @Sendable (MLXArray) -> MLXArray {
        loadLock.lock()
        defer { loadLock.unlock() }
        if let f = compiledForward { return f }
        if !weightsLoaded {
            do {
                try model.loadWeights(from: weightsURL)
                weightsLoaded = true
            } catch let err as SRVGGNetCompactError {
                throw PlaybackTierError.modelLoadFailed(String(describing: err))
            } catch {
                throw PlaybackTierError.modelLoadFailed(String(describing: error))
            }
        }
        let m = model
        let f = compile { x in m(x) }
        compiledForward = f
        return f
    }
}
