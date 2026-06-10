// PlaybackTier.swift
//
// Role: Backend-agnostic abstraction for the ForgeUpscaler playback tier.
//       Concrete tiers (EfRLFN MLX today, SRVGGNetCompact MLX today as the
//       A/B baseline) conform to `PlaybackTier` so `PlaybackUpscaler` can
//       swap engines without changing call sites.
//
// Plan reference: Docs/Forge-CodingPlan-v1.0.md §C (playback tier),
//                 §D.2 (tile / overlap convention shared with export tier)
// ADR:           Docs/ADRs/0006-phase-c-unfreeze-efrlfn.md
//                (the A/B that this protocol exists to make runnable)
//
// Conventions (mirrors ExportTier.swift):
// - Public, Sendable surface.
// - Errors as a Sendable enum; no NSError shaping.
// - `inputTileSize` reports the tile dimension the tier feeds its model.
//   The two shipping tiers run at different tile sizes — SRVGGNetCompact's
//   tiny (~0.6–1.2 M params) body trains/infers natively at 64×64 per
//   upstream `inference_realesrgan_video.py`, while EfRLFN's deeper body
//   prefers 128×128 to amortise tile-overhead. The protocol is shape-honest,
//   not plan-prescriptive.

import CoreVideo
import Foundation

/// A pluggable backend for ForgeUpscaler's playback (fast, quality-first) tier.
/// Not realtime-gated — realtime SR is a separate-project concern (ADR-0009).
///
/// Tiers wrap a model + tile-driver pair behind a uniform call. The
/// `PlaybackUpscaler` selects a concrete tier at init time and delegates
/// every `upscale(_:)` to it; the conversion pipeline is unaware of which
/// tier is in use.
///
/// Implementations today are MLX-Swift backed (`EfRLFN_Playback`,
/// `SRVGGNetCompact_Playback`); the protocol intentionally leaves the door
/// open for a future CoreML re-export of either backend without changing
/// call sites.
public protocol PlaybackTier: Sendable {

    /// Stable identifier for logs / benchmarks. Examples:
    /// `"efrlfn-x4"`, `"srvggnet-general-x4"`, `"srvggnet-general-wdn-x4"`,
    /// `"srvggnet-anime-x4"`.
    var name: String { get }

    /// Spatial upscale factor (typically 2 or 4).
    var scaleFactor: Int { get }

    /// Edge length of one model-input tile, in input-resolution pixels.
    /// `inputResolution.width` and `.height` mirror this for callers that
    /// prefer the tuple form.
    ///
    /// Per-tier guidance:
    ///   - EfRLFN: 128 (matches the export-tier RealESRGAN_CoreML convention)
    ///   - SRVGGNetCompact: 64 (upstream `inference_realesrgan_video.py`)
    var inputTileSize: Int { get }

    /// Tile-to-tile overlap in input-resolution pixels. Plan §D.2 standardises
    /// on a 1:8 ratio (`tileOverlap == inputTileSize / 8`), which yields 16 px
    /// for EfRLFN's 128 tiles and 8 px for SRVGGNetCompact's 64 tiles.
    var tileOverlap: Int { get }

    /// Convenience: `(inputTileSize, inputTileSize)`. Reported as a tuple
    /// to align with the `ExportTier` surface and leave room for future
    /// non-square tiles.
    var inputResolution: (width: Int, height: Int) { get }

    /// Convenience: `(inputTileSize * scaleFactor, inputTileSize * scaleFactor)`.
    var outputResolution: (width: Int, height: Int) { get }

    /// Run the tier on a full-frame `CVPixelBuffer`. Implementations are
    /// expected to internally tile the input through the model and return
    /// a buffer at `scaleFactor`× the input dimensions in BGRA layout
    /// (matches what `FormatBridge` and `ExportPipeline` consume).
    func upscale(_ buffer: CVPixelBuffer) async throws -> CVPixelBuffer
}

/// Errors thrown by `PlaybackTier` implementations.
public enum PlaybackTierError: Error, Sendable, CustomStringConvertible {

    /// The backend's weights / model could not be located or compiled.
    /// Carries a human-readable detail (model name, path, underlying error).
    case modelLoadFailed(String)

    /// The requested upscale factor is not supported by this tier.
    case unsupportedScale(Int)

    /// The bundled safetensors / weight file could not be located inside
    /// `Bundle.module`. Carries the resource stem the lookup was for.
    case weightsNotFound(String)

    /// Inference itself failed (MLX returned no output, shape mismatch,
    /// pixel-buffer allocation failure, etc.). Carries a human-readable
    /// detail.
    case inferenceError(String)

    public var description: String {
        switch self {
        case .modelLoadFailed(let detail):
            return "PlaybackTier model load failed: \(detail)"
        case .unsupportedScale(let scale):
            return "PlaybackTier does not support scale=\(scale)"
        case .weightsNotFound(let resource):
            return "PlaybackTier weights not found in bundle: \(resource)"
        case .inferenceError(let detail):
            return "PlaybackTier inference failed: \(detail)"
        }
    }
}
