// MLXTileProcessor.swift
//
// Role: Tile-based driver for the MLX-Swift backed playback tiers
//       (`EfRLFN_Playback`, `SRVGGNetCompact_Playback`). Splits a full
//       BGRA `CVPixelBuffer` into overlapping NHWC tiles, runs each
//       through the supplied MLX forward closure, and feather-blends the
//       upscaled tiles into a BGRA output buffer.
//
// Why a parallel helper (not `TileProcessor`)?
//   - `TileProcessor` (legacy / CoreML) is hard-typed to `MLModel.prediction`
//     and CHW MLMultiArray inputs.
//   - The MLX path uses NHWC `MLXArray` inputs and a Swift closure for the
//     forward (so we don't have to pass concrete EfRLFN / SRVGGNetCompact
//     types into a shared driver).
//   - Same feathered-overlap blending math as `TileProcessor.writeTile`,
//     kept byte-faithful so seams look identical regardless of backend.
//
// Conventions:
//   - BGRA → RGB float32 [0, 1] on the way in (matches both upstream
//     EfRLFN and SRVGGNetCompact training preprocessing).
//   - RGB float32 [0, 1] → BGRA UInt8 on the way out with feathered seam
//     blending and per-pixel clamp.
//   - `MLX.eval(output)` is called inside the per-tile forward — required
//     before pulling values into CVPixelBuffer (MLX is lazy; an unevaluated
//     tensor materialises as zeros, the silent killer from mlx-porting).
//   - NHWC throughout — matches CLAUDE.md and the underlying MLX modules.
//   - Marked `@unchecked Sendable` because closures capture MLX state
//     that's not formally Sendable; the calling tier already enforces
//     single-thread access through the `async` boundary.

import CoreImage
import CoreVideo
import Foundation
import MLX

/// Tile driver for an MLX-Swift forward closure. The closure receives an
/// `[1, H, W, 3]` NHWC RGB float32 tensor and returns an upscaled tensor
/// `[1, H*scale, W*scale, 3]`. The driver handles CVPixelBuffer extraction,
/// stitching, and feather blending.
public struct MLXTileProcessor: @unchecked Sendable {

    let tileSize: Int
    let overlap: Int
    let scale: Int

    /// Shared CIContext for the input format normalisation below. Building a
    /// CIContext is expensive, so keep one alive for the process lifetime.
    private static let ciContext = CIContext(options: [.cacheIntermediates: false])

    public init(tileSize: Int, overlap: Int, scale: Int) {
        self.tileSize = tileSize
        self.overlap = overlap
        self.scale = scale
    }

    /// Normalise any incoming `CVPixelBuffer` to packed 32BGRA.
    ///
    /// CRITICAL: `FFmpegDecoder` (and the conversion pipeline generally) emits
    /// **NV12** (`kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`) — biplanar
    /// YUV — not BGRA. The tile extraction below reads the buffer as packed
    /// 4-byte BGRA via `CVPixelBufferGetBaseAddress`; handed an NV12 buffer it
    /// reads the Y (luma) plane as if it were BGRA, producing a sheared,
    /// grayscale garbage frame that the SR model then "upscales". This is the
    /// bug that made the Phase C.4 A/B score VMAF ≈ 0 on detailed content
    /// (smooth content survived the shear, which is why it wasn't uniformly
    /// zero). CoreImage decodes NV12 → RGB with the correct YCbCr matrix and
    /// video-range expansion, so render through it whenever the input isn't
    /// already BGRA. A buffer that is already BGRA passes straight through.
    private func normalizedBGRA(_ input: CVPixelBuffer) -> CVPixelBuffer {
        if CVPixelBufferGetPixelFormatType(input) == kCVPixelFormatType_32BGRA {
            return input
        }
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        var converted: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &converted) == kCVReturnSuccess,
              let out = converted else {
            return input  // fall back; extraction will be wrong but won't crash
        }
        let ci = CIImage(cvPixelBuffer: input)
        Self.ciContext.render(ci, to: out)
        return out
    }

    /// Process a full BGRA frame through the supplied MLX forward closure.
    ///
    /// - Parameters:
    ///   - input: source BGRA `CVPixelBuffer`.
    ///   - forward: closure that takes an `[1, tileH, tileW, 3]` NHWC RGB
    ///     float32 tensor and returns the upscaled result. The closure is
    ///     responsible for calling `MLX.eval` on its return value before
    ///     returning, so the driver can safely read floats out.
    /// - Returns: upscaled BGRA `CVPixelBuffer` at `scale` × the input
    ///   dimensions.
    public func process(
        _ rawInput: CVPixelBuffer,
        forward: (MLXArray) throws -> MLXArray
    ) throws -> CVPixelBuffer {
        let input = normalizedBGRA(rawInput)
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        let outWidth = width * scale
        let outHeight = height * scale
        let step = max(tileSize - overlap, 1)

        var outputBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outWidth,
            kCVPixelBufferHeightKey as String: outHeight,
        ]
        let status = CVPixelBufferCreate(
            nil, outWidth, outHeight, kCVPixelFormatType_32BGRA,
            attrs as CFDictionary, &outputBuffer
        )
        guard status == kCVReturnSuccess, let output = outputBuffer else {
            throw PlaybackTierError.inferenceError(
                "CVPixelBufferCreate failed (status=\(status))"
            )
        }

        CVPixelBufferLockBaseAddress(input, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(input, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(input) else {
            throw PlaybackTierError.inferenceError("CVPixelBuffer base address is nil (input)")
        }
        guard let dstBase = CVPixelBufferGetBaseAddress(output) else {
            throw PlaybackTierError.inferenceError("CVPixelBuffer base address is nil (output)")
        }
        let srcPtr = srcBase.assumingMemoryBound(to: UInt8.self)
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        let srcBPR = CVPixelBufferGetBytesPerRow(input)
        let dstBPR = CVPixelBufferGetBytesPerRow(output)

        // Pre-allocate the per-tile float32 scratch buffer so we don't
        // thrash the allocator for every tile of a 1080p frame.
        var tileScratch = [Float](repeating: 0, count: tileSize * tileSize * 3)

        for tileY in stride(from: 0, to: height, by: step) {
            for tileX in stride(from: 0, to: width, by: step) {
                // Clamp the tile origin so the last row/column stays inside the
                // frame (matches TileProcessor's behaviour).
                let x = min(tileX, max(0, width - tileSize))
                let y = min(tileY, max(0, height - tileSize))
                let tw = min(tileSize, width - x)
                let th = min(tileSize, height - y)

                // Extract BGRA → NHWC RGB float32 [0, 1] into the scratch
                // buffer, padding any out-of-clamp region with the nearest
                // valid pixel (same approach as TileProcessor).
                fillTileScratch(
                    src: srcPtr,
                    srcBPR: srcBPR,
                    x: x, y: y, tw: tw, th: th,
                    scratch: &tileScratch
                )

                // Wrap scratch as `[1, tileSize, tileSize, 3]` NHWC.
                let tileArray = MLXArray(tileScratch, [1, tileSize, tileSize, 3])
                let upscaled = try forward(tileArray)
                // `forward` is contractually expected to MLX.eval before
                // returning; we re-eval defensively so a stray lazy graph
                // can't silently zero the read-out.
                MLX.eval(upscaled)

                let shape = upscaled.shape
                guard shape.count == 4,
                      shape[0] == 1,
                      shape[3] == 3 else {
                    throw PlaybackTierError.inferenceError(
                        "unexpected forward output shape \(shape); want [1, H, W, 3]"
                    )
                }
                let outH = shape[1]
                let outW = shape[2]
                let outFloats = upscaled.asArray(Float.self)

                writeTile(
                    outFloats: outFloats,
                    outH: outH, outW: outW,
                    dst: dstPtr,
                    dstBPR: dstBPR,
                    x: x * scale, y: y * scale,
                    tw: tw * scale, th: th * scale,
                    outWidth: outWidth, outHeight: outHeight,
                    overlapScaled: overlap * scale
                )
            }
        }

        return output
    }

    // MARK: - Whole-frame fast path

    /// Adaptive entry point: run the whole frame through `forward` in a
    /// single pass when it fits `wholeFrameMaxPixels`, otherwise fall back
    /// to the tiled `process(_:forward:)`.
    ///
    /// EfRLFN and SRVGGNetCompact are both fully convolutional with **no
    /// internal downsampling**, so a whole-frame forward is numerically
    /// equivalent to the tiled path minus the feather-blend seams — and runs
    /// one GPU dispatch instead of one per tile (≈170 tiles for a 1080p frame
    /// at 128/16). The tiled path stays as the memory-bounded fallback for
    /// frames above the budget (e.g. the 4K corpus clip).
    ///
    /// - Parameter wholeFrameMaxPixels: input-pixel ceiling for the
    ///   single-pass path. Pass `0` to force tiling.
    public func processAdaptive(
        _ input: CVPixelBuffer,
        wholeFrameMaxPixels: Int,
        forward: (MLXArray) throws -> MLXArray
    ) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        if wholeFrameMaxPixels > 0, width * height <= wholeFrameMaxPixels {
            return try processWholeFrame(input, forward: forward)
        }
        return try process(input, forward: forward)
    }

    /// Single-pass forward over the entire frame. No tiling, no feather
    /// blending — the whole BGRA frame is lifted to one `[1, H, W, 3]` NHWC
    /// RGB float32 tensor, run through `forward`, and the `[1, H*scale,
    /// W*scale, 3]` result is written straight back to a BGRA buffer with a
    /// per-pixel `[0, 1]` clamp.
    public func processWholeFrame(
        _ rawInput: CVPixelBuffer,
        forward: (MLXArray) throws -> MLXArray
    ) throws -> CVPixelBuffer {
        let input = normalizedBGRA(rawInput)
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        let outWidth = width * scale
        let outHeight = height * scale

        let output = try makeOutputBuffer(outWidth, outHeight)

        CVPixelBufferLockBaseAddress(input, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(input, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(input) else {
            throw PlaybackTierError.inferenceError("CVPixelBuffer base address is nil (input)")
        }
        guard let dstBase = CVPixelBufferGetBaseAddress(output) else {
            throw PlaybackTierError.inferenceError("CVPixelBuffer base address is nil (output)")
        }
        let srcPtr = srcBase.assumingMemoryBound(to: UInt8.self)
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        let srcBPR = CVPixelBufferGetBytesPerRow(input)
        let dstBPR = CVPixelBufferGetBytesPerRow(output)

        // BGRA → NHWC RGB float32 [0, 1] for the whole frame.
        var scratch = [Float](repeating: 0, count: width * height * 3)
        scratch.withUnsafeMutableBufferPointer { buf in
            let base = buf.baseAddress!
            let inv = Float(1.0 / 255.0)
            for yy in 0 ..< height {
                let rowOff = yy * srcBPR
                for xx in 0 ..< width {
                    let off = rowOff + xx * 4
                    let b = Float(srcPtr[off + 0]) * inv
                    let g = Float(srcPtr[off + 1]) * inv
                    let r = Float(srcPtr[off + 2]) * inv
                    let d = (yy * width + xx) * 3
                    base[d + 0] = r
                    base[d + 1] = g
                    base[d + 2] = b
                }
            }
        }

        let inArray = MLXArray(scratch, [1, height, width, 3])
        let upscaled = try forward(inArray)
        // `forward` contractually evals before returning; re-eval defensively
        // so a stray lazy graph can't zero the read-out (mlx-porting).
        MLX.eval(upscaled)

        let shape = upscaled.shape
        guard shape.count == 4, shape[0] == 1, shape[3] == 3 else {
            throw PlaybackTierError.inferenceError(
                "unexpected whole-frame output shape \(shape); want [1, H, W, 3]"
            )
        }
        let oH = shape[1]
        let oW = shape[2]
        let outFloats = upscaled.asArray(Float.self)

        let copyH = min(oH, outHeight)
        let copyW = min(oW, outWidth)
        outFloats.withUnsafeBufferPointer { buf in
            let base = buf.baseAddress!
            for yy in 0 ..< copyH {
                let dstRow = yy * dstBPR
                for xx in 0 ..< copyW {
                    let s = (yy * oW + xx) * 3
                    let r = max(0, min(1, base[s + 0]))
                    let g = max(0, min(1, base[s + 1]))
                    let b = max(0, min(1, base[s + 2]))
                    let d = dstRow + xx * 4
                    dstPtr[d + 0] = UInt8(b * 255)
                    dstPtr[d + 1] = UInt8(g * 255)
                    dstPtr[d + 2] = UInt8(r * 255)
                    dstPtr[d + 3] = 255
                }
            }
        }

        return output
    }

    /// Allocate a BGRA output `CVPixelBuffer` at the given dimensions.
    private func makeOutputBuffer(_ outWidth: Int, _ outHeight: Int) throws -> CVPixelBuffer {
        var outputBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outWidth,
            kCVPixelBufferHeightKey as String: outHeight,
        ]
        let status = CVPixelBufferCreate(
            nil, outWidth, outHeight, kCVPixelFormatType_32BGRA,
            attrs as CFDictionary, &outputBuffer
        )
        guard status == kCVReturnSuccess, let output = outputBuffer else {
            throw PlaybackTierError.inferenceError(
                "CVPixelBufferCreate failed (status=\(status))"
            )
        }
        return output
    }

    // MARK: - Tile extraction (BGRA → NHWC RGB float32)

    /// Fill `scratch` with the `tileSize × tileSize × 3` RGB float32 view of
    /// the BGRA source region starting at `(x, y)`. Pixels outside `(x..x+tw,
    /// y..y+th)` clamp to the last valid pixel — matches `TileProcessor`'s
    /// edge-extension behaviour so a tile that overhangs the frame edge
    /// doesn't produce a hard cut.
    @inline(__always)
    private func fillTileScratch(
        src: UnsafePointer<UInt8>,
        srcBPR: Int,
        x: Int, y: Int, tw: Int, th: Int,
        scratch: inout [Float]
    ) {
        scratch.withUnsafeMutableBufferPointer { buf in
            let base = buf.baseAddress!
            let scale = Float(1.0 / 255.0)
            for ty in 0 ..< tileSize {
                let srcY = min(y + ty, y + th - 1)
                for tx in 0 ..< tileSize {
                    let srcX = min(x + tx, x + tw - 1)
                    let off = srcY * srcBPR + srcX * 4
                    // BGRA → RGB.
                    let b = Float(src[off + 0]) * scale
                    let g = Float(src[off + 1]) * scale
                    let r = Float(src[off + 2]) * scale
                    let dstOff = (ty * tileSize + tx) * 3
                    base[dstOff + 0] = r
                    base[dstOff + 1] = g
                    base[dstOff + 2] = b
                }
            }
        }
    }

    // MARK: - Tile write with feathered overlap blending

    /// Write the upscaled tile (NHWC RGB float32, contiguous) into the
    /// destination BGRA buffer with the same feathered blend
    /// `TileProcessor.writeTile` uses.
    @inline(__always)
    private func writeTile(
        outFloats: [Float],
        outH: Int, outW: Int,
        dst: UnsafeMutablePointer<UInt8>,
        dstBPR: Int,
        x: Int, y: Int, tw: Int, th: Int,
        outWidth: Int, outHeight: Int,
        overlapScaled: Int
    ) {
        let ov = max(overlapScaled, 1)
        let writeH = min(outH, th)
        let writeW = min(outW, tw)

        outFloats.withUnsafeBufferPointer { buf in
            let base = buf.baseAddress!
            for ty in 0 ..< writeH {
                let dstY = y + ty
                guard dstY < outHeight else { continue }
                for tx in 0 ..< writeW {
                    let dstX = x + tx
                    guard dstX < outWidth else { continue }
                    let srcOff = (ty * outW + tx) * 3
                    let r = max(0, min(1, base[srcOff + 0]))
                    let g = max(0, min(1, base[srcOff + 1]))
                    let b = max(0, min(1, base[srcOff + 2]))

                    // Feather weight: 1.0 in centre, ramps to 0 at edges.
                    let wxLeft  = min(Float(tx) / Float(ov), 1.0)
                    let wxRight = min(Float(outW - 1 - tx) / Float(ov), 1.0)
                    let wyTop   = min(Float(ty) / Float(ov), 1.0)
                    let wyBot   = min(Float(outH - 1 - ty) / Float(ov), 1.0)
                    let weight  = min(wxLeft, wxRight) * min(wyTop, wyBot)

                    let dstOff = dstY * dstBPR + dstX * 4
                    if weight >= 0.999 {
                        dst[dstOff + 0] = UInt8(b * 255)
                        dst[dstOff + 1] = UInt8(g * 255)
                        dst[dstOff + 2] = UInt8(r * 255)
                        dst[dstOff + 3] = 255
                    } else if dst[dstOff + 3] == 0 {
                        // First write into this pixel — feather hasn't been
                        // applied yet, just stamp.
                        dst[dstOff + 0] = UInt8(b * 255)
                        dst[dstOff + 1] = UInt8(g * 255)
                        dst[dstOff + 2] = UInt8(r * 255)
                        dst[dstOff + 3] = 255
                    } else {
                        let existB = Float(dst[dstOff + 0]) / 255.0
                        let existG = Float(dst[dstOff + 1]) / 255.0
                        let existR = Float(dst[dstOff + 2]) / 255.0
                        let blendR = existR * (1.0 - weight) + r * weight
                        let blendG = existG * (1.0 - weight) + g * weight
                        let blendB = existB * (1.0 - weight) + b * weight
                        dst[dstOff + 0] = UInt8(max(0, min(255, blendB * 255)))
                        dst[dstOff + 1] = UInt8(max(0, min(255, blendG * 255)))
                        dst[dstOff + 2] = UInt8(max(0, min(255, blendR * 255)))
                        dst[dstOff + 3] = 255
                    }
                }
            }
        }
    }
}
