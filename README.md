# realesrgan-mlx-swift

Swift / [MLX](https://github.com/ml-explore/mlx-swift) **Real-ESRGAN super-resolution**
(SRVGGNetCompact) for Apple Silicon. Companion to the Python reference
[`xocialize/realesrgan-mlx`](https://github.com/xocialize/realesrgan-mlx).

Extracted from `forge-studio-optimizer`'s ForgeUpscaler playback tier — the shipped SR engine
(ADR-0008: SRVGGNet-general x4 won the A/B; 97.8–99.7 VMAF on real signage). Tile-based
processing (64² tiles, feathered seam blending) over BGRA/NV12 `CVPixelBuffer`s, NHWC MLX inside.

## Variants (vendored, x4)

| Variant | Checkpoint | Params |
|---|---|---|
| `.general` | `realesr_general_x4.safetensors` | ~1.21 M |
| `.generalWDN` | `realesr_general_wdn_x4.safetensors` (denoising) | ~1.21 M |
| `.anime` | `realesr_anime_x4.safetensors` | ~0.62 M |

The same checkpoints are published at
[`mlx-community/Real-ESRGAN-general-x4v3`](https://huggingface.co/mlx-community/Real-ESRGAN-general-x4v3)
and `mlx-community/Real-ESRGAN-animevideov3`. The heavy RRDBNet tiers
(`Real-ESRGAN-x4plus` / `x2plus` / `x4plus-anime-6B`) are CoreML/export-tier and not part of this
package.

## Usage

```swift
import RealESRGANMLX

let upscaler = SRVGGNetCompact_Playback(variant: .general)   // lazy weight load
let out: CVPixelBuffer = try await upscaler.upscale(inputBuffer)   // 4× BGRA
```

## Testing

`xcodebuild test -scheme realesrgan-mlx-swift` — 17 tests incl. real forward passes on the
vendored weights (needs the staged metallib; plain `swift test` can't run MLX).

## License

- **Port code (this repo):** MIT — see `LICENSE`.
- **Weights / upstream architecture:** BSD-3-Clause (xinntao/Real-ESRGAN).
