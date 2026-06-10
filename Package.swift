// swift-tools-version: 6.0
import PackageDescription

// realesrgan-mlx-swift — Swift/MLX Real-ESRGAN super-resolution (SRVGGNetCompact) for Apple
// Silicon. Companion to the Python reference `xocialize/realesrgan-mlx`. Extracted from
// forge-studio-optimizer's ForgeUpscaler playback tier (ADR-0008: SRVGGNet-general x4 is the
// shipped winner). Three vendored x4 variants (general / general-WDN / anime) + tile-based
// processing with feathered seam blending; the RRDBNet x4plus tier (CoreML) stays Forge-side.
let package = Package(
    name: "realesrgan-mlx-swift",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "RealESRGANMLX", targets: ["RealESRGANMLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        .target(
            name: "RealESRGANMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            // Per-file .copy so the bundle layout is flat (forge ADR-0011).
            resources: [
                .copy("Resources/realesr_general_x4.safetensors"),
                .copy("Resources/realesr_general_wdn_x4.safetensors"),
                .copy("Resources/realesr_anime_x4.safetensors"),
            ]
        ),
        .testTarget(
            name: "RealESRGANMLXTests",
            dependencies: [
                "RealESRGANMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
    ]
)
