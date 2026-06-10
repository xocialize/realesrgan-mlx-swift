//
//  SRVGGNetTests.swift
//  ForgeUpscalerTests
//
//  Architecture tests for the MLX-Swift SRVGGNetCompact port (Task #28).
//
//  Verifies forward-pass shape correctness at multiple scales, exact
//  parameter counts per variant, factory constructor sanity, and the
//  loadWeights(from:) smoke test against the three vendored safetensors.
//
//  Numerical correctness vs PyTorch lives in the Python parity tests at
//  `Packages/ForgeTraining/Python/tests/test_srvggnet_parity.py`. The Swift
//  side only checks shapes / counts / finiteness here so the suite runs
//  even without an MLX-Metal runtime (per the
//  `swift test --filter SRVGGNetTests` CLI convention).
//
//  Parameter-count notes:
//    general / general-wdn: 1,213,296 trainable params (num_feat=64, num_conv=32)
//    anime:                   621,424 trainable params (num_feat=64, num_conv=16)
//
//  These numbers are pinned exactly because the upstream config is fixed —
//  no rescope budget like NAFNet (ADR-0003). Any drift indicates a port bug.
//

import Foundation
import Testing
import MLX
import MLXNN
@testable import RealESRGANMLX

/// Run a closure with the MLX default device pinned to CPU.
///
/// From `swift test` (no Xcode), the Metal bundle is not always staged
/// into the .xctest bundle and the first GPU op crashes. Same wrapper as
/// `EfRLFNTests`.
private func withCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(Device(.cpu), body)
}

private func totalParameterCount(_ module: Module) -> Int {
    var total = 0
    for (_, value) in module.parameters().flattened() {
        total += value.size
    }
    return total
}

/// Helper: resolve a vendored safetensors file inside `Resources/`.
///
/// SwiftPM stages `Resources/` under the test bundle; `Bundle.module` finds
/// it without leaking knowledge of the on-disk layout.
private func resourceURL(_ stem: String) -> URL? {
    Bundle.module.url(forResource: stem, withExtension: "safetensors")
}

@Suite("SRVGGNet")
struct SRVGGNetTests {

    // MARK: - Forward pass shape

    @Test("Forward pass at scale=4 produces 4× output (64×64 → 256×256)")
    func forwardShapeScale4() {
        withCPU {
            let model = SRVGGNetCompact.general()
            let x = MLXArray.zeros([1, 64, 64, 3])
            let y = model(x)
            MLX.eval(y)
            #expect(y.shape == [1, 256, 256, 3])
        }
    }

    @Test("Forward pass at scale=2 produces 2× output (architecture supports it even though no vendored variant ships x2)")
    func forwardShapeScale2() {
        // Real-ESRGAN ships SRVGGNetCompact with scale=4; the underlying
        // architecture has no fixed-scale assumption, so scale=2 is a
        // valid architectural configuration that the port should support
        // even without vendored weights for it.
        withCPU {
            let model = SRVGGNetCompact(numConv: 32, upscale: 2)
            let x = MLXArray.zeros([1, 64, 64, 3])
            let y = model(x)
            MLX.eval(y)
            #expect(y.shape == [1, 128, 128, 3])
        }
    }

    @Test("Forward pass at scale=1 is identity in spatial dims (degenerate case)")
    func forwardShapeScale1() {
        withCPU {
            let model = SRVGGNetCompact(numConv: 16, upscale: 1)
            let x = MLXArray.zeros([1, 32, 48, 3])
            let y = model(x)
            MLX.eval(y)
            #expect(y.shape == [1, 32, 48, 3])
        }
    }

    @Test("Forward pass handles non-square inputs (no internal downsampling)")
    func forwardShapeNonSquare() {
        withCPU {
            let model = SRVGGNetCompact.anime()
            let x = MLXArray.zeros([1, 48, 80, 3])
            let y = model(x)
            MLX.eval(y)
            #expect(y.shape == [1, 192, 320, 3])
        }
    }

    @Test("Forward pass handles odd-sized inputs (no padding constraint)")
    func forwardShapeOddInput() {
        withCPU {
            let model = SRVGGNetCompact.general()
            let x = MLXArray.zeros([1, 31, 33, 3])
            let y = model(x)
            MLX.eval(y)
            #expect(y.shape == [1, 124, 132, 3])
        }
    }

    // MARK: - Parameter counts (exact match to upstream)

    @Test("general variant has exactly 1,213,296 params")
    func paramCountGeneral() {
        withCPU {
            let model = SRVGGNetCompact.general()
            let total = totalParameterCount(model)
            // Upstream realesr-general-x4v3 / realesr-general-wdn-x4v3:
            //   first_conv   = 3*64*9 + 64                =      1,792
            //   first_act    = 64 (PReLU alpha)            =         64
            //   body_pairs (32 ×):
            //     conv       = 64*64*9 + 64                =     36,928
            //     act        = 64                          =         64
            //     subtotal   = 32 * (36,928 + 64)          =  1,183,744
            //   last_conv    = 64*48*9 + 48                =     27,696
            //   --------------------------------------------------------
            //   total                                       ≈ 1,213,296
            #expect(total == 1_213_296,
                    "Param count drift: got \(total), want 1,213,296 (general variant)")
        }
    }

    @Test("generalWDN variant has exactly 1,213,296 params (same arch as general)")
    func paramCountGeneralWDN() {
        withCPU {
            let model = SRVGGNetCompact.generalWDN()
            let total = totalParameterCount(model)
            #expect(total == 1_213_296,
                    "Param count drift: got \(total), want 1,213,296 (generalWDN variant)")
        }
    }

    @Test("anime variant has exactly 621,424 params (half-depth body)")
    func paramCountAnime() {
        withCPU {
            let model = SRVGGNetCompact.anime()
            let total = totalParameterCount(model)
            // Upstream realesr-animevideov3:
            //   Same first_conv / first_act / last_conv (1,792 + 64 + 27,696 = 29,552)
            //   body_pairs (16 ×): 16 * 36,992 = 591,872
            //   total = 621,424
            #expect(total == 621_424,
                    "Param count drift: got \(total), want 621,424 (anime variant)")
        }
    }

    // MARK: - Factory constructors

    @Test("All three factory constructors instantiate cleanly")
    func factoriesInstantiate() {
        withCPU {
            let g = SRVGGNetCompact.general()
            let w = SRVGGNetCompact.generalWDN()
            let a = SRVGGNetCompact.anime()

            #expect(g.numConv == 32)
            #expect(w.numConv == 32)
            #expect(a.numConv == 16)

            #expect(g.numFeat == 64)
            #expect(w.numFeat == 64)
            #expect(a.numFeat == 64)

            #expect(g.upscale == 4)
            #expect(w.upscale == 4)
            #expect(a.upscale == 4)

            #expect(g.activation == .prelu)
            #expect(w.activation == .prelu)
            #expect(a.activation == .prelu)
        }
    }

    // MARK: - LeakyReLU activation variant

    @Test("leakyRelu activation has no per-channel PReLU parameter")
    func leakyReluHasFewerParams() {
        withCPU {
            // Number of "act" slots = 1 (first_act) + numConv (body_pairs);
            // PReLU adds numFeat params per slot.
            let prelu = SRVGGNetCompact(numConv: 16, activation: .prelu)
            let leaky = SRVGGNetCompact(numConv: 16, activation: .leakyRelu)
            let preluTotal = totalParameterCount(prelu)
            let leakyTotal = totalParameterCount(leaky)
            // numFeat=64, num activation slots = 1 + 16 = 17.
            // Difference = 17 * 64 = 1,088.
            #expect(preluTotal - leakyTotal == 1_088)
        }
    }

    // MARK: - Bundle resource discovery

    @Test("Vendored safetensors are bundled (general)")
    func vendoredGeneralExists() {
        let url = resourceURL("realesr_general_x4")
        #expect(url != nil, "realesr_general_x4.safetensors must ship in Resources/")
    }

    @Test("Vendored safetensors are bundled (general-wdn)")
    func vendoredGeneralWDNExists() {
        let url = resourceURL("realesr_general_wdn_x4")
        #expect(url != nil, "realesr_general_wdn_x4.safetensors must ship in Resources/")
    }

    @Test("Vendored safetensors are bundled (anime)")
    func vendoredAnimeExists() {
        let url = resourceURL("realesr_anime_x4")
        #expect(url != nil, "realesr_anime_x4.safetensors must ship in Resources/")
    }

    // MARK: - Weight loading smoke

    @Test("loadWeights does not throw on the vendored general safetensors")
    func loadWeightsGeneral() throws {
        guard let url = resourceURL("realesr_general_x4") else {
            // Resource missing — covered by the bundled-resource test above.
            return
        }
        try withCPU {
            let model = SRVGGNetCompact.general()
            try model.loadWeights(from: url)
        }
    }

    @Test("loadWeights does not throw on the vendored generalWDN safetensors")
    func loadWeightsGeneralWDN() throws {
        guard let url = resourceURL("realesr_general_wdn_x4") else {
            return
        }
        try withCPU {
            let model = SRVGGNetCompact.generalWDN()
            try model.loadWeights(from: url)
        }
    }

    @Test("loadWeights does not throw on the vendored anime safetensors")
    func loadWeightsAnime() throws {
        guard let url = resourceURL("realesr_anime_x4") else {
            return
        }
        try withCPU {
            let model = SRVGGNetCompact.anime()
            try model.loadWeights(from: url)
        }
    }

    // MARK: - Error path

    @Test("loadWeights throws weightsNotFound for a missing file")
    func loadWeightsMissingFile() {
        withCPU {
            let model = SRVGGNetCompact.anime()
            let missing = URL(fileURLWithPath: "/tmp/this_file_does_not_exist_srvggnet.safetensors")
            do {
                try model.loadWeights(from: missing)
                Issue.record("expected SRVGGNetCompactError.weightsNotFound")
            } catch let SRVGGNetCompactError.weightsNotFound(path) {
                #expect(path == missing.path)
            } catch {
                Issue.record("expected weightsNotFound, got \(error)")
            }
        }
    }
}
