//
//  SRVGGNetCompact.swift
//  ForgeUpscaler / Playback
//
//  Role: MLX-Swift port of SRVGGNetCompact (Real-ESRGAN), the lightweight
//        baseline that the Phase C.4 EfRLFN A/B has to beat. Three pinned
//        upstream variants — realesr-general-x4v3 / realesr-general-wdn-x4v3
//        / realesr-animevideov3 — vendored at FP16 in this package's
//        `Resources/`. Same NHWC + `(C, r, r)` pixel-shuffle conventions as
//        EfRLFN.swift / NAFNet.swift.
//
//  Plan ref: Forge-CodingPlan-v1.0.md §C — playback-tier baseline (Task #28)
//  ADR: Docs/ADRs/0006-phase-c-unfreeze-efrlfn.md §"Ship criterion"
//  Upstream: https://github.com/xinntao/Real-ESRGAN
//            (`realesrgan/archs/srvgg_arch.py`, BSD-3-Clause, © 2021 Xintao Wang)
//  Variants — see also Resources/MODELS.md §SRVGGNet:
//
//      realesr-general-x4v3       num_feat=64, num_conv=32, prelu, x4   1,213,296 params
//      realesr-general-wdn-x4v3   num_feat=64, num_conv=32, prelu, x4   1,213,296 params
//      realesr-animevideov3       num_feat=64, num_conv=16, prelu, x4     621,424 params
//
//  Architecture summary (verbatim per upstream `srvgg_arch.py`):
//
//      Input  [N, H, W, 3]
//        body[0]   = Conv2d(3, num_feat, 3×3, padding=1)
//        body[1]   = Activation                          # PReLU(num_feat), shared formula across variants
//        body[2k]  = Conv2d(num_feat, num_feat, 3×3)     for k in 1..num_conv
//        body[2k+1]= Activation                          for k in 1..num_conv
//        body[-1]  = Conv2d(num_feat, 3*upscale², 3×3)
//      out = PixelShuffle(upscale)(out)
//      base = nearest-upsample(input, upscale)
//      return out + base
//
//  Critical points (per upstream — do NOT redesign):
//      - **Residual upsample is NEAREST**, not bilinear. Easy to misread the
//        plan/spec — the canonical PyTorch source uses
//        `F.interpolate(x, scale_factor=self.upscale, mode='nearest')`.
//      - **`act_type='prelu'` ⇒ `PReLU(num_parameters=num_feat)`** — one alpha
//        per channel, not a single shared scalar. State-dict carries a
//        `(num_feat,)` weight tensor per activation.
//      - **Pixel-shuffle uses `(C, r, r)` channel split** — matches PyTorch
//        `nn.PixelShuffle`. The buggy `(r, r, C)` variant produces the
//        same shape but wrong content (max_abs ≈ 2.27 in parity). Same
//        helper logic as `EfRLFN.swift::pixelShuffleNHWC` — preserved here
//        verbatim because that file's helper is fileprivate.
//      - **NO normalization layers** — pure conv + activation. None of the
//        eps-default traps from `mlx-porting` pitfall #5 apply.
//
//  Conventions:
//      - NHWC throughout (matches CLAUDE.md "Conventions")
//      - `@unchecked Sendable` on classes holding MLX state
//      - macOS 14 deployment target (no bump)
//      - Public types `Sendable`
//      - Standard weight-load pipeline:
//          `MLX.loadArrays → ModuleParameters.unflattened → update(verify: .noUnusedKeys)`
//        (mirrors `EfRLFN.loadWeights(from:)`)
//
//  Numerical-parity validation lives in
//  `Packages/ForgeTraining/Python/tests/test_srvggnet_parity.py` — the
//  Swift port is shape-tested here; the byte-level parity assertion runs
//  against the MLX-Python twin (`Packages/ForgeTraining/Python/models/
//  srvggnet_mlx.py`), which is the same architecture in MLX-Python and the
//  oracle for the PyTorch reference.
//
//  License: Real-ESRGAN © 2021 Xintao Wang and Real-ESRGAN authors,
//  BSD-3-Clause. See `Packages/ForgeUpscaler/LICENSES.md` §1B.
//

import Foundation
import MLX
import MLXNN

// MARK: - Conv layer helper

/// 3×3 / 1×1 "same" Conv2d matching the upstream `padding=1` (or `padding=0`
/// for 1×1) convention used in `srvgg_arch.py`.
///
/// Mirrors `EfRLFN.sameConv` — duplicated rather than promoted to package
/// surface because the helper is two lines and the two ports prefer to be
/// self-contained for easy upstream-diff review (per mlx-porting skill
/// "preserve isomorphic structure").
@inline(__always)
private func sameConv(
    _ inCh: Int,
    _ outCh: Int,
    kernel k: Int,
    bias: Bool = true
) -> Conv2d {
    Conv2d(
        inputChannels: inCh,
        outputChannels: outCh,
        kernelSize: .init(k),
        stride: 1,
        padding: .init((k - 1) / 2),
        bias: bias
    )
}

// MARK: - PixelShuffle (NHWC)

/// Upsample by `r` via channel-to-space reshuffling — NHWC.
///
/// Input  : `[N, H, W, C * r * r]`
/// Output : `[N, H * r, W * r, C]`
///
/// Uses the verified `(C, r, r)` channel-split ordering to match PyTorch's
/// `nn.PixelShuffle`. The `(r, r, C)` variant produces the correct *shape*
/// but wrong *content* (max_abs ≈ 2.27 in parity tests). The fix lives in
/// `EfRLFN.swift::pixelShuffleNHWC` and `NAFNet.swift::pixelShuffleNHWC` —
/// duplicated here so this file stands alone for upstream-diff review.
///
/// See `anthropic-skills:mlx-porting` pitfall #7 ("the recurring one") and
/// the CLAUDE.md §Conventions note on pixel-shuffle ordering.
private func pixelShuffleNHWC(_ x: MLXArray, upscaleFactor r: Int) -> MLXArray {
    let s = x.shape
    let N = s[0]
    let H = s[1]
    let W = s[2]
    let Cin = s[3]
    precondition(Cin % (r * r) == 0,
                 "pixelShuffleNHWC: input channels (\(Cin)) must be divisible by r*r (\(r * r))")
    let C = Cin / (r * r)

    // Channel-split (C, r_i, r_j) → PyTorch's reading order.
    // [N, H, W, C, r_i, r_j]
    let reshaped = x.reshaped([N, H, W, C, r, r])
    // Permute so r_i sits next to H, r_j next to W, C trails:
    // [N, H, r_i, W, r_j, C]
    let transposed = reshaped.transposed(0, 1, 4, 2, 5, 3)
    // [N, H*r, W*r, C]
    return transposed.reshaped([N, H * r, W * r, C])
}

// MARK: - Nearest-neighbour upsample (NHWC)

/// 2-D nearest-neighbour upsample by an integer `factor` on the spatial axes.
///
/// Mirrors `torch.nn.functional.interpolate(x, scale_factor=factor, mode='nearest')`
/// for 4-D NHWC tensors. The upstream `SRVGGNetCompact.forward` uses this
/// for the residual `base` term that's added to the pixel-shuffle output:
///
///     base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
///     return self.upsampler(out) + base
///
/// PyTorch's nearest mode for integer scale factors is equivalent to a
/// repeat along the spatial axes. We do that via `repeated(repeats:axis:)`
/// — element-wise replication, not block repeat (block repeat would be
/// `tile`, which gives the wrong layout: `[A,B,A,B]` instead of
/// `[A,A,B,B]`). Easy to get backwards; see mlx-porting pitfall #7's
/// `mx.tile` vs `mx.repeat` note.
@inline(__always)
private func upsampleNearestNHWC(_ x: MLXArray, factor: Int) -> MLXArray {
    if factor == 1 { return x }
    // Repeat along H axis (1), then along W axis (2). Each repeat duplicates
    // every input element `factor` times in place along that axis.
    // `MLXArray.repeated` is a static method on MLXArray (not an instance
    // method); it returns each element of the input array repeated `count`
    // times along `axis`, producing the in-place duplication semantics that
    // match `mx.repeat` in Python and PyTorch's nearest-mode interpolate.
    let alongH = MLXArray.repeated(x, count: factor, axis: 1)
    let alongHW = MLXArray.repeated(alongH, count: factor, axis: 2)
    return alongHW
}

// MARK: - SRVGGNetCompact

/// Compact VGG-style SR network used by Real-ESRGAN. Three vendored
/// variants — see the file header for the per-variant `num_feat`/`num_conv`
/// table — share the same architecture and key scheme; only the body depth
/// varies.
///
/// State-dict key scheme (matches upstream `nn.ModuleList` indexing):
///
///     body.0.weight, body.0.bias            # first conv (3 → num_feat)
///     body.1.weight                          # first PReLU alpha (num_feat,)
///     body.2.weight, body.2.bias            # body conv 1
///     body.3.weight                          # body PReLU 1
///     ...
///     body.{2*num_conv}.weight, .bias        # body conv num_conv
///     body.{2*num_conv+1}.weight             # body PReLU num_conv
///     body.{2*num_conv+2}.weight, .bias      # last conv (num_feat → 3*r²)
///
/// We carry that scheme verbatim through `@ModuleInfo(key:)` annotations so
/// `MLX.loadArrays → ModuleParameters.unflattened → update(verify:)` lands
/// the upstream weights without an out-of-band rename map. The converter
/// (`Packages/ForgeTraining/Scripts/convert_srvggnet_to_mlx.py`) only does
/// the `(O, I, kH, kW) → (O, kH, kW, I)` conv transpose.
public final class SRVGGNetCompact: Module, @unchecked Sendable {

    /// Activation function used between body convs.
    ///
    /// Upstream supports `'relu' | 'prelu' | 'leakyrelu'`. All three vendored
    /// v3 checkpoints train with PReLU; we expose `leakyRelu` for future
    /// variants without expanding the test surface yet.
    public enum Activation: String, Sendable {
        case prelu
        case leakyRelu
    }

    // Inputs / outputs follow the upstream `__init__` defaults verbatim.
    public let numInCh: Int
    public let numOutCh: Int
    public let numFeat: Int
    public let numConv: Int
    public let upscale: Int
    public let activation: Activation

    // The body is a flat `[Module]` (mirrors upstream's `nn.ModuleList`)
    // rather than two parallel arrays. `@ModuleInfo` with explicit keys keeps
    // the dot-indexed state-dict layout on weight save/load.
    //
    // We cannot use `@ModuleInfo var body: [Module]` because Swift property
    // wrappers' generic resolver doesn't preserve heterogeneous Module
    // subtypes in a list — and `[any Module]` doesn't unflatten cleanly.
    // Use named first/last conv + an indexed array of (conv, act) pairs.
    // Property names use Swift camelCase per project convention; the
    // converter's safetensors use snake_case to match the existing
    // ForgeTraining/Python/models/srvggnet_mlx.py reference. Bridge with
    // explicit `(key:)` overrides so `Module.update(verify: .noUnusedKeys)`
    // accepts the converter's output. Without these, the smoke test catches
    // `unhandledKeys: ["body_pairs", "first_act", "first_conv", "last_conv"]`.
    @ModuleInfo(key: "first_conv") var firstConv: Conv2d
    @ModuleInfo(key: "first_act") var firstAct: ActivationLayer

    /// Body conv/act pairs. Each entry knows its own `bodyIndex` (the
    /// `i`-th body conv, NOT the upstream `body.N` index — the converter
    /// re-encodes the upstream index from `(2*(bodyIndex+1), 2*(bodyIndex+1)+1)`).
    ///
    /// We keep this as `[BodyPair]` rather than two parallel `@ModuleInfo`
    /// arrays so the parameter tree stays flat and the converter can emit
    /// the upstream `body.{N}` keys directly. See `BodyPair.flattenedKey(_:)`.
    @ModuleInfo(key: "body_pairs") var bodyPairs: [BodyPair]

    @ModuleInfo(key: "last_conv") var lastConv: Conv2d

    /// One (conv → act) pair from the body. Holds an activation that may be
    /// `PReLU` or `LeakyReLU`; both expose a `callAsFunction(_:)` and the
    /// PReLU one owns a learnable `weight` parameter.
    public final class BodyPair: Module, @unchecked Sendable {
        @ModuleInfo public var conv: Conv2d
        @ModuleInfo public var act: ActivationLayer

        public init(channels: Int, activation: Activation) {
            self._conv.wrappedValue = sameConv(channels, channels, kernel: 3)
            self._act.wrappedValue = ActivationLayer.build(activation, channels: channels)
        }

        @inline(__always)
        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            act(conv(x))
        }
    }

    /// Initialise an SRVGGNetCompact instance.
    ///
    /// All defaults match the upstream `realesr-general-x4v3` config; the
    /// anime variant overrides `numConv: 16`.
    ///
    /// - Parameters:
    ///   - numInCh: input channels. Upstream default 3.
    ///   - numOutCh: output channels. Upstream default 3.
    ///   - numFeat: intermediate feature width. Upstream default 64.
    ///   - numConv: number of body conv layers (NOT including the first /
    ///     last conv). 32 for the general variants, 16 for the anime variant.
    ///   - upscale: SR factor. The three v3 variants ship at upscale=4; the
    ///     architecture supports any `upscale ≥ 1` (scale=1 degenerates the
    ///     pixel-shuffle into a no-op and the residual into a passthrough).
    ///   - activation: activation type. All v3 variants use `.prelu`.
    public init(
        numInCh: Int = 3,
        numOutCh: Int = 3,
        numFeat: Int = 64,
        numConv: Int = 32,
        upscale: Int = 4,
        activation: Activation = .prelu
    ) {
        precondition(numInCh >= 1, "SRVGGNetCompact numInCh must be ≥ 1; got \(numInCh)")
        precondition(numOutCh >= 1, "SRVGGNetCompact numOutCh must be ≥ 1; got \(numOutCh)")
        precondition(numFeat >= 1, "SRVGGNetCompact numFeat must be ≥ 1; got \(numFeat)")
        precondition(numConv >= 0, "SRVGGNetCompact numConv must be ≥ 0; got \(numConv)")
        precondition(upscale >= 1, "SRVGGNetCompact upscale must be ≥ 1; got \(upscale)")

        self.numInCh = numInCh
        self.numOutCh = numOutCh
        self.numFeat = numFeat
        self.numConv = numConv
        self.upscale = upscale
        self.activation = activation

        // First conv: 3 → num_feat, 3×3, padding=1.
        self._firstConv.wrappedValue = sameConv(numInCh, numFeat, kernel: 3)
        // First activation — runs once after the first conv.
        self._firstAct.wrappedValue = ActivationLayer.build(activation, channels: numFeat)

        // Body: `numConv` repetitions of (Conv2d num_feat→num_feat, Activation).
        var pairs: [BodyPair] = []
        pairs.reserveCapacity(numConv)
        for _ in 0 ..< numConv {
            pairs.append(BodyPair(channels: numFeat, activation: activation))
        }
        self._bodyPairs.wrappedValue = pairs

        // Last conv: num_feat → num_out * upscale². No trailing activation in
        // the upstream model; activation only appears between body convs and
        // after the first conv.
        self._lastConv.wrappedValue = sameConv(
            numFeat,
            numOutCh * upscale * upscale,
            kernel: 3
        )
    }

    /// Forward pass.
    ///
    /// - Parameter x: `[N, H, W, numInCh]` NHWC image tensor. Upstream
    ///   inference expects values in the same range the model was trained on
    ///   (Real-ESRGAN trains in `[0, 1]`).
    /// - Returns: `[N, H * upscale, W * upscale, numOutCh]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Body: first conv → first act → (conv → act) × numConv → last conv.
        var out = firstAct(firstConv(x))
        for pair in bodyPairs {
            out = pair(out)
        }
        out = lastConv(out)

        // Pixel-shuffle reconstruction head.
        let shuffled: MLXArray
        if upscale == 1 {
            // Degenerate case: PixelShuffle(1) is the identity.
            shuffled = out
        } else {
            shuffled = pixelShuffleNHWC(out, upscaleFactor: upscale)
        }

        // Residual: nearest-upsample the input and add. Matches upstream's
        // `F.interpolate(x, scale_factor=self.upscale, mode='nearest')`.
        let base = upsampleNearestNHWC(x, factor: upscale)
        return shuffled + base
    }
}

// MARK: - Activation wrapper

/// Lightweight wrapper so the body can hold either a `PReLU` or a
/// `LeakyReLU` (both subclass `Module`) inside a `@ModuleInfo` slot.
///
/// MLXNN's `PReLU.weight` is annotated as a tracked parameter so the
/// standard `MLX.loadArrays → ModuleParameters.unflattened → update(verify:)`
/// chain places the upstream `body.{2k+1}.weight` tensor into the right slot.
/// `LeakyReLU` has no parameters; its slot is parameter-free.
public final class ActivationLayer: Module, @unchecked Sendable {

    /// PReLU instance when `activation = .prelu`. `nil` for leaky ReLU.
    ///
    /// We keep it as a `@ModuleInfo` rather than a stored constant so MLX's
    /// parameter tree walks into it cleanly. The upstream key for the
    /// activation is `body.{2k+1}.weight` (a single-tensor entry, no sub-
    /// module), so the converter remaps that to `<this slot>.weight` by
    /// flattening through this wrapper — see the converter's `_remap_key`.
    @ModuleInfo public var prelu: PReLU?

    private let kind: SRVGGNetCompact.Activation
    private let leakySlope: Float

    static func build(_ kind: SRVGGNetCompact.Activation, channels: Int) -> ActivationLayer {
        ActivationLayer(kind: kind, channels: channels)
    }

    public init(kind: SRVGGNetCompact.Activation, channels: Int, leakySlope: Float = 0.1) {
        self.kind = kind
        self.leakySlope = leakySlope
        switch kind {
        case .prelu:
            // Upstream: `nn.PReLU(num_parameters=num_feat)` — one learnable
            // alpha per channel. MLXNN.PReLU(count: …) matches.
            // Init value 0.25 mirrors PyTorch's default; the upstream
            // checkpoint overwrites it on load.
            self._prelu.wrappedValue = PReLU(count: channels, value: 0.25)
        case .leakyRelu:
            self._prelu.wrappedValue = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch kind {
        case .prelu:
            // Force-unwrap is safe: prelu is non-nil whenever kind == .prelu.
            return prelu!(x)
        case .leakyRelu:
            return MLXNN.leakyRelu(x, negativeSlope: leakySlope)
        }
    }
}

// MARK: - Factory constructors for the three vendored variants

public extension SRVGGNetCompact {

    /// `realesr-general-x4v3` config: general photos / video at x4.
    ///
    /// num_feat=64, num_conv=32, prelu, ~1.21M params, ~2.4 MB FP16.
    /// The `upscale` arg is provided for forward compatibility — the
    /// vendored weight file ships at x4; other scales would require new
    /// weights.
    static func general(upscale: Int = 4) -> SRVGGNetCompact {
        SRVGGNetCompact(
            numInCh: 3,
            numOutCh: 3,
            numFeat: 64,
            numConv: 32,
            upscale: upscale,
            activation: .prelu
        )
    }

    /// `realesr-general-wdn-x4v3` config: same arch as `general` plus
    /// upstream's WDN (weight denoising) training. Shape-identical to
    /// `general(upscale:)` — separated by name for clarity at the call site.
    static func generalWDN(upscale: Int = 4) -> SRVGGNetCompact {
        SRVGGNetCompact(
            numInCh: 3,
            numOutCh: 3,
            numFeat: 64,
            numConv: 32,
            upscale: upscale,
            activation: .prelu
        )
    }

    /// `realesr-animevideov3` config: half-depth body for real-time anime.
    ///
    /// num_feat=64, num_conv=16, prelu, ~0.62M params, ~1.2 MB FP16.
    static func anime(upscale: Int = 4) -> SRVGGNetCompact {
        SRVGGNetCompact(
            numInCh: 3,
            numOutCh: 3,
            numFeat: 64,
            numConv: 16,
            upscale: upscale,
            activation: .prelu
        )
    }
}

// MARK: - Weight loading

/// Errors raised by SRVGGNetCompact's weight-loading helpers.
public enum SRVGGNetCompactError: Error, Sendable, CustomStringConvertible {
    case weightsNotFound(String)
    case loadFailed(String)

    public var description: String {
        switch self {
        case .weightsNotFound(let path):
            return "SRVGGNetCompact weights file not found: \(path)"
        case .loadFailed(let detail):
            return "SRVGGNetCompact weight load failed: \(detail)"
        }
    }
}

public extension SRVGGNetCompact {

    /// Load weights from a safetensors file produced by
    /// `Packages/ForgeTraining/Scripts/convert_srvggnet_to_mlx.py`.
    ///
    /// The converter is responsible for:
    /// - Transposing Conv2d weights from PyTorch `(O, I, kH, kW)` to MLX
    ///   `(O, kH, kW, I)`.
    /// - Mapping the upstream `body.{N}.{weight,bias}` keys into this
    ///   module's parameter tree (see comment in this file for the full
    ///   index mapping). The converter is the single source of truth — this
    ///   loader just runs the standard MLX-Swift pipeline.
    /// - Eval-ing every output tensor before save (MLX is lazy — see
    ///   `anthropic-skills:mlx-porting` "silent killer").
    ///
    /// - Parameter url: absolute path to a `.safetensors` file with the
    ///   converted weights.
    /// - Throws: `SRVGGNetCompactError.weightsNotFound` if the file is
    ///   missing; `.loadFailed` for any deserialization / shape-mismatch /
    ///   unused-key error from MLX-Swift.
    func loadWeights(from url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SRVGGNetCompactError.weightsNotFound(url.path)
        }

        let arrays: [String: MLXArray]
        do {
            arrays = try MLX.loadArrays(url: url)
        } catch {
            throw SRVGGNetCompactError.loadFailed(String(describing: error))
        }

        // Key remap: the converter emits PReLU weight at `<...>.act.weight`
        // (the upstream PyTorch convention — PReLU is a flat parameter on
        // body.{2k+1}). The Swift parameter tree exposes it at
        // `<...>.act.prelu.weight` because ActivationLayer wraps PReLU as
        // a sub-module. Bridge here so the converter doesn't need to know
        // about the wrapper. `first_act.weight` and every
        // `body_pairs.N.act.weight` get the `.prelu` segment injected.
        // The `(key:)`-overridden `first_act` / `body_pairs` names land us
        // at the right path; we only need to rewrite the trailing
        // `.act.weight` → `.act.prelu.weight`.
        var remapped: [String: MLXArray] = [:]
        remapped.reserveCapacity(arrays.count)
        for (key, value) in arrays {
            // Catches both `first_act.weight` and `body_pairs.N.act.weight`
            // (the only two ActivationLayer slots in the module tree).
            // Conv2d slots end with `.conv.weight` / `first_conv.weight` /
            // `last_conv.weight` — those don't match `act` and pass through.
            if key.hasSuffix(".weight") {
                let prefix = String(key.dropLast(".weight".count))
                if prefix.hasSuffix("act") {
                    remapped["\(prefix).prelu.weight"] = value
                    continue
                }
            }
            remapped[key] = value
        }

        let loaded = ModuleParameters.unflattened(remapped)
        do {
            try update(parameters: loaded, verify: .noUnusedKeys)
        } catch {
            throw SRVGGNetCompactError.loadFailed(String(describing: error))
        }

        MLX.eval(parameters())
    }
}
