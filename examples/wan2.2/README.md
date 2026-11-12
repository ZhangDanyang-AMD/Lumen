# Wan2.2-TI2V-5B VAE (SimWAM NAVSIM) on Lumen's aiter conv

SimWAM's Wan2.2 VAE, `WanVideoVAE38`, with its convolutions patched by
`LumenConfig(aiter_conv=True).enable(vae)`, at exactly SimWAM NAVSIM's shapes. Plain torch; no VeOmni.

| | shape |
|---|---|
| input video | (1, 3, 9, 384, 672) |
| VAE latent | (1, 48, 3, 24, 42): time (9−1)/4+1 = 3, space 384/16 = 24, 672/16 = 42 |
| decode output | (1, 3, 9, 384, 672) |
| DiT patchify [1, 2, 2] | 12×21 tokens per latent frame, 756 in all |

The VAE is Wan2.2-TI2V-5B's: z_dim 48, 16× spatial (patchify 2 and three 2× downsamples), 4× temporal, 0.70B
parameters (704,688,668), 62 `CausalConv3d` and 10 `nn.Conv2d`. One encode makes 92 convolution calls, one decode 116.

| file | |
|---|---|
| `wan22_vae.py` | loads SimWAM's `wan_video_vae.py` from a checkout (it needs only torch, einops, tqdm) and the weights (strict); NAVSIM shapes; `ConvCalls`, which counts each convolution call as aiter or torch; the per-convolution exact check |
| `verify_vae.py` | single GPU: encode and decode, stock vs patched BF16 at rtol = atol = 1e-5, aiter/torch call counts, SNR against FP32, ms; then the per-convolution check |
| `test_wan22_vae.py` | the same as tests: CPU (structure, shapes, which modules are bound) and GPU (the 1e-5 comparisons) |

## What the patch does

`lumen.ops.conv.patch_conv` rebinds `_conv_forward` of every supported `nn.Conv2d` / `nn.Conv3d`, subclasses
included, so `CausalConv3d.forward` -- causal pad and feature cache -- is kept and only the convolution under it moves
to aiter's Triton `conv2d` / `conv3d`. The forward goes to aiter only for CUDA fp16/bf16 input; fp32, CPU and a kernel
that raises fall back to torch, with at most one warning per module. So the VAE has to run in BF16, and the call counts
are what tell a real aiter run from a silent fallback. Only the VAE is patched; the DiT's `patch_embedding`
(`Conv3d(48, 3072, k=s=(1, 2, 2))`) is out of scope.

## Correctness at rtol = atol = 1e-5

aiter's Triton conv accepts fp16/bf16 only, so the comparison is BF16 on the GPU, through the aiter path, with every
call counted (a fallback fails the check):

1. **Encode and decode, patched vs stock BF16** (`test_bf16_encode_matches_stock`, `test_bf16_decode_matches_stock`):
   the reviewer's end-to-end criterion as written. Stock BF16 is MIOpen; patched is aiter. Both round each layer's fp32
   accumulator to BF16, so any difference in summation order or algorithm can move an element by a BF16 ulp
   (2⁻⁸ relative), and the network carries it on. `verify_vae.py` reports max|d| and how many elements are outside the
   tolerance as well as PASS/FAIL.
2. **Every distinct NAVSIM convolution, exact** (`test_aiter_kernels_exact_on_navsim_conv_shapes`): the 58 shapes one
   encode and one decode give aiter, on {-1, 0, 1} BF16 input, weights and bias. Every product and partial sum is then
   an integer below 2²⁴, so a direct, GEMM or implicit-GEMM kernel with an fp32 accumulator gets the exact sum and the
   same BF16 rounding as a CPU fp32 reference: anything but bit equality is a kernel error. This covers aiter's actual
   kernels at the actual shapes. It does not cover Winograd routes: their transforms have fractional constants and,
   for BF16 input, an fp16 buffer, so they cannot be exact; those shapes are reported with their max|d| and not
   asserted on. It also says nothing about real-valued data beyond what (1) does.

On CPU the tests check the structure, the shapes, the conv trace (92 / 116 calls, 8.40 / 43.85 TFLOP, the same as
SimWAM's own) and that all 72 convolutions are bound with `CausalConv3d.forward` untouched. They do not check numerics:
on CPU every call is torch either way.

## Run

```bash
git clone https://github.com/H-EmbodVis/SimWAM.git $SIMWAM_DIR
git -C $SIMWAM_DIR checkout 1433ee7e06a62bd8b89da74455df0d82bda7b726
hf download Wan-AI/Wan2.2-TI2V-5B --include Wan2.2_VAE.pth --local-dir $(dirname $WAN22_VAE)

export PYTHONPATH=/path/to/Lumen
python3 verify_vae.py
python3 -m pytest -q test_wan22_vae.py
```

`SIMWAM_DIR` defaults to `/work/SimWAM`, `WAN22_VAE` to `/work/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`. SimWAM's own
loader reads DiffSynth-Studio's `Wan2.2_VAE.safetensors` by default; `wan22_vae.load` takes either.

aiter needs Triton conv3d: `aiter/ops/triton/conv/conv3d.py` (aiter ≥ `c8325e00c`). With an older install that has
`aiter.ops.triton.conv` but not `conv3d.py`, point `AITER_SRC` at a newer aiter checkout and Lumen loads conv3d from
there. Without it, every Conv3d stays on torch and the GPU tests skip.
