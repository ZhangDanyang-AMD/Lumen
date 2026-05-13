// Fused FP8 quantization + transpose + amax kernel for Lumen (HIP/ROCm).
//
// Reads BF16/FP16 input (M, N) once and writes:
//   - out_row (M, N) in FP8 row-major
//   - out_col (N, M) in FP8 (transposed)
//   - amax    (1,)   float32 max(abs(input))

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

namespace lumen {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value)
{
    float old;
    old = (value >= 0)
              ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
              : __uint_as_float(
                    atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

// MI300X (gfx942) uses e4m3_fnuz; detect at compile time via USE_ROCM.
// For CUDA builds, fall back to e4m3.
#if defined(USE_ROCM) || defined(USE_FP8_FNUZ) || defined(__gfx942__) || defined(__gfx90a__)
using fp8_t = __hip_fp8_e4m3_fnuz;
static constexpr float FP8_MAX_VAL = 240.0f;
#else
using fp8_t = __hip_fp8_e4m3;
static constexpr float FP8_MAX_VAL = 448.0f;
#endif

__device__ __forceinline__ float to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

// 64x64 tile, 256 threads. Each thread loads 16 elements.
// Shared memory is padded (+1 column) to avoid bank conflicts on transpose read.
template <typename DTYPE_I>
__global__ void fused_quant_transpose_amax_kernel_v2(
    fp8_t* __restrict__ out_row,
    fp8_t* __restrict__ out_col,
    float* __restrict__ amax_out,
    DTYPE_I const* __restrict__ input,
    float const* __restrict__ scale,
    int64_t const M,
    int32_t const N)
{
    constexpr int TILE = 64;
    constexpr int BLOCK = 256;
    constexpr int ELEMS = TILE * TILE;
    constexpr int ITERS = ELEMS / BLOCK;  // 16

    int const tile_row = blockIdx.x;
    int const tile_col = blockIdx.y;
    int const tid = threadIdx.x;

    int const row_start = tile_row * TILE;
    int const col_start = tile_col * TILE;
    float const inv_scale = 1.0f / (*scale);

    // +1 padding to avoid shared memory bank conflicts during column reads
    __shared__ fp8_t smem[TILE][TILE + 1];

    float thread_amax = 0.0f;

    // Phase 1: load, quantize, write row-major, store to shared memory
    #pragma unroll
    for (int i = 0; i < ITERS; ++i)
    {
        int idx = tid + i * BLOCK;
        int lr = idx / TILE;
        int lc = idx % TILE;
        int gr = row_start + lr;
        int gc = col_start + lc;

        float val = 0.0f;
        if (gr < M && gc < N)
            val = to_float(input[gr * N + gc]);

        thread_amax = fmaxf(thread_amax, fabsf(val));

        float scaled = fminf(fmaxf(val * inv_scale, -FP8_MAX_VAL), FP8_MAX_VAL);
        fp8_t qval = static_cast<fp8_t>(scaled);

        if (gr < M && gc < N)
            out_row[gr * N + gc] = qval;

        smem[lr][lc] = qval;
    }

    __syncthreads();

    // Phase 2: read from shared memory in transposed order, write column-major
    #pragma unroll
    for (int i = 0; i < ITERS; ++i)
    {
        int idx = tid + i * BLOCK;
        int lr = idx / TILE;  // now iterating as (col_local, row_local) for coalesced write
        int lc = idx % TILE;
        int gr = row_start + lc;
        int gc = col_start + lr;

        if (gr < M && gc < N)
            out_col[gc * M + gr] = smem[lc][lr];
    }

    // Warp-level reduction for amax
    for (int offset = 32; offset > 0; offset >>= 1)
        thread_amax = fmaxf(thread_amax, __shfl_xor(thread_amax, offset, 64));

    // Per-warp amax into shared (reuse smem as float array)
    __shared__ float warp_max[4];
    int lane = tid % 64;
    int wid = tid / 64;
    if (lane == 0) warp_max[wid] = thread_amax;
    __syncthreads();

    if (tid < 4)
    {
        float val = warp_max[tid];
        for (int offset = 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_xor(val, offset, 64));
        if (tid == 0)
            atomicMaxFloat(amax_out, val);
    }
}

void static_quant_transpose_amax(
    torch::Tensor& out_row,
    torch::Tensor& out_col,
    torch::Tensor& amax_out,
    torch::Tensor const& input,
    torch::Tensor const& scale)
{
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");

    int64_t const M = input.size(0);
    int32_t const N = input.size(1);

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    constexpr int TILE = 64;
    dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 block(256);

    if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        fused_quant_transpose_amax_kernel_v2<__hip_bfloat16>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<fp8_t*>(out_row.data_ptr()),
                reinterpret_cast<fp8_t*>(out_col.data_ptr()),
                amax_out.data_ptr<float>(),
                reinterpret_cast<__hip_bfloat16 const*>(input.data_ptr()),
                scale.data_ptr<float>(),
                M, N);
    }
    else if (input.scalar_type() == at::ScalarType::Half)
    {
        fused_quant_transpose_amax_kernel_v2<__half>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<fp8_t*>(out_row.data_ptr()),
                reinterpret_cast<fp8_t*>(out_col.data_ptr()),
                amax_out.data_ptr<float>(),
                reinterpret_cast<__half const*>(input.data_ptr()),
                scale.data_ptr<float>(),
                M, N);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());
    }
}

} // namespace lumen

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("static_quant_transpose_amax",
          &lumen::static_quant_transpose_amax,
          py::arg("out_row"),
          py::arg("out_col"),
          py::arg("amax_out"),
          py::arg("input"),
          py::arg("scale"));
}
