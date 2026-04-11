// Fused FP8 quantization + transpose + amax kernel for Lumen.
// Standalone fallback when AITER cannot be modified.
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
#include <hipcub/hipcub.hpp>

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

template <int BLOCK_SIZE = 256>
__device__ float block_reduce_max(float val)
{
    __shared__ float shared[BLOCK_SIZE / 64 + 1];
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;

    for (int offset = 32; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor(val, offset, 64));

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = BLOCK_SIZE / 64;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int offset = 32; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_xor(val, offset, 64));
    }
    return val;
}

#if defined(__gfx942__) || defined(__gfx90a__)
using fp8_t = __hip_fp8_e4m3_fnuz;
static constexpr float FP8_MAX_VAL = 240.0f;
#else
using fp8_t = __hip_fp8_e4m3;
static constexpr float FP8_MAX_VAL = 448.0f;
#endif

template <typename DTYPE_I, int TILE_M = 32, int TILE_N = 32>
__global__ void fused_quant_transpose_amax_kernel(
    fp8_t* __restrict__ out_row,
    fp8_t* __restrict__ out_col,
    float* __restrict__ amax_out,
    DTYPE_I const* __restrict__ input,
    float const* __restrict__ scale,
    int64_t const M,
    int32_t const N)
{
    int const tile_row = blockIdx.x;
    int const tile_col = blockIdx.y;
    int const tid = threadIdx.x;

    int const row_start = tile_row * TILE_M;
    int const col_start = tile_col * TILE_N;
    int const elems_per_tile = TILE_M * TILE_N;
    float const inv_scale = 1.0f / (*scale);

    float thread_amax = 0.0f;
    __shared__ fp8_t smem[TILE_M][TILE_N];

    for (int idx = tid; idx < elems_per_tile; idx += blockDim.x)
    {
        int const local_r = idx / TILE_N;
        int const local_c = idx % TILE_N;
        int const global_r = row_start + local_r;
        int const global_c = col_start + local_c;

        float val = 0.0f;
        if (global_r < M && global_c < N)
            val = static_cast<float>(input[global_r * N + global_c]);

        thread_amax = fmaxf(thread_amax, fabsf(val));

        float scaled = val * inv_scale;
        scaled = fminf(fmaxf(scaled, -FP8_MAX_VAL), FP8_MAX_VAL);
        fp8_t qval = static_cast<fp8_t>(scaled);

        if (global_r < M && global_c < N)
            out_row[global_r * N + global_c] = qval;

        smem[local_r][local_c] = qval;
    }

    __syncthreads();

    for (int idx = tid; idx < elems_per_tile; idx += blockDim.x)
    {
        int const local_r = idx / TILE_N;
        int const local_c = idx % TILE_N;
        int const global_r = row_start + local_r;
        int const global_c = col_start + local_c;

        if (global_r < M && global_c < N)
            out_col[global_c * M + global_r] = smem[local_r][local_c];
    }

    thread_amax = block_reduce_max<256>(thread_amax);
    if (tid == 0)
        atomicMaxFloat(amax_out, thread_amax);
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

    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;

    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    dim3 block(256);

    if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        fused_quant_transpose_amax_kernel<__hip_bfloat16, TILE_M, TILE_N>
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
        fused_quant_transpose_amax_kernel<__half, TILE_M, TILE_N>
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
