#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t, int BLOCK_SIZE>
__global__ void rmsnorm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    const int64_t rows,
    const int64_t hidden_size,
    const float eps
) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const int64_t base = row * hidden_size;

    __shared__ float ssum[BLOCK_SIZE];

    float local_sum = 0.0f;

    // Step 1: 每个 thread 负责 hidden_size 中的一部分元素，计算平方和
    for (int64_t i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float v = static_cast<float>(x[base + i]);
        local_sum += v * v;
    }

    ssum[tid] = local_sum;
    __syncthreads();

    // Step 2: block 内 shared memory reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(ssum[0] / static_cast<float>(hidden_size) + eps);

    // Step 3: 归一化并乘 weight，融合在同一个 kernel 中
    for (int64_t i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float v = static_cast<float>(x[base + i]);
        float w = static_cast<float>(weight[i]);
        float y = v * inv_rms * w;
        out[base + i] = static_cast<scalar_t>(y);
    }
}


template <typename scalar_t>
void launch_rmsnorm_kernel(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    torch::Tensor& out,
    const int64_t rows,
    const int64_t hidden_size,
    const float eps,
    const int block_size,
    cudaStream_t stream
) {
    const dim3 blocks(rows);

    if (block_size == 128) {
        rmsnorm_forward_kernel<scalar_t, 128><<<blocks, 128, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            rows,
            hidden_size,
            eps
        );
    } else if (block_size == 256) {
        rmsnorm_forward_kernel<scalar_t, 256><<<blocks, 256, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            rows,
            hidden_size,
            eps
        );
    } else if (block_size == 512) {
        rmsnorm_forward_kernel<scalar_t, 512><<<blocks, 512, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            rows,
            hidden_size,
            eps
        );
    } else {
        TORCH_CHECK(false, "block_size must be one of 128, 256, or 512");
    }
}


torch::Tensor rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps,
    int64_t block_size
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight must have the same dtype");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(block_size == 128 || block_size == 256 || block_size == 512,
                "block_size must be 128, 256, or 512");

    const auto hidden_size = weight.numel();
    TORCH_CHECK(x.size(-1) == hidden_size, "last dimension of x must match weight size");

    const int64_t rows = x.numel() / hidden_size;

    auto out = torch::empty_like(x);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "rmsnorm_forward_cuda",
        [&] {
            launch_rmsnorm_kernel<scalar_t>(
                x,
                weight,
                out,
                rows,
                hidden_size,
                static_cast<float>(eps),
                static_cast<int>(block_size),
                stream
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
