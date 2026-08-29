/*
Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt fused_residual_rmsnorm_backward.cu -o fused_residual_rmsnorm_backward
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math fused_residual_rmsnorm_backward.cu -o fused_residual_rmsnorm_backward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./rmsnorm_backward 1

*/

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <assert.h>

#include "common.h"

constexpr float kEps = 1e-5f;

void fused_residual_rmsnorm_forward_cpu(
    float* y,
    float* z,
    float* mean2,
    const float* x,
    const float* residual,
    const float* w,
    int N,
    int C
){
    for(int row = 0; row < N; ++row){
        const float* xr = x + (size_t)row * C;
        const float* residualr = residual + (size_t)row * C;
        float* yr = y + (size_t)row * C;
        float* zr = z + (size_t)row * C;

        float sum = 0.0f;
        for(int c = 0; c < C; ++c){
            zr[c] = xr[c] + residualr[c];
            sum += zr[c] * zr[c];
        }
        float m2 = sum / C;
        mean2[row] = m2;
        float inv = 1.0f / std::sqrt(m2 + kEps);
        for(int c= 0; c < C; ++c) yr[c] = zr[c] * w[c] * inv;
    }
}

void fused_residual_rmsnorm_backward_cpu(
    float* dx,
    float* dresidual,
    float* dw,
    const float* dy,
    const float* z,
    const float* weight,
    const float* mean2,
    int N,
    int C
){
    std::fill(dw, dw + C, 0.0f);
    for(int row = 0; row < N; ++row){
        const float* zr = z + (size_t)row * C;
        const float* dyr = dy + (size_t)row * C;
        float* dxr = dx + (size_t)row * C;
        float* dresidualr = dresidual + (size_t)row * C; 

        float sum = 0.0f;
        for(int c = 0; c < C; ++c){
            sum += dyr[c] * weight[c] * zr[c];
        }

        float inv = 1.0f / std::sqrt(mean2[row] + kEps);
        float correction = sum * inv * inv / C;

        for(int c = 0; c < C; ++c){
            float v = dyr[c] * weight[c] * inv - zr[c] * inv * correction;
            dxr[c] = v;
            dresidualr[c] = v;
            dw[c] += dyr[c] * zr[c] * inv;
        }
    }
}

// naive kernel
__global__ void fused_residual_rmsnorm_backward_kernel1(
    float* dx,
    float* dresidual,
    float* dw,
    const float* dy,
    const float* z,
    const float* weight,
    const float* mean2,
    int N,
    int C
){
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= N) return;

    const float* zr = z + (size_t)row * C;
    const float* dyr = dy + (size_t)row * C;
    float* dxr = dx + (size_t)row * C;
    float* dresidualr = dresidual + (size_t)row * C; 

    // 第一步求sum = dy * z * w
    float sum = 0.0f;
    for(int c = 0; c < C; ++c){
        sum += dyr[c] * zr[c] * weight[c];
    }
 
    float inv = rsqrtf(mean2[row] + kEps);
    float correction = sum * inv * inv / C;

    for(int c = 0; c < C; ++c){
        float v = dyr[c] * weight[c] * inv - zr[c] * inv * correction;
        dxr[c] = v;
        dresidualr[c] = v;
        atomicAdd(&dw[c], dyr[c] * zr[c] * inv);
    }
}

// warp
__global__ void fused_residual_rmsnorm_backward_kernel2(
    float* dx,
    float* dresidual,
    float* dw,
    const float* dy,
    const float* z,
    const float* weight,
    const float* mean2,
    int N,
    int C
){
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    int row = blockIdx.x * (blockDim.x / 32) + warp_id;

    if(row >= N) return;

    const float* zr = z + (size_t)row * C;
    const float* dyr = dy + (size_t)row * C;
    float* dxr = dx + (size_t)row * C;
    float* dresidualr = dresidual + (size_t)row * C; 

    // 第一步求sum = dy * z * w
    float sum = 0.0f;
    for(int c = lane; c < C; c+=32){
        sum += dyr[c] * zr[c] * weight[c];
    }

    for(int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);


    float inv = rsqrtf(mean2[row] + kEps);
    float correction = sum * inv * inv / C;

    for(int c = lane; c < C; c+= 32){
        float v = dyr[c] * weight[c] * inv - zr[c] * inv * correction;
        dxr[c] = v;
        dresidualr[c] = v;
        atomicAdd(&dw[c], dyr[c] * zr[c] * inv);
    }
}

// weight 共享内存加载
__global__ void fused_residual_rmsnorm_backward_kernel3(
    float* dx,
    float* dresidual,
    float* dw,
    const float* dy,
    const float* z,
    const float* weight,
    const float* mean2,
    int N,
    int C
){
    extern __shared__ float shared_dw[];

    for(int c = threadIdx.x; c < C; c+= blockDim.x){
        shared_dw[c] = 0.0f;
    }
    __syncthreads();

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    int row = blockIdx.x * (blockDim.x / 32) + warp_id;

    if(row < N){
        const float* zr = z + (size_t)row * C;
        const float* dyr = dy + (size_t)row * C;
        float* dxr = dx + (size_t)row * C;
        float* dresidualr = dresidual + (size_t)row * C;

        // 第一步求sum = dy * z * w
        float sum = 0.0f;
        for(int c = lane; c < C; c+=32){
            sum += dyr[c] * zr[c] * weight[c];
        }

        for(int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);

        float inv = rsqrtf(mean2[row] + kEps);
        float correction = sum * inv * inv / C;

        for(int c = lane; c < C; c+= 32){
            float v = dyr[c] * weight[c] * inv - zr[c] * inv * correction;
            dxr[c] = v;
            dresidualr[c] = v;
            atomicAdd(&shared_dw[c], dyr[c] * zr[c] * inv);
        }
    }

    __syncthreads();
    for(int c = threadIdx.x; c < C; c += blockDim.x)
        atomicAdd(&dw[c], shared_dw[c]);
}

// kernel3的基础上 persistent grid-loop
__global__ void fused_residual_rmsnorm_backward_kernel4(
    float* dx,
    float* dresidual,
    float* dw,
    const float* dy,
    const float* z,
    const float* weight,
    const float* mean2,
    int N,
    int C
){
    extern __shared__ float shared_dw[];

    for(int c = threadIdx.x; c < C; c+= blockDim.x){
        shared_dw[c] = 0.0f;
    }

    __syncthreads();

    int warps = blockDim.x / 32;   // 每个 block 中 warp 的数量
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int first_row  = blockIdx.x * warps + warp_id;   // 第一个负责的行
    int row_stride = gridDim.x * warps;              // 每次前进的行数
    for (long row = first_row; row < N; row += row_stride) {
        const float* zr = z + (size_t)row * C;
        const float* dyr = dy + (size_t)row * C;
        float* dxr = dx + (size_t)row * C;
        float* dresidualr =dresidual + (size_t)row * C;

        // 第一步求sum = dy * x * w
        float sum = 0.0f;
        for(int c = lane; c < C; c+=32){
            sum += dyr[c] * zr[c] * weight[c];
        }

        for(int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);

        float inv = rsqrtf(mean2[row] + kEps);
        float correction = sum * inv * inv / C;

        for(int c = lane; c < C; c+= 32){
            float v = dyr[c] * weight[c] * inv - zr[c] * inv * correction;
            dxr[c] = v;
            dresidualr[c] = v;
            atomicAdd(&shared_dw[c], dyr[c] * zr[c] * inv);
        }
    }

    __syncthreads();
    for(int c = threadIdx.x; c < C; c += blockDim.x)
        atomicAdd(&dw[c], shared_dw[c]);
}

// kernel4的基础上 vec2
__global__ void fused_residual_rmsnorm_backward_kernel5(
    float* dx,
    float* dresidual,
    float* dw,
    const float* dy,
    const float* z,
    const float* w,
    const float* mean2,
    int N,
    int C
){
    extern __shared__ float shared_dw[];

    for(int c = threadIdx.x; c < C; c+= blockDim.x){
        shared_dw[c] = 0.0f;
    }

    __syncthreads();

    int warps = blockDim.x / 32;   // 每个 block 中 warp 的数量
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int first_row  = blockIdx.x * warps + warp_id;   // 第一个负责的行
    int row_stride = gridDim.x * warps;              // 每次前进的行数
    for (long row = first_row; row < N; row += row_stride) {
        const float* zr = z + (size_t)row * C;
        const float* dyr = dy + (size_t)row * C;
        float* dxr = dx + (size_t)row * C;
        float* dresidualr = dresidual + (size_t)row * C;

        // 第一步求sum = dy * x * w
        float sum = 0.0f;
        for(int c = lane * 2; c < C; c+=32 * 2){
            const float2 dyr2 = *reinterpret_cast<const float2*>(dyr + c);
            const float2 zr2 = *reinterpret_cast<const float2*>(zr + c);
            const float2 w2 = *reinterpret_cast<const float2*>(w + c);

            sum += (dyr2.x * zr2.x * w2.x + dyr2.y * zr2.y * w2.y);
        }

        for(int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);

        float inv = rsqrtf(mean2[row] + kEps);
        float correction = sum * inv * inv / C;

        for(int c = lane * 2; c < C; c+= 32 * 2){
            const float2 dyr2 = *reinterpret_cast<const float2*>(dyr + c);
            const float2 zr2 = *reinterpret_cast<const float2*>(zr + c);
            const float2 w2 = *reinterpret_cast<const float2*>(w + c);
            float2 tmp;
            tmp.x = dyr2.x * w2.x * inv - zr2.x * inv * correction;
            tmp.y = dyr2.y * w2.y * inv - zr2.y * inv * correction;
            *reinterpret_cast<float2*>(dxr + c) = tmp;
            *reinterpret_cast<float2*>(dresidualr + c) = tmp;

            atomicAdd(&shared_dw[c],     dyr2.x * zr2.x * inv);
            atomicAdd(&shared_dw[c + 1], dyr2.y * zr2.y * inv);
        }
    }

    __syncthreads();
    for(int c = threadIdx.x; c < C; c += blockDim.x)
        atomicAdd(&dw[c], shared_dw[c]);
}

// ----------------------------------------------------------------------------
// kernel launchers

void fused_residual_rmsnorm_backward1(
    float* dx, 
    float* dresidual,
    float* dw, 
    const float* dy, 
    const float* z, 
    const float* w, 
    const float* mean2, 
    int B, 
    int T, 
    int C, 
    const int block_size
){
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    fused_residual_rmsnorm_backward_kernel1<<<grid_size, block_size>>>(dx, dresidual, dw, dy, z, w, mean2, N, C);
    CUDA_CHECK(cudaGetLastError());
}

void fused_residual_rmsnorm_backward2(
    float* dx, 
    float* dresidual,
    float* dw, 
    const float* dy, 
    const float* z, 
    const float* w, 
    const float* mean2, 
    int B, 
    int T, 
    int C, 
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    fused_residual_rmsnorm_backward_kernel2<<<grid_size, block_size>>>(dx, dresidual, dw, dy, z, w, mean2, N, C);
    CUDA_CHECK(cudaGetLastError());
}

void fused_residual_rmsnorm_backward3(
    float* dx, 
    float* dresidual,
    float* dw, 
    const float* dy, 
    const float* z, 
    const float* w, 
    const float* mean2, 
    int B, 
    int T, 
    int C, 
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T;

    const int grid_size = ceil_div(N * 32, block_size);
    size_t smem_size = C * sizeof(float);
    cudaFuncSetAttribute(fused_residual_rmsnorm_backward_kernel3,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    fused_residual_rmsnorm_backward_kernel3<<<grid_size, block_size, smem_size>>>(dx, dresidual, dw, dy, z, w, mean2, N, C);
    CUDA_CHECK(cudaGetLastError());
}

void fused_residual_rmsnorm_backward4(
    float* dx, 
    float* dresidual,
    float* dw, 
    const float* dy, 
    const float* z, 
    const float* w, 
    const float* mean2, 
    int B, 
    int T, 
    int C, 
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T;

    size_t smem_size = C * sizeof(float);
    cudaFuncSetAttribute(fused_residual_rmsnorm_backward_kernel4,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, fused_residual_rmsnorm_backward_kernel4, block_size, C * sizeof(float));
    int grid_size = sm_count * max_blocks_per_sm;
    // 限制 grid_size 不超过 ceil(N / (block_size/32))，因为每个 warp 处理一行
    int max_needed = (N + (block_size/32) - 1) / (block_size/32);
    grid_size = std::min(grid_size, max_needed);

    fused_residual_rmsnorm_backward_kernel4<<<grid_size, block_size, smem_size>>>(dx, dresidual, dw, dy, z, w, mean2, N, C);
    CUDA_CHECK(cudaGetLastError());
}

void fused_residual_rmsnorm_backward5(
    float* dx, 
    float* dresidual,
    float* dw, 
    const float* dy, 
    const float* z, 
    const float* w, 
    const float* mean2, 
    int B, 
    int T, 
    int C, 
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T;

    // const int grid_size = ceil_div(N * 32, block_size);
    size_t smem_size = C * sizeof(float);
    cudaFuncSetAttribute(fused_residual_rmsnorm_backward_kernel5,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, fused_residual_rmsnorm_backward_kernel5, block_size, C * sizeof(float));
    int grid_size = sm_count * max_blocks_per_sm;
    // 限制 grid_size 不超过 ceil(N / (block_size/32))，因为每个 warp 处理一行
    int max_needed = (N + (block_size/32) - 1) / (block_size/32);
    grid_size = std::min(grid_size, max_needed);

    bool use_vec2 = (C % 2 == 0) &&
                    is_aligned(z, 8) && is_aligned(w, 8) && is_aligned(dy, 8) &&
                    is_aligned(dx, 8) && is_aligned(dresidual, 8);
    if (use_vec2) {
        fused_residual_rmsnorm_backward_kernel5<<<grid_size, block_size, smem_size>>>(dx, dresidual, dw, dy, z, w, mean2, N, C);
    } else {
        fused_residual_rmsnorm_backward_kernel4<<<grid_size, block_size, smem_size>>>(dx, dresidual, dw, dy, z, w, mean2, N, C);
    }
    CUDA_CHECK(cudaGetLastError());
}

void fused_residual_rmsnorm_backward(
    int kernel_num,
    float* dx,
    float* dresidual, 
    float* dw, 
    const float* dy, 
    const float* z, 
    const float* w, 
    const float* mean2, 
    int B, 
    int T, 
    int C, 
    const int block_size
){
    switch(kernel_num){
        case 1:
            fused_residual_rmsnorm_backward1(dx, dresidual, dw, dy, z, w, mean2, B, T, C, block_size);
            break;
        case 2:
            fused_residual_rmsnorm_backward2(dx, dresidual, dw, dy, z, w, mean2, B, T, C, block_size);
            break;
        case 3:
            fused_residual_rmsnorm_backward3(dx, dresidual, dw, dy, z, w, mean2, B, T, C, block_size);
            break;
        case 4:
            fused_residual_rmsnorm_backward4(dx, dresidual, dw, dy, z, w, mean2, B, T, C, block_size);
            break;
        case 5:
            fused_residual_rmsnorm_backward5(dx, dresidual, dw, dy, z, w, mean2, B, T, C, block_size);
            break;
        default:
            std::printf("Invalid kernel number.\n");
            exit(1);
    }
}

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 1600;   

    // first do the forward pass in CPU
    float* y = (float*)malloc(B * T * C * sizeof(float));
    float* z = (float*)malloc(B * T * C * sizeof(float));
    float* mean2 = (float*)malloc(B * T * sizeof(float));

    float* x = make_random_float(B * T * C);
    float* residual = make_random_float(B * T * C);
    float* w = make_random_float(C);

    fused_residual_rmsnorm_forward_cpu(y, z, mean2, x, residual, w, B * T, C);

    // now do the backward pass, again on CPU
    float* dy = make_random_float(B * T * C);
    float* dx = make_zeros_float(B * T * C);
    float* dresidual = make_zeros_float(B * T * C);
    float* dw = make_zeros_float(C);

    fused_residual_rmsnorm_backward_cpu(dx, dresidual, dw, dy, z, w, mean2, B * T, C);

    // the above calculations act as the reference
    // now let's do the same on the GPU

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // move all the variables we need for backward pass onto the GPU
    float* d_dx;
    float* d_dresidual;
    float* d_dw;
    float* d_dy;
    float* d_z;
    float* d_w;
    float* d_mean2;

    CUDA_CHECK(cudaMalloc(&d_dx, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dresidual, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dy, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean2, B * T * sizeof(float)));

    // copy over the "inputs" to the backward call
    CUDA_CHECK(memcpy_convert(d_dy, dy, B * T * C));
    CUDA_CHECK(memcpy_convert(d_z, z, B * T * C));
    CUDA_CHECK(memcpy_convert(d_w, w, C));
    CUDA_CHECK(memcpy_convert(d_mean2, mean2, B * T));


    // launch the kernel
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        // init the "outputs" of the backward call to zeros
        CUDA_CHECK(cudaMemset(d_dx, 0, B * T * C * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dresidual, 0, B * T * C * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dw, 0, C * sizeof(float)));

        fused_residual_rmsnorm_backward(kernel_num, d_dx, d_dresidual, d_dw, d_dy, d_z, d_w, d_mean2, 
                           B, T, C, block_size);

        // check the correctness of the kernel
        float error_threshold_dinp = sizeof(float) == 4 ? 1e-3f : 1e-1f; // allow larger errors for BF16/FP16
        float error_threshold_dparams = sizeof(float) == 4 ? 2e-3f : 5e-1f; // much, much larger...
        printf("Checking correctness...\n");
        printf("dx:\n");
        validate_result(d_dx, dx, "dx", B * T * C, error_threshold_dinp);
        printf("dw:\n");
        validate_result(d_dw, dw, "dw", C, error_threshold_dparams);


        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now time the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, fused_residual_rmsnorm_backward, kernel_num,
                                              d_dx, d_dresidual, d_dw, d_dy, d_z, d_w, d_mean2, 
                                              B, T, C, block_size);

        long memory_ops = (4 * B * T * C + B * T + 2 * C) * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);                                      
        // printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanups
    free(y);
    free(z);
    free(mean2);
    free(x);
    free(residual);
    free(dresidual);
    free(w);
    free(dy);
    free(dx);
    free(dw);

    CUDA_CHECK(cudaFree(d_dx));
    CUDA_CHECK(cudaFree(d_dresidual));
    CUDA_CHECK(cudaFree(d_dw));
    CUDA_CHECK(cudaFree(d_dy));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_mean2));

    return 0;
}
