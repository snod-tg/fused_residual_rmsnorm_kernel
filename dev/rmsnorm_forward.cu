/*
Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt rmsnorm_forward.cu -o rmsnorm_forward
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math rmsnorm_forward.cu -o rmsnorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./rmsnorm_forward 1

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

void rmsnorm_forward_cpu(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int N,
    int C
){
    for(int row = 0; row < N; ++row){
        const float* xr = x + (size_t)row * C;
        float* yr = y + (size_t)row * C;
        float sum = 0.0f;
        for(int c = 0; c < C; ++c) sum += xr[c] * xr[c];
        float m2 = sum / C;
        mean2[row] = m2;
        float inv = 1.0f / std::sqrt(m2 + kEps);
        for(int c= 0; c < C; ++c) yr[c] = xr[c] * w[c] * inv;
    }
}

// 对照cpu实现，一个线程对应一行row
__global__ void rmsnorm_forward_kernel1(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int N,
    int C
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= N) return;

    const float* xr = x + (size_t)row * C;
    float* yr = y + (size_t)row * C;

    float sum = 0.0f;
    for(int c = 0; c < C; ++c){
        float v = xr[c];
        sum += v * v;
    }

    float m2 = sum / C;
    mean2[row] = m2;
    float inv = rsqrt(m2 + kEps);

    for(int c = 0; c < C; ++c){
        yr[c] = xr[c] * w[c] * inv;
    }
}

// 一个线程束（warp） 对应 一个行row
__global__ void rmsnorm_forward_kernel2(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int N,
    int C
){
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    int row = blockIdx.x * (blockDim.x / 32) + warp_id;

    if(row >= N) return;

    const float* xr = x + (size_t)row * C;
    float* yr = y + (size_t)row * C;

    float sum = 0.0f;
    for(int c = lane; c < C; c+=32){
        float v = xr[c];
        sum += v * v;
    }

    for(int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
 

    float m2 = sum / C;
    if(lane == 0) mean2[row] = m2;

    float inv = rsqrt(m2 + kEps);

    for(int c = lane; c < C; c+= 32){
        yr[c] = xr[c] * w[c] * inv;
    }
}

// 每次读取2个元素
__global__ void rmsnorm_forward_kernel3(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int N,
    int C
){
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    int row = blockIdx.x * (blockDim.x / 32) + warp_id;

    if(row >= N) return;

    const float* xr = x + (size_t)row * C;
    float* yr = y + (size_t)row * C;

    float sum = 0.0f;
    for(int c = lane * 2; c < C; c+=32 * 2){
        float2 v = *reinterpret_cast<const float2*>(xr + c);
        sum += v.x * v.x + v.y * v.y;
    }

    for(int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
 

    float m2 = sum / C;
    if(lane == 0) mean2[row] = m2;
    float inv = rsqrt(m2 + kEps);

    for(int c = lane * 2; c < C; c+= 32 * 2){
        float2 xv = *reinterpret_cast<const float2*>(xr + c);
        float2 wv = *reinterpret_cast<const float2*>(w + c);
        float2 r;
        r.x = xv.x * wv.x * inv;
        r.y = xv.y * wv.y * inv;
        *reinterpret_cast<float2*>(yr + c) = r;
    }
}

// 每次读取4个元素
__global__ void rmsnorm_forward_kernel4(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int N,
    int C
){
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    int row = blockIdx.x * (blockDim.x / 32) + warp_id;

    if(row >= N) return;

    const float* xr = x + (size_t)row * C;
    float* yr = y + (size_t)row * C;

    float sum = 0.0f;
    for(int c = lane * 4; c < C; c+=32 * 4){
        float4 v = *reinterpret_cast<const float4*>(xr + c);
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    for(int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
 

    float m2 = sum / C;
    if(lane == 0) mean2[row] = m2;
    float inv = rsqrt(m2 + kEps);

    for(int c = lane * 4; c < C; c+= 32 * 4){
        float4 xv = *reinterpret_cast<const float4*>(xr + c);
        float4 wv = *reinterpret_cast<const float4*>(w + c);
        float4 r;
        r.x = xv.x * wv.x * inv;
        r.y = xv.y * wv.y * inv;
        r.z = xv.z * wv.z * inv;
        r.w = xv.w * wv.w * inv;
        *reinterpret_cast<float4*>(yr + c) = r;
    }
}

// 在kernel2的基础上使用共享内存加载weight
__global__ void rmsnorm_forward_kernel5(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int N,
    int C
){

    extern __shared__  float shared_w[];

    for(int c = threadIdx.x; c < C; c+= blockDim.x){
        shared_w[c] = w[c];
    }

    __syncthreads();

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    int row = blockIdx.x * (blockDim.x / 32) + warp_id;

    if(row >= N) return;

    const float* xr = x + (size_t)row * C;
    float* yr = y + (size_t)row * C;

    float sum = 0.0f;
    for(int c = lane; c < C; c+=32){
        float v = xr[c];
        sum += v * v;
    }

    for(int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
 

    float m2 = sum / C;
    if(lane == 0) mean2[row] = m2;

    float inv = rsqrt(m2 + kEps);

    for(int c = lane; c < C; c+= 32){
        yr[c] = xr[c] * shared_w[c] * inv;
    }
}

// Grid-Stride + 尾部处理 + __ldg
__global__ void rmsnorm_forward_kernel6(
    float* __restrict__ y, 
    float* __restrict__ mean2,
    const float* __restrict__ x, 
    const float* __restrict__ w,
    int N, 
    int C
){
    extern __shared__ float shared_w[];  // 缓存 w

    // 加载 w 到共享内存（每个 block 一次）
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        shared_w[i] = __ldg(&w[i]);

    __syncthreads();

    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int stride = gridDim.x * (blockDim.x / 32);

    for (int row = warp_id_global; row < N; row += stride) {
        const float* xr = x + (size_t)row * C;
        float* yr = y + (size_t)row * C;
        int lane = threadIdx.x % 32;

        // 计算平方和（向量化 + 尾部处理）
        float sum = 0.0f;
        int c = lane * 4;
        int vec_end = (C / 4) * 4;
        for (; c < vec_end; c += 32 * 4) {
            float4 v = *reinterpret_cast<const float4*>(&xr[c]);  // 确保对齐
            sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
        for (c = vec_end + lane; c < C; c += 32) {
            float v = __ldg(&xr[c]);
            sum += v * v;
        }

        // warp 归约
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);

        float m2 = sum / C;
        if (lane == 0) mean2[row] = m2;
        float inv = rsqrtf(m2 + kEps);

        // 归一化写回（向量化 + 尾部）
        c = lane * 4;
        for (; c < vec_end; c += 32 * 4) {
            float4 xv = *reinterpret_cast<const float4*>(&xr[c]);
            float4 wv = *reinterpret_cast<const float4*>(&shared_w[c]);
            float4 r;
            r.x = xv.x * wv.x * inv;
            r.y = xv.y * wv.y * inv;
            r.z = xv.z * wv.z * inv;
            r.w = xv.w * wv.w * inv;
            *reinterpret_cast<float4*>(&yr[c]) = r;
        }
        for (c = vec_end + lane; c < C; c += 32) {
            yr[c] = __ldg(&xr[c]) * shared_w[c] * inv;
        }
    }
}


// ----------------------------------------------------------------------------
// kernel launch

void rmsnorm_forward1(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int B,
    int T, 
    int C,
    const int block_size
){
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    rmsnorm_forward_kernel1<<<grid_size, block_size>>>(y, mean2, x, w, N, C);
    CUDA_CHECK(cudaGetLastError());

}

void rmsnorm_forward2(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int B,
    int T, 
    int C,
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    rmsnorm_forward_kernel2<<<grid_size, block_size>>>(y, mean2, x, w, N, C);
    CUDA_CHECK(cudaGetLastError());

}

void rmsnorm_forward3(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int B,
    int T, 
    int C,
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);

    bool use_vec2 = (C % 2 == 0) &&
                    is_aligned(x, 8) && is_aligned(w, 8) && is_aligned(y, 8);
    if (use_vec2) {
        rmsnorm_forward_kernel3<<<grid_size, block_size>>>(
            y, mean2, x, w, N, C);
    } else {
        rmsnorm_forward_kernel2<<<grid_size, block_size>>>(
            y, mean2, x, w, N, C);
    }
    CUDA_CHECK(cudaGetLastError());

}

void rmsnorm_forward4(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int B,
    int T, 
    int C,
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);

    // 检查是否满足 float4 向量化条件
    bool use_vec4 = (C % 4 == 0) &&
                    is_aligned(x, 16) && is_aligned(w, 16) && is_aligned(y, 16);
    if (use_vec4) {
        rmsnorm_forward_kernel4<<<grid_size, block_size>>>(
            y, mean2, x, w, N, C);
    } else {
        rmsnorm_forward_kernel2<<<grid_size, block_size>>>(
            y, mean2, x, w, N, C);
    }
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_forward5(
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int B,
    int T, 
    int C,
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    size_t smem_size = C * sizeof(float);
    // 检查共享内存是否超限（可选）
    cudaFuncSetAttribute(rmsnorm_forward_kernel5,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    rmsnorm_forward_kernel5<<<grid_size, block_size, smem_size>>>(y, mean2, x, w, N, C);
    CUDA_CHECK(cudaGetLastError());

}

void rmsnorm_forward6(
    float* y, 
    float* mean2,
    const float* x, 
    const float* w,
    int B, 
    int T, 
    int C,
    int block_size
){
    const int N = B * T;
    if (block_size % 32 != 0) { /* 要求 block_size 是 32 的倍数 */ }
    int grid_size = -1;
    // 如果使用 grid-stride，grid_size 通常固定
    if (grid_size <= 0) {
        // 自动计算：让每个 SM 尽可能多驻留 block，但不超过 row 数
        int device;
        cudaGetDevice(&device);
        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        int max_blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, rmsnorm_forward_kernel6, block_size, C * sizeof(float));
        grid_size = sm_count * max_blocks_per_sm;
        // 限制 grid_size 不超过 ceil(N / (block_size/32))，因为每个 warp 处理一行
        int max_needed = (N + (block_size/32) - 1) / (block_size/32);
        grid_size = std::min(grid_size, max_needed);
    }

    size_t smem_size = C * sizeof(float);   // 缓存权重 w
    rmsnorm_forward_kernel6<<<grid_size, block_size, smem_size>>>(
        y, mean2, x, w, N, C);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_forward(
    int kernel_num,
    float* y,
    float* mean2,
    const float* x,
    const float* w,
    int B,
    int T,
    int C,
    const int block_size
){
    switch(kernel_num){
        case 1:
            rmsnorm_forward1(y, mean2, x, w, B, T, C, block_size);
            break;
        case 2:
            rmsnorm_forward2(y, mean2, x, w, B, T, C, block_size);
            break;
        case 3:
            rmsnorm_forward3(y, mean2, x, w, B, T, C, block_size);
            break;
        case 4:
            rmsnorm_forward4(y, mean2, x, w, B, T, C, block_size);
            break;
        case 5:
            rmsnorm_forward5(y, mean2, x, w, B, T, C, block_size);
            break;
        case 6:
            rmsnorm_forward6(y, mean2, x, w, B, T, C, block_size);
            break;
        default:
            std::printf("Invalid kernel number.\n");
            exit(1);
    }
}


int main(int argc, char **argv){
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;

    int deviceIdx = 0;
    CUDA_CHECK(cudaSetDevice(deviceIdx));

    // 创建cpu上内存分配并输出化输入
    float* y = (float*)malloc(B * T * C * sizeof(float));
    float* mean2 = (float*)malloc(B * T * sizeof(float));
    float* x = make_random_float(B * T * C);
    float* w = make_random_float(C);

    // 创建gpu上内存分配并复制输入
    float* d_y;
    float* d_mean2;
    float* d_x;
    float* d_w;
    CUDA_CHECK(cudaMalloc(&d_y, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean2, B * T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, B * T * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w, C * sizeof(float), cudaMemcpyHostToDevice));

    // 从命令行读取kernel序号
    int kernel_num = 1;
    if(argc > 1){
        kernel_num = atoi(argv[1]);
    }
    std::printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    rmsnorm_forward_cpu(y, mean2, x, w, B * T, C);

    for(int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j){
        int block_size = block_sizes[j];
        std::printf("Checking block size %d.\n", block_size);

        rmsnorm_forward(kernel_num, d_y, d_mean2, d_x, d_w, B, T, C, block_size);

        validate_result(d_y, y, "y", B * T * C, 1e-5f);
        validate_result(d_mean2, mean2, "mean2", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for(int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j){
        int block_size = block_sizes[j];

        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, rmsnorm_forward, 
                                            kernel_num, d_y, d_mean2, 
                                            d_x, d_w, B, T, C, block_size);
        long memory_ops = (2 * B * T * C) * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    free(y);
    free(mean2);
    free(x);
    free(w);
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_mean2));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    return 0;
}
