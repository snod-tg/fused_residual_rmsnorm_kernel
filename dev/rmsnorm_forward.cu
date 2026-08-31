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
#include <type_traits>
#include <vector>
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

// CPU 参考（double 归约，更接近真值)
template <typename T>
void rmsnorm_forward_cpu(
    T* y,
    float* mean2,
    const T* x,
    const T* w,
    int N, 
    int C 
){
    for(int row = 0; row < N; ++row){
        const T* xr = x + (size_t)row * C;
        T* yr = y + (size_t)row * C;
        double sum = 0.0;       // 参开用double，更接近真值
        for (int c = 0; c < C; ++c){
            double v = as_float(xr[c]);
            sum += v * v;
        }

        float m2 = (float)(sum / C);
        mean2[row] = m2;
        float inv = (float)(1.0 / std::sqrt((double)m2 + kEps));
        for (int c = 0; c < C; ++c){
            yr[c] = from_float<T>(as_float(xr[c]) * as_float(w[c]) * inv);
        }
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
    float inv = rsqrtf(m2 + kEps);

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

    float inv = rsqrtf(m2 + kEps);

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
    float inv = rsqrtf(m2 + kEps);

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
    float inv = rsqrtf(m2 + kEps);

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

    float inv = rsqrtf(m2 + kEps);

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

// warp 归约版 RMSNorm 前向（模板，标量 load）
template <typename T>
__global__ void rmsnorm_forward_kernel7(
    T* y,
    float* mean2,
    const T* x,
    const T* w,
    int N,
    int C
){
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int row = blockIdx.x * (blockDim.x / 32) + warp_id;
    if(row >= N) return;

    const T* xr = x + (size_t)row * C;
    T* yr = y + (size_t)row * C;

    float sum = 0.0f;
    for(int c = lane; c < C; c+= 32){
        float v = as_float(xr[c]);
        sum += v * v;
    }
    for(int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    float m2 = sum / C;
    float inv = rsqrtf(m2 + kEps);
    if (lane == 0) mean2[row] = m2;

    for(int c = lane; c < C; c+= 32){
        yr[c] = from_float<T>(as_float(xr[c]) * as_float(w[c]) * inv); // 算完转回低精度写回
    }
}

// vec2 向量化版本：一次搬 2 个元素（float2 / half2 / __nv_bfloat162）
template <typename T> struct vec2_of;
template <> struct vec2_of<float> { using type = float2; };
template <> struct vec2_of<half> {using type = half2; };
template <> struct vec2_of<__nv_bfloat16> {using type = __nv_bfloat162; };

template <typename T>
__global__ void rmsnorm_forward_kernel8(
    T* y,
    float* mean2,
    const T* x,
    const T* w,
    int N,
    int C
){
    using V2 = typename vec2_of<T>::type;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int row = blockIdx.x * (blockDim.x / 32) + warp_id;
    if(row >= N) return;

    const T* xr = x + (size_t)row * C;
    T* yr = y + (size_t)row * C;

    float sum = 0.0f;
    for (int c = lane * 2; c < C; c += 32 * 2){
        V2 v = *reinterpret_cast<const V2*>(xr + c);
        sum += as_float(v.x) * as_float(v.x) + as_float(v.y) * as_float(v.y);
    }
    for(int off = 16; off > 0; off >>= 1){
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }

    float m2 = sum / C;
    float inv = rsqrtf(m2 + kEps);
    if(lane == 0) mean2[row] = m2;

    for(int c = lane * 2; c < C; c+=32 * 2){
        V2 xv = *reinterpret_cast<const V2*>(xr + c);
        V2 wv = *reinterpret_cast<const V2*>(w + c);
        V2 r;
        r.x = from_float<T>(as_float(xv.x) * as_float(wv.x) * inv);
        r.y = from_float<T>(as_float(xv.y) * as_float(wv.y) * inv);
        *reinterpret_cast<V2*>(yr + c) = r;
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
    CUDA_CHECK(cudaFuncSetAttribute(rmsnorm_forward_kernel5,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size));
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
    // 如果使用 grid-stride，grid_size 通常固定
    // 自动计算：让每个 SM 尽可能多驻留 block，但不超过 row 数
    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, rmsnorm_forward_kernel6, block_size, C * sizeof(float));
    int grid_size = sm_count * max_blocks_per_sm;
    // 限制 grid_size 不超过 ceil(N / (block_size/32))，因为每个 warp 处理一行
    int max_needed = (N + (block_size/32) - 1) / (block_size/32);
    grid_size = std::min(grid_size, max_needed);

    size_t smem_size = C * sizeof(float);   // 缓存权重 w
    CUDA_CHECK(cudaFuncSetAttribute(rmsnorm_forward_kernel6,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size));
    rmsnorm_forward_kernel6<<<grid_size, block_size, smem_size>>>(
        y, mean2, x, w, N, C);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void rmsnorm_forward7(
    T* y,
    float* mean2,
    const T* x,
    const T* w,
    int B,
    int T1,
    int C,
    int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T1;
    const int grid_size = ceil_div(N * 32, block_size);
    rmsnorm_forward_kernel7<T><<<grid_size, block_size>>>(y, mean2, x, w, N, C);
    CUDA_CHECK(cudaGetLastError());

}

template <typename T>
void rmsnorm_forward8(
    T* y,
    float* mean2,
    const T* x,
    const T* w,
    int B,
    int T1,
    int C,
    int block_size    
){
    assert(block_size % 32 == 0);
    const int N = B * T1;
    const int grid_size = ceil_div(N * 32, block_size);

    bool use_vec2 = (C % 2 == 0) &&
                    is_aligned(x, 2 * sizeof(T)) && is_aligned(w, 2 * sizeof(T)) &&
                    is_aligned(y, 2 * sizeof(T));
    if (use_vec2) {
        rmsnorm_forward_kernel8<T><<<grid_size, block_size>>>(
            y, mean2, x, w, N, C);
    } else {
        rmsnorm_forward_kernel7<T><<<grid_size, block_size>>>(
            y, mean2, x, w, N, C);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void rmsnorm_forward(
    int kernel_num,
    T* y,
    float* mean2,
    const T* x,
    const T* w,
    int B,
    int T1,
    int C,
    const int block_size
){
    if constexpr (std::is_same_v<T, float>) {
        // T=float：1~8 全可用（1~6 是 FP32 专用 kernel）
        switch (kernel_num) {
            case 1: rmsnorm_forward1(y, mean2, x, w, B, T1, C, block_size); break;
            case 2: rmsnorm_forward2(y, mean2, x, w, B, T1, C, block_size); break;
            case 3: rmsnorm_forward3(y, mean2, x, w, B, T1, C, block_size); break;
            case 4: rmsnorm_forward4(y, mean2, x, w, B, T1, C, block_size); break;
            case 5: rmsnorm_forward5(y, mean2, x, w, B, T1, C, block_size); break;
            case 6: rmsnorm_forward6(y, mean2, x, w, B, T1, C, block_size); break;
            case 7: rmsnorm_forward7<T>(y, mean2, x, w, B, T1, C, block_size); break;
            case 8: rmsnorm_forward8<T>(y, mean2, x, w, B, T1, C, block_size); break;
            default:
                std::printf("Invalid kernel number.\n");
                exit(1);
        }
    } else {
        // T=half/bf16：只有混合精度 kernel 7(标量)/8(vec2) 可用
        switch (kernel_num) {
            case 7: rmsnorm_forward7<T>(y, mean2, x, w, B, T1, C, block_size); break;
            case 8: rmsnorm_forward8<T>(y, mean2, x, w, B, T1, C, block_size); break;
            default:
                std::printf("kernel %d only supports fp32 (T=float).\n", kernel_num);
                exit(1);
        }
    }
}

// ---- 通用测试：先对拍、再测速度（FP32/FP16/BF16 共用）----
template <typename T>
int run_benchmark(const char* dtype_name, int kernel_num, float tol) {
    const int B = 8, T_tokens = 1024, C = 768;
    const int N = B * T_tokens;

    // 1) 生成 T 类型的输入（[-1, 1]）
    std::vector<T> h_x(N * C), h_w(C);
    srand(0);
    for (int i = 0; i < N * C; ++i) h_x[i] = from_float<T>((float)rand() / RAND_MAX * 2.0f - 1.0f);
    for (int i = 0; i < C; ++i)     h_w[i] = from_float<T>((float)rand() / RAND_MAX * 2.0f - 1.0f);

    // 2) CPU 参考（double 归约，更接近真值）
    std::vector<T> h_y(N * C);
    std::vector<float> h_mean2(N);
    rmsnorm_forward_cpu<T>(h_y.data(), h_mean2.data(), h_x.data(), h_w.data(), N, C);

    // 3) 设备内存
    T *d_y, *d_x, *d_w;
    float* d_mean2;
    CUDA_CHECK(cudaMalloc(&d_y, N * C * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_mean2, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * C * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_w, C * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * C * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), C * sizeof(T), cudaMemcpyHostToDevice));

    // 4) 对拍：遍历 block 大小
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int bad = 0;
    for (int j = 0; j < 6; ++j) {
        int block_size = block_sizes[j];
        rmsnorm_forward<T>(kernel_num, d_y, d_mean2, d_x, d_w, B, T_tokens, C, block_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> got_y(N * C);
        std::vector<float> got_mean2(N);
        CUDA_CHECK(cudaMemcpy(got_y.data(), d_y, N * C * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_mean2.data(), d_mean2, N * sizeof(float), cudaMemcpyDeviceToHost));

        double max_abs = 0, max_rel = 0;
        for (int i = 0; i < N * C; ++i) {
            double a = as_float(got_y[i]), b = as_float(h_y[i]);
            double diff = std::fabs(a - b);
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, diff / (std::fabs(b) + 1e-12));
            if (diff > tol + tol * std::fabs(b)) ++bad;
        }
        // mean2 始终是 FP32，用固定容差
        for (int i = 0; i < N; ++i) {
            double diff = std::fabs((double)got_mean2[i] - h_mean2[i]);
            if (diff > 2e-4f + 2e-4f * std::fabs(h_mean2[i])) ++bad;
        }
        std::printf("[%s] block %4d | y max_abs=%.3g max_rel=%.3g\n",
                    dtype_name, block_size, max_abs, max_rel);
    }
    if (bad) {
        std::printf("[%s] FAILED (%d errors)\n", dtype_name, bad);
    } else {
        std::printf("[%s] All results match. Starting benchmarks.\n\n", dtype_name);
    }

    // 5) 测速度：遍历 block 大小
    for (int j = 0; j < 6; ++j) {
        int block_size = block_sizes[j];
        float elapsed_time = benchmark_kernel(2000, rmsnorm_forward<T>, kernel_num,
                                              d_y, d_mean2, d_x, d_w, B, T_tokens, C, block_size);
        long memory_ops = (2L * N * C) * sizeof(T);   // 读 x + 写 y 的字节数
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;
        printf("[%s] block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n",
               dtype_name, block_size, elapsed_time, memory_bandwidth);
    }

    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_mean2));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    return bad ? 1 : 0;
}

int main(int argc, char **argv){
    int kernel_num = 7;                    // 默认跑混合精度标量版
    if (argc > 1) kernel_num = atoi(argv[1]);
    std::printf("Using kernel %d\n", kernel_num);

    // kernel 1~6 是 FP32 专用
    if (kernel_num >= 1 && kernel_num <= 6) {
        return run_benchmark<float>("fp32", kernel_num, 3e-4f);
    }
    // kernel 7(标量) / 8(vec2) 是混合精度版：三种 dtype 各跑一遍对拍 + 带宽
    if (kernel_num == 7 || kernel_num == 8) {
        int fails = 0;
        fails += run_benchmark<float>("fp32", kernel_num, 3e-4f);
        fails += run_benchmark<half>("fp16", kernel_num, 5e-3f);
        fails += run_benchmark<__nv_bfloat16>("bf16", kernel_num, 4e-2f);
        return fails ? 1 : 0;
    }
    std::printf("Invalid kernel number (1~8).\n");
    return 1;
}
