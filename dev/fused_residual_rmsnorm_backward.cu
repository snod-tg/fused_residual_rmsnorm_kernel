/*
Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt fused_residual_rmsnorm_backward.cu -o fused_residual_rmsnorm_backward
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math fused_residual_rmsnorm_backward.cu -o fused_residual_rmsnorm_backward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./fused_residual_rmsnorm_backward 1

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

// 混合精度前向 CPU 参考：给反向提供 z 和 mean2
template <typename T>
void fused_residual_rmsnorm_forward_cpu(
    T* y, T* z, float* mean2, const T* x, const T* residual, const T* w, int N, int C
){
    for(int row = 0; row < N; ++row){
        const T* xr = x + (size_t)row * C;
        const T* residualr = residual + (size_t)row * C;
        T* yr = y + (size_t)row * C;
        T* zr = z + (size_t)row * C;

        double sum = 0.0;
        for(int c = 0; c < C; ++c){
            T val = from_float<T>(as_float(xr[c]) + as_float(residualr[c]));
            zr[c] = val;
            double v = as_float(val);
            sum += v * v;
        }
        float m2 = (float)(sum / C);
        mean2[row] = m2;
        float inv = (float)(1.0 / std::sqrt((double)m2 + kEps));
        for(int c = 0; c < C; ++c)
            yr[c] = from_float<T>(as_float(zr[c]) * as_float(w[c]) * inv);
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

// 混合精度反向 CPU 参考：dx=dresidual=round(dnorm)，dw 用 double 累加
template <typename T>
void fused_residual_rmsnorm_backward_cpu(
    T* dx, T* dresidual, float* dw, const T* dy,
    const T* z, const T* weight, const float* mean2, int N, int C
){
    std::vector<double> dw_acc(C, 0.0);
    for(int row = 0; row < N; ++row){
        const T* zr = z + (size_t)row * C;
        const T* dyr = dy + (size_t)row * C;
        T* dxr = dx + (size_t)row * C;
        T* dresidualr = dresidual + (size_t)row * C;

        double sum = 0.0;
        for(int c = 0; c < C; ++c){
            sum += (double)as_float(dyr[c]) * as_float(weight[c]) * as_float(zr[c]);
        }
        float inv = (float)(1.0 / std::sqrt((double)mean2[row] + kEps));
        double correction = sum * inv * inv / C;
        for(int c = 0; c < C; ++c){
            float grad = as_float(dyr[c]);
            float value = as_float(zr[c]);
            float result = (grad * as_float(weight[c]) - value * (float)correction) * inv;
            dxr[c] = from_float<T>(result);
            dresidualr[c] = from_float<T>(result);
            dw_acc[c] += (double)grad * value * inv;
        }
    }
    for(int c = 0; c < C; ++c) dw[c] = (float)dw_acc[c];
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

// ---- 混合精度：kernel6 = warp + shared 聚合（标量），kernel7 = persistent + vec2 ----
template <typename T> struct vec2_of;
template <> struct vec2_of<float> { using type = float2; };
template <> struct vec2_of<half> { using type = half2; };
template <> struct vec2_of<__nv_bfloat16> { using type = __nv_bfloat162; };

// 混合精度：block 内共享内存聚合（对标 FP32 kernel3）
template <typename T>
__global__ void fused_residual_rmsnorm_backward_kernel6(
    T* dx, T* dresidual, float* dw, const T* dy,
    const T* z, const T* weight, const float* mean2, int N, int C
){
    extern __shared__ float shared_dw[];
    for(int c = threadIdx.x; c < C; c+= blockDim.x) shared_dw[c] = 0.0f;
    __syncthreads();

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int row = blockIdx.x * (blockDim.x / 32) + warp_id;

    if(row < N){
        const T* zr = z + (size_t)row * C;
        const T* dyr = dy + (size_t)row * C;
        T* dxr = dx + (size_t)row * C;
        T* dresidualr = dresidual + (size_t)row * C;

        float sum = 0.0f;
        for(int c = lane; c < C; c+=32){
            sum += as_float(dyr[c]) * as_float(zr[c]) * as_float(weight[c]);
        }
        for(int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);

        float inv = rsqrtf(mean2[row] + kEps);
        float correction = sum * inv * inv / C;

        for(int c = lane; c < C; c+= 32){
            float grad = as_float(dyr[c]);
            float value = as_float(zr[c]);
            float result = (grad * as_float(weight[c]) - value * correction) * inv;
            T out = from_float<T>(result);
            dxr[c] = out;
            dresidualr[c] = out;
            atomicAdd(&shared_dw[c], grad * value * inv);
        }
    }
    __syncthreads();
    for(int c = threadIdx.x; c < C; c += blockDim.x)
        atomicAdd(&dw[c], shared_dw[c]);
}

// 混合精度：persistent + vec2（对标 FP32 kernel5）
template <typename T>
__global__ void fused_residual_rmsnorm_backward_kernel7(
    T* dx, T* dresidual, float* dw, const T* dy,
    const T* z, const T* w, const float* mean2, int N, int C
){
    extern __shared__ float shared_dw[];
    for(int c = threadIdx.x; c < C; c+= blockDim.x) shared_dw[c] = 0.0f;
    __syncthreads();

    using V2 = typename vec2_of<T>::type;
    int warps = blockDim.x / 32;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int first_row = blockIdx.x * warps + warp_id;
    int row_stride = gridDim.x * warps;
    for (long row = first_row; row < N; row += row_stride) {
        const T* zr = z + (size_t)row * C;
        const T* dyr = dy + (size_t)row * C;
        T* dxr = dx + (size_t)row * C;
        T* dresidualr = dresidual + (size_t)row * C;

        float sum = 0.0f;
        for(int c = lane * 2; c < C; c+=32 * 2){
            const V2 dyr2 = *reinterpret_cast<const V2*>(dyr + c);
            const V2 zr2 = *reinterpret_cast<const V2*>(zr + c);
            const V2 w2 = *reinterpret_cast<const V2*>(w + c);
            sum += as_float(dyr2.x) * as_float(zr2.x) * as_float(w2.x)
                 + as_float(dyr2.y) * as_float(zr2.y) * as_float(w2.y);
        }
        for(int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);

        float inv = rsqrtf(mean2[row] + kEps);
        float correction = sum * inv * inv / C;

        for(int c = lane * 2; c < C; c+= 32 * 2){
            const V2 dyr2 = *reinterpret_cast<const V2*>(dyr + c);
            const V2 zr2 = *reinterpret_cast<const V2*>(zr + c);
            const V2 w2 = *reinterpret_cast<const V2*>(w + c);
            V2 tmp;
            tmp.x = from_float<T>(as_float(dyr2.x) * as_float(w2.x) * inv
                                 - as_float(zr2.x) * inv * correction);
            tmp.y = from_float<T>(as_float(dyr2.y) * as_float(w2.y) * inv
                                 - as_float(zr2.y) * inv * correction);
            *reinterpret_cast<V2*>(dxr + c) = tmp;
            *reinterpret_cast<V2*>(dresidualr + c) = tmp;
            atomicAdd(&shared_dw[c],     as_float(dyr2.x) * as_float(zr2.x) * inv);
            atomicAdd(&shared_dw[c + 1], as_float(dyr2.y) * as_float(zr2.y) * inv);
        }
    }
    __syncthreads();
    for(int c = threadIdx.x; c < C; c += blockDim.x)
        atomicAdd(&dw[c], shared_dw[c]);
}

// kernel8：C=768 专用的寄存器 dw 分层归约。每个 warp 在寄存器中累计 24 个
// 通道，block 末尾无冲突地合并，避免 kernel6/7 对 shared_dw 的逐行 atomicAdd。
// FP32、FP16、BF16 为三个独立实现，不使用模板。
__global__ void fused_residual_rmsnorm_backward_kernel8_fp32(
    float* __restrict__ dx, float* __restrict__ dresidual, float* dw,
    const float* __restrict__ dy, const float* __restrict__ z,
    const float* __restrict__ w, const float* __restrict__ mean2, int N
){
    constexpr int C = 768;
    constexpr int ITEMS = C / 32;
    extern __shared__ float warp_dw[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int first_row = blockIdx.x * warps + warp;
    const int row_stride = gridDim.x * warps;
    float acc[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) acc[i] = 0.0f;

    for (int row = first_row; row < N; row += row_stride) {
        const float* zr = z + (size_t)row * C;
        const float* dyr = dy + (size_t)row * C;
        float* dxr = dx + (size_t)row * C;
        float* drr = dresidual + (size_t)row * C;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int c = i * 32 + lane;
            dot = fmaf(dyr[c] * zr[c], w[c], dot);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        const float inv = rsqrtf(mean2[row] + kEps);
        const float correction = dot * inv * inv / C;
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int c = i * 32 + lane;
            const float grad = dyr[c];
            const float value = zr[c];
            const float out = (grad * w[c] - value * correction) * inv;
            dxr[c] = out;
            drr[c] = out;
            acc[i] = fmaf(grad * value, inv, acc[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i)
        warp_dw[warp * C + i * 32 + lane] = acc[i];
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float total = 0.0f;
        #pragma unroll
        for (int wid = 0; wid < 32; ++wid) {
            if (wid < warps) total += warp_dw[wid * C + c];
        }
        atomicAdd(dw + c, total);
    }
}

__global__ void fused_residual_rmsnorm_backward_kernel8_fp16(
    half* __restrict__ dx, half* __restrict__ dresidual, float* dw,
    const half* __restrict__ dy, const half* __restrict__ z,
    const half* __restrict__ w, const float* __restrict__ mean2, int N
){
    constexpr int C = 768;
    constexpr int ITEMS = C / 32;
    extern __shared__ float warp_dw[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int first_row = blockIdx.x * warps + warp;
    const int row_stride = gridDim.x * warps;
    float acc[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) acc[i] = 0.0f;

    for (int row = first_row; row < N; row += row_stride) {
        const half* zr = z + (size_t)row * C;
        const half* dyr = dy + (size_t)row * C;
        half* dxr = dx + (size_t)row * C;
        half* drr = dresidual + (size_t)row * C;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int c = i * 32 + lane;
            dot = fmaf(__half2float(dyr[c]) * __half2float(zr[c]), __half2float(w[c]), dot);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        const float inv = rsqrtf(mean2[row] + kEps);
        const float correction = dot * inv * inv / C;
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int c = i * 32 + lane;
            const float grad = __half2float(dyr[c]);
            const float value = __half2float(zr[c]);
            const half out = __float2half_rn((grad * __half2float(w[c]) - value * correction) * inv);
            dxr[c] = out;
            drr[c] = out;
            acc[i] = fmaf(grad * value, inv, acc[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i)
        warp_dw[warp * C + i * 32 + lane] = acc[i];
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float total = 0.0f;
        #pragma unroll
        for (int wid = 0; wid < 32; ++wid) {
            if (wid < warps) total += warp_dw[wid * C + c];
        }
        atomicAdd(dw + c, total);
    }
}

__global__ void fused_residual_rmsnorm_backward_kernel8_bf16(
    __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dresidual,
    float* dw, const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ mean2, int N
){
    constexpr int C = 768;
    constexpr int ITEMS = C / 32;
    extern __shared__ float warp_dw[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int first_row = blockIdx.x * warps + warp;
    const int row_stride = gridDim.x * warps;
    float acc[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) acc[i] = 0.0f;

    for (int row = first_row; row < N; row += row_stride) {
        const __nv_bfloat16* zr = z + (size_t)row * C;
        const __nv_bfloat16* dyr = dy + (size_t)row * C;
        __nv_bfloat16* dxr = dx + (size_t)row * C;
        __nv_bfloat16* drr = dresidual + (size_t)row * C;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int c = i * 32 + lane;
            dot = fmaf(__bfloat162float(dyr[c]) * __bfloat162float(zr[c]),
                       __bfloat162float(w[c]), dot);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        const float inv = rsqrtf(mean2[row] + kEps);
        const float correction = dot * inv * inv / C;
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int c = i * 32 + lane;
            const float grad = __bfloat162float(dyr[c]);
            const float value = __bfloat162float(zr[c]);
            const __nv_bfloat16 out = __float2bfloat16_rn(
                (grad * __bfloat162float(w[c]) - value * correction) * inv);
            dxr[c] = out;
            drr[c] = out;
            acc[i] = fmaf(grad * value, inv, acc[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i)
        warp_dw[warp * C + i * 32 + lane] = acc[i];
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float total = 0.0f;
        #pragma unroll
        for (int wid = 0; wid < 32; ++wid) {
            if (wid < warps) total += warp_dw[wid * C + c];
        }
        atomicAdd(dw + c, total);
    }
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
    int T1,
    int C,
    const int block_size
){
    const int N = B * T1;
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
    int T1,
    int C,
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T1;
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
    int T1,
    int C,
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T1;

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
    int T1,
    int C,
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T1;

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
    int T1,
    int C,
    const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T1;

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

// 混合精度 launcher：kernel6 = warp + shared 聚合
template <typename T>
void fused_residual_rmsnorm_backward6(
    T* dx, T* dresidual, float* dw, const T* dy,
    const T* z, const T* w, const float* mean2, int B, int T1, int C, const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T1;
    const int grid_size = ceil_div(N * 32, block_size);
    size_t smem_size = C * sizeof(float);
    cudaFuncSetAttribute(fused_residual_rmsnorm_backward_kernel6<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    fused_residual_rmsnorm_backward_kernel6<T><<<grid_size, block_size, smem_size>>>(
        dx, dresidual, dw, dy, z, w, mean2, N, C);
    CUDA_CHECK(cudaGetLastError());
}

// 混合精度 launcher：kernel7 = persistent + vec2
template <typename T>
void fused_residual_rmsnorm_backward7(
    T* dx, T* dresidual, float* dw, const T* dy,
    const T* z, const T* w, const float* mean2, int B, int T1, int C, const int block_size
){
    assert(block_size % 32 == 0);
    const int N = B * T1;
    size_t smem_size = C * sizeof(float);
    cudaFuncSetAttribute(fused_residual_rmsnorm_backward_kernel7<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, fused_residual_rmsnorm_backward_kernel7<T>, block_size, C * sizeof(float));
    int grid_size = sm_count * max_blocks_per_sm;
    int max_needed = (N + (block_size/32) - 1) / (block_size/32);
    grid_size = std::min(grid_size, max_needed);

    bool use_vec2 = (C % 2 == 0) &&
                    is_aligned(z, 2 * sizeof(T)) && is_aligned(w, 2 * sizeof(T)) &&
                    is_aligned(dy, 2 * sizeof(T)) &&
                    is_aligned(dx, 2 * sizeof(T)) && is_aligned(dresidual, 2 * sizeof(T));
    if (use_vec2) {
        fused_residual_rmsnorm_backward_kernel7<T><<<grid_size, block_size, smem_size>>>(
            dx, dresidual, dw, dy, z, w, mean2, N, C);
    } else {
        fused_residual_rmsnorm_backward_kernel6<T><<<grid_size, block_size, smem_size>>>(
            dx, dresidual, dw, dy, z, w, mean2, N, C);
    }
    CUDA_CHECK(cudaGetLastError());
}

void fused_residual_rmsnorm_backward8(
    float* dx, float* dresidual, float* dw, const float* dy, const float* z,
    const float* w, const float* mean2, int B, int T1, int C, const int block_size
){
    if (C != 768) {
        fused_residual_rmsnorm_backward6<float>(
            dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size);
        return;
    }
    const int N = B * T1;
    const int warps = block_size / 32;
    const size_t smem_size = (size_t)warps * C * sizeof(float);
    int device, max_smem;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (smem_size > (size_t)max_smem) {
        fused_residual_rmsnorm_backward6<float>(
            dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size);
        return;
    }
    CUDA_CHECK(cudaFuncSetAttribute(fused_residual_rmsnorm_backward_kernel8_fp32,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    int sm_count, active;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, fused_residual_rmsnorm_backward_kernel8_fp32, block_size, smem_size));
    if (active == 0) {
        fused_residual_rmsnorm_backward6<float>(
            dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size);
        return;
    }
    const int grid_size = std::min(sm_count * active, ceil_div(N, warps));
    fused_residual_rmsnorm_backward_kernel8_fp32<<<grid_size, block_size, smem_size>>>(
        dx, dresidual, dw, dy, z, w, mean2, N);
    CUDA_CHECK(cudaGetLastError());
}

void fused_residual_rmsnorm_backward8(
    half* dx, half* dresidual, float* dw, const half* dy, const half* z,
    const half* w, const float* mean2, int B, int T1, int C, const int block_size
){
    if (C != 768) {
        fused_residual_rmsnorm_backward6<half>(
            dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size);
        return;
    }
    const int N = B * T1;
    const int warps = block_size / 32;
    const size_t smem_size = (size_t)warps * C * sizeof(float);
    int device, max_smem;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (smem_size > (size_t)max_smem) {
        fused_residual_rmsnorm_backward6<half>(
            dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size);
        return;
    }
    CUDA_CHECK(cudaFuncSetAttribute(fused_residual_rmsnorm_backward_kernel8_fp16,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    int sm_count, active;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, fused_residual_rmsnorm_backward_kernel8_fp16, block_size, smem_size));
    if (active == 0) {
        fused_residual_rmsnorm_backward6<half>(
            dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size);
        return;
    }
    const int grid_size = std::min(sm_count * active, ceil_div(N, warps));
    fused_residual_rmsnorm_backward_kernel8_fp16<<<grid_size, block_size, smem_size>>>(
        dx, dresidual, dw, dy, z, w, mean2, N);
    CUDA_CHECK(cudaGetLastError());
}

void fused_residual_rmsnorm_backward8(
    __nv_bfloat16* dx, __nv_bfloat16* dresidual, float* dw,
    const __nv_bfloat16* dy, const __nv_bfloat16* z, const __nv_bfloat16* w,
    const float* mean2, int B, int T1, int C, const int block_size
){
    if (C != 768) {
        fused_residual_rmsnorm_backward6<__nv_bfloat16>(
            dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size);
        return;
    }
    const int N = B * T1;
    const int warps = block_size / 32;
    const size_t smem_size = (size_t)warps * C * sizeof(float);
    int device, max_smem;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (smem_size > (size_t)max_smem) {
        fused_residual_rmsnorm_backward6<__nv_bfloat16>(
            dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size);
        return;
    }
    CUDA_CHECK(cudaFuncSetAttribute(fused_residual_rmsnorm_backward_kernel8_bf16,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    int sm_count, active;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, fused_residual_rmsnorm_backward_kernel8_bf16, block_size, smem_size));
    if (active == 0) {
        fused_residual_rmsnorm_backward6<__nv_bfloat16>(
            dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size);
        return;
    }
    const int grid_size = std::min(sm_count * active, ceil_div(N, warps));
    fused_residual_rmsnorm_backward_kernel8_bf16<<<grid_size, block_size, smem_size>>>(
        dx, dresidual, dw, dy, z, w, mean2, N);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void fused_residual_rmsnorm_backward(
    int kernel_num,
    T* dx,
    T* dresidual,
    float* dw,
    const T* dy,
    const T* z,
    const T* w,
    const float* mean2,
    int B,
    int T1,
    int C,
    const int block_size
){
    if constexpr (std::is_same_v<T, float>) {
        switch(kernel_num){
            case 1: fused_residual_rmsnorm_backward1(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            case 2: fused_residual_rmsnorm_backward2(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            case 3: fused_residual_rmsnorm_backward3(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            case 4: fused_residual_rmsnorm_backward4(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            case 5: fused_residual_rmsnorm_backward5(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            case 6: fused_residual_rmsnorm_backward6<T>(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            case 7: fused_residual_rmsnorm_backward7<T>(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            case 8: fused_residual_rmsnorm_backward8(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            default:
                std::printf("Invalid kernel number.\n");
                exit(1);
        }
    } else {
        switch(kernel_num){
            case 6: fused_residual_rmsnorm_backward6<T>(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            case 7: fused_residual_rmsnorm_backward7<T>(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
            case 8: fused_residual_rmsnorm_backward8(dx, dresidual, dw, dy, z, w, mean2, B, T1, C, block_size); break;
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

    // 1) 生成 T 类型的输入
    std::vector<T> h_x(N * C), h_residual(N * C), h_w(C), h_dy(N * C);
    srand(0);
    for (int i = 0; i < N * C; ++i) {
        h_x[i] = from_float<T>((float)rand() / RAND_MAX * 2.0f - 1.0f);
        h_residual[i] = from_float<T>((float)rand() / RAND_MAX * 2.0f - 1.0f);
        h_dy[i] = from_float<T>((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
    for (int i = 0; i < C; ++i) h_w[i] = from_float<T>((float)rand() / RAND_MAX * 2.0f - 1.0f);

    // 2) CPU 参考：先前向得到 z/mean2，再反向得到 dx/dresidual/dw
    std::vector<T> h_y(N * C), h_z(N * C);
    std::vector<float> h_mean2(N);
    fused_residual_rmsnorm_forward_cpu<T>(h_y.data(), h_z.data(), h_mean2.data(),
                                          h_x.data(), h_residual.data(), h_w.data(), N, C);

    std::vector<T> h_dx(N * C), h_dresidual(N * C);
    std::vector<float> h_dw(C);
    fused_residual_rmsnorm_backward_cpu<T>(h_dx.data(), h_dresidual.data(), h_dw.data(),
                                           h_dy.data(), h_z.data(), h_w.data(),
                                           h_mean2.data(), N, C);

    // 3) 设备内存
    T *d_dx, *d_dresidual, *d_dy, *d_z, *d_w;
    float *d_dw, *d_mean2;
    CUDA_CHECK(cudaMalloc(&d_dx, N * C * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dresidual, N * C * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dy, N * C * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_z, N * C * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_w, C * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dw, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean2, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_dy, h_dy.data(), N * C * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), N * C * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), C * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean2, h_mean2.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // 4) 对拍：遍历 block 大小
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int bad = 0;
    for (int j = 0; j < 6; ++j) {
        int block_size = block_sizes[j];
        CUDA_CHECK(cudaMemset(d_dx, 0, N * C * sizeof(T)));
        CUDA_CHECK(cudaMemset(d_dresidual, 0, N * C * sizeof(T)));
        CUDA_CHECK(cudaMemset(d_dw, 0, C * sizeof(float)));
        fused_residual_rmsnorm_backward<T>(kernel_num, d_dx, d_dresidual, d_dw,
                                           d_dy, d_z, d_w, d_mean2,
                                           B, T_tokens, C, block_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> got_dx(N * C);
        std::vector<float> got_dw(C);
        CUDA_CHECK(cudaMemcpy(got_dx.data(), d_dx, N * C * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_dw.data(), d_dw, C * sizeof(float), cudaMemcpyDeviceToHost));

        double max_abs = 0, max_rel = 0;
        for (int i = 0; i < N * C; ++i) {
            double a = as_float(got_dx[i]), b = as_float(h_dx[i]);
            double diff = std::fabs(a - b);
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, diff / (std::fabs(b) + 1e-12));
            if (diff > tol + tol * std::fabs(b)) ++bad;
        }
        // dw 是 FP32 累加，容差固定 2e-3
        for (int i = 0; i < C; ++i) {
            double diff = std::fabs((double)got_dw[i] - h_dw[i]);
            if (diff > 2e-3f + 2e-3f * std::fabs(h_dw[i])) ++bad;
        }
        std::printf("[%s] block %4d | dx max_abs=%.3g max_rel=%.3g\n",
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
        CUDA_CHECK(cudaMemset(d_dw, 0, C * sizeof(float)));
        float elapsed_time = benchmark_kernel(100, fused_residual_rmsnorm_backward<T>, kernel_num,
                                              d_dx, d_dresidual, d_dw, d_dy, d_z, d_w, d_mean2,
                                              B, T_tokens, C, block_size);
        long memory_ops = (5L * N * C) * sizeof(T) + (long)C * sizeof(T) + (long)N * 4 + (long)C * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;
        printf("[%s] block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n",
               dtype_name, block_size, elapsed_time, memory_bandwidth);
    }

    CUDA_CHECK(cudaFree(d_dx));
    CUDA_CHECK(cudaFree(d_dresidual));
    CUDA_CHECK(cudaFree(d_dy));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_dw));
    CUDA_CHECK(cudaFree(d_mean2));
    return bad ? 1 : 0;
}

int main(int argc, char **argv){
    int kernel_num = 6;                    // 默认跑混合精度 warp+shared 聚合版
    if (argc > 1) kernel_num = atoi(argv[1]);
    std::printf("Using kernel %d\n", kernel_num);

    // kernel 1~5 是 FP32 专用
    if (kernel_num >= 1 && kernel_num <= 5) {
        return run_benchmark<float>("fp32", kernel_num, 1e-3f);
    }
    // kernel 6(shared聚合) / 7(persistent+vec2) / 8(三种精度独立专用化)
    if (kernel_num == 6 || kernel_num == 7 || kernel_num == 8) {
        int fails = 0;
        fails += run_benchmark<float>("fp32", kernel_num, 1e-3f);
        fails += run_benchmark<half>("fp16", kernel_num, 5e-3f);
        fails += run_benchmark<__nv_bfloat16>("bf16", kernel_num, 4e-2f);
        return fails ? 1 : 0;
    }
    std::printf("Invalid kernel number (1~8).\n");
    return 1;
}
