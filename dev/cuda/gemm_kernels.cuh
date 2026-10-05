#pragma once
#include "common.h"
#include <mma.h>

// kernel1：native，每个线程计算 C 的一个元素，K 循环直接访问 global memory。
template<class T> __device__ void gemm_native_element(
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k,
    int row,
    int col
){
    if (row >= m || col >= n) return;
    float acc=0.f;
    for (int p=0;p<k;++p) acc=fmaf(as_float(a[size_t(row)*k+p]),as_float(b[size_t(p)*n+col]),acc);
    c[size_t(row)*n+col]=from_float<T>(acc);
}

template<class T> __global__ void gemm_forward_kernel1(
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k
){
    gemm_native_element(c,a,b,m,n,k,blockIdx.y*16+threadIdx.y,blockIdx.x*16+threadIdx.x);
}

// kernel3：第一版优化。16x16 shared-memory tile，线程之间复用 A/B，仍每线程一个输出。
// 越界元素补零；所有线程经过同一个 __syncthreads，支持非整除 M/N/K。
template<class T> __device__ void gemm_shared_tile(
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k,
    int tile_m,
    int tile_n,
    float (&sa)[16][16],float (&sb)[16][16]
){
    int x=threadIdx.x,y=threadIdx.y,row=tile_m*16+y,col=tile_n*16+x;
    float acc=0.f;
    for (int base=0;base<k;base+=16) {
        sa[y][x]=(row<m && base+x<k) ? as_float(a[size_t(row)*k+base+x]) : 0.f;
        sb[y][x]=(base+y<k && col<n) ? as_float(b[size_t(base+y)*n+col]) : 0.f;
        __syncthreads();
        #pragma unroll
        for (int p=0;p<16;++p) acc=fmaf(sa[y][p],sb[p][x],acc);
        __syncthreads();
    }
    if(row<m && col<n) c[size_t(row)*n+col]=from_float<T>(acc);
}

template<class T> __global__ void gemm_forward_kernel3(
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k
){
    __shared__ float sa[16][16],sb[16][16];
    gemm_shared_tile(c,a,b,m,n,k,blockIdx.y,blockIdx.x,sa,sb);
}

// kernel4：测过 kernel3 后的第二版。BM=BN=64, BK=16，256 个线程各计算 4x4。
// A 转置放入 shared 并 padding，B 连续放入 shared；寄存器保存外积累加结果。
// 相比 kernel3，每个 K tile 的输出从 256 增至 4096，提高数据复用。
template<class T,int BM=64> __device__ void gemm_register_tile(
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k,
    int tile_m,
    int tile_n,
    float (&sa)[16][65],float (&sb)[16][64]
){
    const int x=threadIdx.x,y=threadIdx.y,tid=y*16+x;
    const int row0=tile_m*BM,col0=tile_n*64;
    float acc[BM/16][4]={};
    for(int base=0;base<k;base+=16) {
        #pragma unroll
        for(int i=tid;i<BM*16;i+=256) {
            int row=i/16,p=i%16;
            sa[p][row]=(row0+row<m && base+p<k) ? as_float(a[size_t(row0+row)*k+base+p]) : 0.f;
        }
        #pragma unroll
        for(int i=tid;i<16*64;i+=256) {
            int p=i/64,col=i%64;
            sb[p][col]=(base+p<k && col0+col<n) ? as_float(b[size_t(base+p)*n+col0+col]) : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int p=0;p<16;++p) {
            float av[BM/16],bv[4];
            #pragma unroll
            for(int i=0;i<BM/16;++i) av[i]=sa[p][y+i*16];
            #pragma unroll
            for(int j=0;j<4;++j) bv[j]=sb[p][x+j*16];
            #pragma unroll
            for(int i=0;i<BM/16;++i) {
                #pragma unroll
                for(int j=0;j<4;++j) acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<BM/16;++i) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            int row=row0+y+i*16,col=col0+x+j*16;
            if(row<m && col<n)c[size_t(row)*n+col]=from_float<T>(acc[i][j]);
        }
    }
}

template<class T> __global__ void gemm_forward_kernel4(
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k
){
    __shared__ float sa[16][65],sb[16][64];
    gemm_register_tile(c,a,b,m,n,k,blockIdx.y,blockIdx.x,sa,sb);
}

// 128-bit FP32 / 64-bit FP16/BF16 global load：每次读 4 个元素，转换到 float 寄存器。
template<class T> __device__ __forceinline__ float4 gemm_load4(
    const T* ptr
){
    if constexpr (std::is_same_v<T,float>) {
        return *reinterpret_cast<const float4*>(ptr);
    } else {
        uint2 bits = *reinterpret_cast<const uint2*>(ptr);
        float2 lo, hi;
        if constexpr (std::is_same_v<T,half>) {
            lo = __half22float2(*reinterpret_cast<const half2*>(&bits.x));
            hi = __half22float2(*reinterpret_cast<const half2*>(&bits.y));
        } else {
            lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits.x));
            hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits.y));
        }
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
}

template<class T> __device__ __forceinline__ void gemm_store4(
    T* ptr,
    float4 value
){
    if constexpr (std::is_same_v<T,float>) {
        *reinterpret_cast<float4*>(ptr) = value;
    } else {
        uint2 bits;
        if constexpr (std::is_same_v<T,half>) {
            half2 lo = __floats2half2_rn(value.x, value.y);
            half2 hi = __floats2half2_rn(value.z, value.w);
            bits = make_uint2(*reinterpret_cast<const unsigned*>(&lo), *reinterpret_cast<const unsigned*>(&hi));
        } else {
            __nv_bfloat162 lo = __floats2bfloat162_rn(value.x, value.y);
            __nv_bfloat162 hi = __floats2bfloat162_rn(value.z, value.w);
            bits = make_uint2(*reinterpret_cast<const unsigned*>(&lo), *reinterpret_cast<const unsigned*>(&hi));
        }
        *reinterpret_cast<uint2*>(ptr) = bits;
    }
}

// kernel5/6 共用的 K tile loader。A 转置后使用 XOR 布局，去掉旧版的写入 bank conflict。
// BK=16 时，一个 warp 的 A store bank = row XOR ((p/4)*8)，恰好覆盖 32 个 bank。
template<class T,int BM,int BN,int BK,int THREADS>
__device__ __forceinline__ void gemm_prefetch_tile(
    const T* a,
    const T* b,
    int row0,
    int col0,
    int base,
    float4* av,
    float4* bv
){
    #pragma unroll
    for(int v=0; v<ceil_div(BM*BK/4,THREADS); ++v) {
        int index=threadIdx.x+v*THREADS;
        if(index<BM*BK/4) av[v]=gemm_load4(a+(row0+index/(BK/4))*1024+base+(index%(BK/4))*4);
    }
    #pragma unroll
    for(int v=0; v<ceil_div(BK*BN/4,THREADS); ++v) {
        int index=threadIdx.x+v*THREADS;
        if(index<BK*BN/4) bv[v]=gemm_load4(b+(base+index/(BN/4))*1024+col0+(index%(BN/4))*4);
    }
}

template<int BM,int BN,int BK,int THREADS>
__device__ __forceinline__ void gemm_publish_tile(
    const float4* av,
    const float4* bv,
    float (&sa)[BK][BM],
    float (&sb)[BK][BN]
){
    #pragma unroll
    for(int v=0; v<ceil_div(BM*BK/4,THREADS); ++v) {
        int index=threadIdx.x+v*THREADS;
        if(index<BM*BK/4) {
            int row=index/(BK/4),p=index%(BK/4)*4;
            float4 value=av[v];
            int swizzled=row^((p/4)*(128/BK));
            sa[p+0][swizzled]=value.x;sa[p+1][swizzled]=value.y;
            sa[p+2][swizzled]=value.z;sa[p+3][swizzled]=value.w;
        }
    }
    #pragma unroll
    for(int v=0; v<ceil_div(BK*BN/4,THREADS); ++v) {
        int index=threadIdx.x+v*THREADS;
        if(index<BK*BN/4) *reinterpret_cast<float4*>(&sb[index/(BN/4)][index%(BN/4)*4])=bv[v];
    }
}

// Shared -> register 的操作数预取，保留 kernel5/6 的 XOR 和 float4 布局。
template<int BM,int BN,int BK,int TM,int TN>
__device__ __forceinline__ void gemm_simt_load_fragments(
    float4 (&ar)[TM/4],
    float4 (&br)[TN/4],
    const float (&sa)[BK][BM],
    const float (&sb)[BK][BN],
    int thread_row,
    int thread_col,
    int p
){
    #pragma unroll
    for(int i=0;i<TM/4;++i) {
        int row=thread_row*4+i*(BM/(TM/4));
        ar[i]=*reinterpret_cast<const float4*>(&sa[p][row^((p/4)*(128/BK))]);
    }
    #pragma unroll
    for(int j=0;j<TN/4;++j) br[j]=*reinterpret_cast<const float4*>(&sb[p][thread_col*4+j*(BN/(TN/4))]);
}

// kernel5：向量化 global/shared load、向量化写回、XOR bank 布局、展开、restrict。
// kernel6：在 kernel5 上加入两个 shared pool；先发起下一 tile 的 LDG，再计算当前 tile。
// TM/TN=8 时将两组 float4 分到 tile 的两半，避免连续 8 列导致 LDS.128 两路冲突。
// 仅用于固定 1024^3，省去边界分支；模板参数在 launcher 内选定，不暴露给调用者。
template<class T,int BM,int BN,int BK,int TM,int TN,bool DOUBLE_BUFFER,bool REGISTER_PIPELINE=false,int K_UNROLL=BK>
__global__ void gemm_forward_vector_kernel(
    T* __restrict__ c,
    const T* __restrict__ a,
    const T* __restrict__ b
){
    constexpr int THREADS=(BM/TM)*(BN/TN), POOLS=DOUBLE_BUFFER?2:1;
    __shared__ __align__(16) float sa[POOLS][BK][BM],sb[POOLS][BK][BN];
    const int row0=blockIdx.y*BM,col0=blockIdx.x*BN;
    const int thread_row=threadIdx.x/(BN/TN),thread_col=threadIdx.x%(BN/TN);
    float acc[TM][TN]={};
    float4 next_a[(BM*BK/4+THREADS-1)/THREADS],next_b[(BK*BN/4+THREADS-1)/THREADS];
    gemm_prefetch_tile<T,BM,BN,BK,THREADS>(a,b,row0,col0,0,next_a,next_b);
    gemm_publish_tile<BM,BN,BK,THREADS>(next_a,next_b,sa[0],sb[0]);
    __syncthreads();
    int pool=0;
    for(int base=0; base<1024; base+=BK) {
        if constexpr (DOUBLE_BUFFER) {
            if(base+BK<1024) gemm_prefetch_tile<T,BM,BN,BK,THREADS>(a,b,row0,col0,base+BK,next_a,next_b);
        }
        if constexpr (REGISTER_PIPELINE) {
            // 两套寄存器操作数交替：先发起 p+1 的 LDS，再计算 p 的外积。
            float4 ar[2][TM/4],br[2][TN/4];
            gemm_simt_load_fragments<BM,BN,BK,TM,TN>(ar[0],br[0],sa[pool],sb[pool],thread_row,thread_col,0);
            #pragma unroll K_UNROLL
            for(int p=0;p<BK;++p) {
                int current=p&1;
                if(p+1<BK) gemm_simt_load_fragments<BM,BN,BK,TM,TN>(ar[current^1],br[current^1],sa[pool],sb[pool],thread_row,thread_col,p+1);
                #pragma unroll
                for(int i=0;i<TM;++i) {
                    float av=(&ar[current][i/4].x)[i%4];
                    #pragma unroll
                    for(int j=0;j<TN;++j) acc[i][j]=fmaf(av,(&br[current][j/4].x)[j%4],acc[i][j]);
                }
            }
        } else {
            #pragma unroll
            for(int p=0; p<BK; ++p) {
                float4 ar[TM/4],br[TN/4];
                #pragma unroll
                for(int i=0; i<TM/4; ++i) {
                    int row=thread_row*4+i*(BM/(TM/4));
                    ar[i]=*reinterpret_cast<const float4*>(&sa[pool][p][row^((p/4)*(128/BK))]);
                }
                #pragma unroll
                for(int j=0; j<TN/4; ++j) br[j]=*reinterpret_cast<const float4*>(&sb[pool][p][thread_col*4+j*(BN/(TN/4))]);
                #pragma unroll
                for(int i=0; i<TM; ++i) {
                    float av=(&ar[i/4].x)[i%4];
                    #pragma unroll
                    for(int j=0; j<TN; ++j) acc[i][j]=fmaf(av,(&br[j/4].x)[j%4],acc[i][j]);
                }
            }
        }
        if(base+BK<1024) {
            if constexpr (!DOUBLE_BUFFER) {
                __syncthreads(); // 单池必须等所有线程读完，才能覆盖当前 shared tile。
                gemm_prefetch_tile<T,BM,BN,BK,THREADS>(a,b,row0,col0,base+BK,next_a,next_b);
            }
            int next=DOUBLE_BUFFER ? pool^1 : 0;
            gemm_publish_tile<BM,BN,BK,THREADS>(next_a,next_b,sa[next],sb[next]);
            __syncthreads();
            pool=next;
        }
    }
    #pragma unroll
    for(int i=0; i<TM; ++i) {
        int row=row0+thread_row*4+i%4+(i/4)*(BM/(TM/4));
        #pragma unroll
        for(int j=0; j<TN; j+=4) {
            int col=col0+thread_col*4+(j/4)*(BN/(TN/4));
            gemm_store4(c+row*1024+col,make_float4(acc[i][j],acc[i][j+1],acc[i][j+2],acc[i][j+3]));
        }
    }
}

// SM80+ 16-byte async copy：绕过数据寄存器，将 global 数据直接放入另一个 shared pool。
__device__ __forceinline__ void gemm_copy_async16(
    void* destination,
    const void* source
){
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(address), "l"(source) : "memory");
}
__device__ __forceinline__ void gemm_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void gemm_async_wait() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template<class T,int BM,int BN,int BK,int PA,int PB,int THREADS>
__device__ __forceinline__ void gemm_async_tile(
    const T* a,
    const T* b,
    int row0,
    int col0,
    int base,
    T (&sa)[BM][PA],
    T (&sb)[BK][PB]
){
    constexpr int V=16/sizeof(T);
    static_assert(PA*sizeof(T)%16==0 && PB*sizeof(T)%16==0, "cp.async row pitch must be 16-byte aligned");
    for(int index=threadIdx.x; index<BM*BK/V; index+=THREADS) {
        int row=index/(BK/V),p=index%(BK/V)*V;
        gemm_copy_async16(&sa[row][p],a+(row0+row)*1024+base+p);
    }
    for(int index=threadIdx.x; index<BK*BN/V; index+=THREADS) {
        int p=index/(BN/V),col=index%(BN/V)*V;
        gemm_copy_async16(&sb[p][col],b+(base+p)*1024+col0+col);
    }
    gemm_async_commit();
}

// kernel7：cp.async + 两池。A 保留 row-major，padding 为 16 字节，线程输出行采用交错映射。
// 与 kernel6 不同：拷贝无需经过 float 寄存器；half/bf16 在消费 shared 数据时转换。
template<class T,int BM=64,int BN=64,int BK=16,int TM=4>
__global__ void gemm_forward_kernel7(
    T* __restrict__ c,
    const T* __restrict__ a,
    const T* __restrict__ b
){
    constexpr int THREADS=(BM/TM)*(BN/4),PA=BK+16/sizeof(T);
    __shared__ __align__(16) T sa[2][BM][PA],sb[2][BK][BN];
    int row0=blockIdx.y*BM,col0=blockIdx.x*BN;
    int x=threadIdx.x%(BN/4),y=threadIdx.x/(BN/4);
    float acc[TM][4]={};
    gemm_async_tile<T,BM,BN,BK,PA,BN,THREADS>(a,b,row0,col0,0,sa[0],sb[0]);
    gemm_async_wait();__syncthreads();
    int pool=0;
    for(int base=0; base<1024; base+=BK) {
        if(base+BK<1024) gemm_async_tile<T,BM,BN,BK,PA,BN,THREADS>(a,b,row0,col0,base+BK,sa[pool^1],sb[pool^1]);
        #pragma unroll
        for(int p=0; p<BK; ++p) {
            float av[TM];float4 bv=gemm_load4(&sb[pool][p][x*4]);
            #pragma unroll
            for(int i=0; i<TM; ++i) av[i]=as_float(sa[pool][y+i*(BM/TM)][p]);
            #pragma unroll
            for(int i=0; i<TM; ++i) {
                #pragma unroll
                for(int j=0; j<4; ++j) acc[i][j]=fmaf(av[i],(&bv.x)[j],acc[i][j]);
            }
        }
        gemm_async_wait();__syncthreads();pool^=1;
    }
    #pragma unroll
    for(int i=0; i<TM; ++i) gemm_store4(c+(row0+y+i*(BM/TM))*1024+col0+x*4,
        make_float4(acc[i][0],acc[i][1],acc[i][2],acc[i][3]));
}

// kernel8：低精度 Tensor Core，每个 warp 负责 32x32 输出。
// WMMA 16x16x16 + FP32 accumulator；cp.async 双池，pitch +8 保持 16-byte 对齐并错开行。
// 输入池和输出 tile 复用同一块 shared 空间；最后一次 block barrier 后才切换用途。
// 只接受 FP16/BF16，绝不隐式将 FP32 输入改成 TF32。
template<class T,int BM=64,int BN=64,int BK=32,bool DOUBLE_BUFFER=true>
__global__ void gemm_forward_kernel8(
    T* __restrict__ c,
    const T* __restrict__ a,
    const T* __restrict__ b
){
    using namespace nvcuda;
    constexpr int THREADS=(BM/32)*(BN/32)*32,POOLS=DOUBLE_BUFFER?2:1;
    struct InputPools {T sa[POOLS][BM][BK+8],sb[POOLS][BK][BN+8];};
    union SharedStorage {InputPools input;float output[BM][BN];};
    __shared__ __align__(32) SharedStorage shared;
    auto& sa=shared.input.sa;auto& sb=shared.input.sb;
    int row0=blockIdx.y*BM,col0=blockIdx.x*BN,warp=threadIdx.x/32;
    int warp_row=(warp/(BN/32))*32,warp_col=(warp%(BN/32))*32;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][2];
    #pragma unroll
    for(int i=0; i<2; ++i) {
        #pragma unroll
        for(int j=0; j<2; ++j) wmma::fill_fragment(acc[i][j],0.f);
    }
    gemm_async_tile<T,BM,BN,BK,BK+8,BN+8,THREADS>(a,b,row0,col0,0,sa[0],sb[0]);
    gemm_async_wait();__syncthreads();
    int pool=0;
    for(int base=0; base<1024; base+=BK) {
        if constexpr (DOUBLE_BUFFER) {
            if(base+BK<1024)gemm_async_tile<T,BM,BN,BK,BK+8,BN+8,THREADS>(a,b,row0,col0,base+BK,sa[pool^1],sb[pool^1]);
        }
        #pragma unroll
        for(int p=0; p<BK; p+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> ar[2];
            wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> br[2];
            #pragma unroll
            for(int i=0; i<2; ++i) wmma::load_matrix_sync(ar[i],&sa[pool][warp_row+i*16][p],BK+8);
            #pragma unroll
            for(int j=0; j<2; ++j) wmma::load_matrix_sync(br[j],&sb[pool][p][warp_col+j*16],BN+8);
            #pragma unroll
            for(int i=0; i<2; ++i) {
                #pragma unroll
                for(int j=0; j<2; ++j) wmma::mma_sync(acc[i][j],ar[i],br[j],acc[i][j]);
            }
        }
        if constexpr (!DOUBLE_BUFFER) {
            __syncthreads();
            if(base+BK<1024)gemm_async_tile<T,BM,BN,BK,BK+8,BN+8,THREADS>(a,b,row0,col0,base+BK,sa[0],sb[0]);
        }
        gemm_async_wait();__syncthreads();
        if constexpr (DOUBLE_BUFFER) pool^=1;
    }
    // 最后一轮 wait + block barrier 后，所有输入消费者已完成；共享池可安全复用为输出。
    #pragma unroll
    for(int i=0; i<2; ++i) {
        #pragma unroll
        for(int j=0; j<2; ++j) wmma::store_matrix_sync(&shared.output[warp_row+i*16][warp_col+j*16],acc[i][j],BN,wmma::mem_row_major);
    }
    __syncthreads();
    for(int index=threadIdx.x; index<BM*BN/4; index+=THREADS) {
        int row=index/(BN/4),col=index%(BN/4)*4;
        gemm_store4(c+(row0+row)*1024+col0+col,*reinterpret_cast<const float4*>(&shared.output[row][col]));
    }
}


// kernel9 的 warp 原语：显式 ldmatrix + mma，half/bf16 共用同一个矩阵布局。
// 这些原语只用于数学内核；计时、验证、数据管理继续共用 common.h。
__device__ __forceinline__ void gemm_ldmatrix_a(
    unsigned (&fragment)[4],
    const void* ptr
){
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(fragment[0]),"=r"(fragment[1]),"=r"(fragment[2]),"=r"(fragment[3]) : "r"(address));
}
__device__ __forceinline__ void gemm_ldmatrix_b(
    unsigned (&fragment)[2],
    const void* ptr
){
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(fragment[0]),"=r"(fragment[1]) : "r"(address));
}
template<class T> __device__ __forceinline__ void gemm_mma_16x8x16(
    float (&acc)[4],
    const unsigned (&a)[4],
    const unsigned (&b)[2]
){
    if constexpr (std::is_same_v<T,half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
}
template<class T> __device__ __forceinline__ void gemm_store2(
    T* ptr,
    float x,
    float y
){
    if constexpr (std::is_same_v<T,half>) {
        half2 value=__floats2half2_rn(x,y);
        *reinterpret_cast<half2*>(ptr)=value;
    } else {
        __nv_bfloat162 value=__floats2bfloat162_rn(x,y);
        *reinterpret_cast<__nv_bfloat162*>(ptr)=value;
    }
}

// 每个 16-byte 行片段保持连续；XOR 只置换片段在 shared 中的位置。
// SWIZZLE 时 pitch 为 64 的倍数，row 的低 3 bit 分散到 8 个片段。
template<bool SWIZZLE> __device__ __forceinline__ int gemm_mma_column(
    int row,
    int col
){
    return SWIZZLE ? col^((row&7)*8) : col;
}
template<class T,int BM,int BN,int BK,int PA,int PB,int THREADS,bool SWIZZLE>
__device__ __forceinline__ void gemm_mma_async_tile(
    const T* a,
    const T* b,
    int row0,
    int col0,
    int base,
    T (&sa)[BM][PA],
    T (&sb)[BK][PB]
){
    for(int index=threadIdx.x;index<BM*BK/8;index+=THREADS) {
        int row=index/(BK/8),p=index%(BK/8)*8;
        gemm_copy_async16(&sa[row][gemm_mma_column<SWIZZLE>(row,p)],a+(row0+row)*1024+base+p);
    }
    for(int index=threadIdx.x;index<BK*BN/8;index+=THREADS) {
        int p=index/(BN/8),col=index%(BN/8)*8;
        gemm_copy_async16(&sb[p][gemm_mma_column<SWIZZLE>(p,col)],b+(base+p)*1024+col0+col);
    }
    gemm_async_commit();
}
template<class T,int BM,int BN,int BK,int WM,int WN,int PA,int PB,bool SWIZZLE>
__device__ __forceinline__ void gemm_mma_load_fragments(
    unsigned (&ar)[WM/16][4],
    unsigned (&br)[WN/8][2],
    const T (&sa)[BM][PA],
    const T (&sb)[BK][PB],
    int warp_row,
    int warp_col,
    int p
){
    int lane=threadIdx.x&31;
    #pragma unroll
    for(int i=0;i<WM/16;++i) {
        int row=warp_row+i*16+(lane&15),col=p+(lane/16)*8;
        gemm_ldmatrix_a(ar[i],&sa[row][gemm_mma_column<SWIZZLE>(row,col)]);
    }
    #pragma unroll
    for(int j=0;j<WN/8;++j) {
        int row=p+(lane&15),col=warp_col+j*8;
        gemm_ldmatrix_b(br[j],&sb[row][gemm_mma_column<SWIZZLE>(row,col)]);
    }
}

// kernel9：显式 Tensor Core 指令，cp.async 双池，可选寄存器片段双缓冲。
// MMA accumulator 的 lane 映射已知，half2/bfloat162 直接写回，不再经 shared 输出 tile。
// 仍保持 row-major C=A@B、原 dtype 输入/输出、FP32 累加，专用于 1024^3。
template<class T,int BM=32,int BN=64,int BK=32,int WM=16,int WN=32,bool SWIZZLE=true,bool PIPELINE=true>
__global__ void gemm_forward_kernel9_mma(
    T* __restrict__ c,
    const T* __restrict__ a,
    const T* __restrict__ b
){
    static_assert(!std::is_same_v<T,float>,"mma kernel9 requires fp16/bf16");
    static_assert(BM%WM==0 && BN%WN==0 && WM%16==0 && WN%8==0 && BK%16==0 && 1024%BK==0,"invalid MMA tile");
    constexpr int THREADS=(BM/WM)*(BN/WN)*32;
    constexpr int PA=SWIZZLE ? ((BK+63)/64)*64 : BK+8;
    constexpr int PB=SWIZZLE ? ((BN+63)/64)*64 : BN+8;
    constexpr int FRAG_POOLS=PIPELINE?2:1;
    __shared__ __align__(16) T sa[2][BM][PA],sb[2][BK][PB];
    int row0=blockIdx.y*BM,col0=blockIdx.x*BN,warp=threadIdx.x/32,lane=threadIdx.x&31;
    int warp_row=warp/(BN/WN)*WM,warp_col=warp%(BN/WN)*WN;
    float acc[WM/16][WN/8][4]={};
    unsigned ar[FRAG_POOLS][WM/16][4],br[FRAG_POOLS][WN/8][2];
    gemm_mma_async_tile<T,BM,BN,BK,PA,PB,THREADS,SWIZZLE>(a,b,row0,col0,0,sa[0],sb[0]);
    gemm_async_wait();__syncthreads();
    int pool=0;
    for(int base=0;base<1024;base+=BK) {
        if(base+BK<1024) gemm_mma_async_tile<T,BM,BN,BK,PA,PB,THREADS,SWIZZLE>(a,b,row0,col0,base+BK,sa[pool^1],sb[pool^1]);
        if constexpr (PIPELINE) gemm_mma_load_fragments<T,BM,BN,BK,WM,WN,PA,PB,SWIZZLE>(ar[0],br[0],sa[pool],sb[pool],warp_row,warp_col,0);
        #pragma unroll
        for(int p=0;p<BK;p+=16) {
            constexpr int MASK=FRAG_POOLS-1;
            int current=(p/16)&MASK;
            if constexpr (PIPELINE) {
                if(p+16<BK) gemm_mma_load_fragments<T,BM,BN,BK,WM,WN,PA,PB,SWIZZLE>(ar[current^1],br[current^1],sa[pool],sb[pool],warp_row,warp_col,p+16);
            } else {
                gemm_mma_load_fragments<T,BM,BN,BK,WM,WN,PA,PB,SWIZZLE>(ar[0],br[0],sa[pool],sb[pool],warp_row,warp_col,p);
            }
            #pragma unroll
            for(int i=0;i<WM/16;++i) {
                #pragma unroll
                for(int j=0;j<WN/8;++j) gemm_mma_16x8x16<T>(acc[i][j],ar[current][i],br[current][j]);
            }
        }
        // 同时保护：下一池拷贝已完成，当前池的所有消费者已读完。
        // 末轮没有下一次 shared 读取或复用，可以省掉末轮 block barrier。
        if(base+BK<1024) {gemm_async_wait();__syncthreads();pool^=1;}
    }
    #pragma unroll
    for(int i=0;i<WM/16;++i) {
        #pragma unroll
        for(int j=0;j<WN/8;++j) {
            int row=row0+warp_row+i*16+lane/4,col=col0+warp_col+j*8+(lane%4)*2;
            gemm_store2(c+row*1024+col,acc[i][j][0],acc[i][j][1]);
            gemm_store2(c+(row+8)*1024+col,acc[i][j][2],acc[i][j][3]);
        }
    }
}

// 1024^3 sweep 后选定的配置，调用参数仍保持简洁：分块参数只在这一处指定。
template<class T,bool DOUBLE_BUFFER> void gemm_vector_1024(
    T* c,
    const T* a,
    const T* b
){
    if constexpr (std::is_same_v<T,float>) {
        gemm_forward_vector_kernel<T,32,128,16,4,8,DOUBLE_BUFFER><<<dim3(8,32),128>>>(c,a,b);
    } else {
        gemm_forward_vector_kernel<T,128,32,16,8,4,DOUBLE_BUFFER><<<dim3(32,8),128>>>(c,a,b);
    }
}

template<class T> void gemm_forward(
    int version,
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k,
    cublasHandle_t blas
){
    if (!m) return;
    if(version>=5 && (m!=1024 || n!=1024 || k!=1024)) throw std::runtime_error("kernel5..9 specialize 1024^3");
    if(version==1) gemm_forward_kernel1<<<dim3(ceil_div(n,16),ceil_div(m,16)),dim3(16,16)>>>(c,a,b,m,n,k);
    else if(version==2) gemm_cublas(blas,c,a,b,m,n,k);
    else if(version==3) gemm_forward_kernel3<<<dim3(ceil_div(n,16),ceil_div(m,16)),dim3(16,16)>>>(c,a,b,m,n,k);
    else if(version==4) gemm_forward_kernel4<<<dim3(ceil_div(n,64),ceil_div(m,64)),dim3(16,16)>>>(c,a,b,m,n,k);
    else if(version==5) gemm_vector_1024<T,false>(c,a,b);
    else if(version==6) gemm_vector_1024<T,true>(c,a,b);
    else if(version==7) gemm_forward_kernel7<T,128,32,16,8><<<dim3(32,8),128>>>(c,a,b);
    else if(version==8) {
        if constexpr (!std::is_same_v<T,float>) gemm_forward_kernel8<<<dim3(16,16),128>>>(c,a,b);
        else throw std::runtime_error("kernel8 supports fp16/bf16, not fp32/TF32");
    }
    else if(version==9) {
        if constexpr (std::is_same_v<T,float>) {
            // FP32: 64x64 tile，每线程 8x4；两套 shared 和两套 register 操作数。
            gemm_forward_vector_kernel<T,64,64,16,8,4,true,true><<<dim3(16,16),128>>>(c,a,b);
        } else {
            // 低精度: 32x64 tile，4 warps；每 warp 16x32，XOR + ldmatrix + MMA。
            gemm_forward_kernel9_mma<T><<<dim3(16,32),128>>>(c,a,b);
        }
    }
    else throw std::runtime_error("kernel not implemented yet");
    CUDA_CHECK(cudaGetLastError());
}
