// nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/moe_grouped_gemm_forward.cu -lcublas -o result/build/moe_grouped_gemm_forward
// 可选 CUTLASS baseline：再加 -DUSE_CUTLASS -I "$CUTLASS_ROOT/include"
// ./result/build/moe_grouped_gemm_forward --m 2048 --experts 64 --n 512 --k 512 --dtype fp16
// 只实现前向专家 GEMM，不包含 router / top-k / token gather-scatter / gating。
// 输入已经按专家打包：A[sum(M_e), K], B[E,K,N], C[sum(M_e),N]。
// offsets 是行前缀和；同一个原始 token 的 top-k 副本视作不同的 packed row。
#include <memory>
#include <sstream>
#include "gemm_kernels.cuh"
#include "gemm_cutlass.cuh"

struct GroupTile {int expert,row,col;};
inline std::vector<int> make_counts(const GemmOptions& o) {
    std::vector<int> counts;
    if(!o.counts.empty()) {
        std::istringstream s(o.counts);std::string v;
        if(o.counts.back()==',')throw std::runtime_error("empty count");
        while(std::getline(s,v,','))counts.push_back(gemm_integer(v.c_str(),0,32768));
        if(counts.empty() || counts.size()>256)throw std::runtime_error("counts needs 1..256 experts");
    } else {
        counts.resize(o.experts);
        if(o.distribution=="empty")return counts;
        int start=0,remaining=o.m;
        if(o.distribution=="skewed") {counts[0]=o.m*3/4;remaining-=counts[0];start=1;}
        else if(o.distribution!="balanced")throw std::runtime_error("unknown distribution");
        if(start==o.experts)counts[0]=o.m;
        else for(int e=start;e<o.experts;++e)counts[e]+=remaining/(o.experts-start)+(e-start<remaining%(o.experts-start));
    }
    long long total=0;for(int m:counts)total+=m;
    if(total>32768)throw std::runtime_error("sum(counts) exceeds 32768");
    return counts;
}

inline std::vector<GroupTile> make_tiles(const std::vector<int>& counts,int n,int version) {
    std::vector<GroupTile> tiles;
    for(int e=0;e<int(counts.size());++e) {
        // kernel4 对 <=16 tokens 的专家使用 BM=16，减少小专家的空算。
        int bm=version==3 ? 16 : (counts[e]<=16 ? 16 : 64),bn=version==3?16:64;
        for(int row=0;row<ceil_div(counts[e],bm);++row)
            for(int col=0;col<ceil_div(n,bn);++col)tiles.push_back({e,row,col});
    }
    return tiles;
}

// kernel1：一次 launch，grid.z 是 expert；小专家有被 padding 浪费的 block。
template<class T> __global__ void moe_grouped_gemm_forward_kernel1(
    T* c,
    const T* a,
    const T* b,
    const int* counts,
    const int* offsets,
    int n,
    int k
){
    int e=blockIdx.z,off=offsets[e];
    gemm_native_element(c+size_t(off)*n,a+size_t(off)*k,b+size_t(e)*k*n,
        counts[e],n,k,blockIdx.y*16+threadIdx.y,blockIdx.x*16+threadIdx.x);
}

// kernel3：host 预先生成有效 tile 列表，空专家没有 tile；各专家共享一个 grid。
template<class T> __global__ void moe_grouped_gemm_forward_kernel3(
    T* c,
    const T* a,
    const T* b,
    const int* counts,
    const int* offsets,
    const GroupTile* tiles,
    int n,
    int k
){
    __shared__ float sa[16][16],sb[16][16];
    auto t=tiles[blockIdx.x];int off=offsets[t.expert];
    gemm_shared_tile(c+size_t(off)*n,a+size_t(off)*k,b+size_t(t.expert)*k*n,
        counts[t.expert],n,k,t.row,t.col,sa,sb);
}

// kernel4：复用普通 GEMM 的寄存器分块；小专家使用 16x64，大专家使用 64x64。
template<class T> __global__ void moe_grouped_gemm_forward_kernel4(
    T* c,
    const T* a,
    const T* b,
    const int* counts,
    const int* offsets,
    const GroupTile* tiles,
    int n,
    int k
){
    __shared__ float sa[16][65],sb[16][64];
    auto t=tiles[blockIdx.x];int off=offsets[t.expert],m=counts[t.expert];
    if(m<=16)gemm_register_tile<T,16>(c+size_t(off)*n,a+size_t(off)*k,b+size_t(t.expert)*k*n,m,n,k,t.row,t.col,sa,sb);
    else gemm_register_tile<T,64>(c+size_t(off)*n,a+size_t(off)*k,b+size_t(t.expert)*k*n,m,n,k,t.row,t.col,sa,sb);
}


// kernel6 只列出有效输出 tile。小专家使用较矮的 BM，空专家不参与调度。
template<int BM=32,int BN=64,int SMALL_BN=64>
inline std::vector<GroupTile> make_optimized_tiles(
    const std::vector<int>& counts,
    int n,
    bool low_precision
){
    std::vector<GroupTile> tiles;
    for(int e=0;e<int(counts.size());++e) {
        bool small=low_precision && counts[e]<=16;
        int bm=small?16:BM,bn=small?SMALL_BN:BN;
        for(int row=0;row<ceil_div(counts[e],bm);++row)
            for(int col=0;col<ceil_div(n,bn);++col)tiles.push_back({e,row,col});
    }
    return tiles;
}

// 由实际 counts 选择，不依赖 --distribution 字符串。偏斜时扩大 BN，增加 A 复用。
// 这是本轮扫参后的有限启发式，不保证覆盖所有 MoE 形状的最优配置。
inline bool moe_use_wide_tiles(
    const std::vector<int>& counts,
    int n,
    int k
){
    int active=0,total=0,largest=0;
    for(int m:counts) {active+=m>0;total+=m;largest=std::max(largest,m);}
    return active>1 && largest>=64 && largest*2>total && n>=128 && k>=128;
}

// 对齐路径中一个 chunk 恰好是 8 个 half/bf16。无效 chunk 由硬件补 16 字节零。
// 无效源使用矩阵首地址，不构造越过 expert/分配范围的读取地址。
__device__ __forceinline__ void moe_copy_async_or_zero(
    void* destination,
    const void* source,
    bool valid
){
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(destination));
    int bytes=valid?16:0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
        :: "r"(address),"l"(source),"r"(bytes) : "memory");
}
template<class T,int BM,int BN,int BK,int PA,int PB,bool ALIGNED>
__device__ __forceinline__ void moe_mma_load_tile(
    const T* a,
    const T* b,
    int m,
    int n,
    int k,
    int row0,
    int col0,
    int base,
    T (&sa)[BM][PA],
    T (&sb)[BK][PB]
){
    constexpr int THREADS=128;
    for(int index=threadIdx.x;index<BM*BK/8;index+=THREADS) {
        int row=row0+index/(BK/8),p=base+index%(BK/8)*8;
        T* destination=&sa[index/(BK/8)][gemm_mma_column<true>(index/(BK/8),index%(BK/8)*8)];
        if constexpr (ALIGNED) {
            bool valid=row<m && p<k;
            moe_copy_async_or_zero(destination,valid?a+size_t(row)*k+p:a,valid);
        } else {
            #pragma unroll
            for(int j=0;j<8;++j) destination[j]=(row<m && p+j<k)?a[size_t(row)*k+p+j]:from_float<T>(0.f);
        }
    }
    for(int index=threadIdx.x;index<BK*BN/8;index+=THREADS) {
        int p=base+index/(BN/8),col=col0+index%(BN/8)*8;
        T* destination=&sb[index/(BN/8)][gemm_mma_column<true>(index/(BN/8),index%(BN/8)*8)];
        if constexpr (ALIGNED) {
            bool valid=p<k && col<n;
            moe_copy_async_or_zero(destination,valid?b+size_t(p)*n+col:b,valid);
        } else {
            #pragma unroll
            for(int j=0;j<8;++j) destination[j]=(p<k && col+j<n)?b[size_t(p)*n+col+j]:from_float<T>(0.f);
        }
    }
    if constexpr (ALIGNED)gemm_async_commit();
}
template<class T> __device__ __forceinline__ void moe_store_pair(
    T* c,
    int row,
    int col,
    int m,
    int n,
    float x,
    float y
){
    if(row>=m || col>=n)return;
    // n 为奇数时不能假定下一行首地址仍满足 half2/bfloat162 的 4-byte 对齐。
    if(n%2==0 && col+1<n)gemm_store2(c+size_t(row)*n+col,x,y);
    else {
        c[size_t(row)*n+col]=from_float<T>(x);
        if(col+1<n)c[size_t(row)*n+col+1]=from_float<T>(y);
    }
}

// 复用普通 GEMM kernel9 的 warp 原语；本函数增加任意 M/N/K 与安全的补零/写回。
// shared 从外层统一分配，两个 expert 分支复用同一块空间，不叠加 shared 占用。
template<class T,int BM,int BN,int BK,int WM,int WN,bool ALIGNED>
__device__ __forceinline__ void moe_mma_tile(
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k,
    int row0,
    int col0,
    T* storage
){
    constexpr int PA=((BK+63)/64)*64,PB=((BN+63)/64)*64;
    auto& sa=*reinterpret_cast<T(*)[2][BM][PA]>(storage);
    auto& sb=*reinterpret_cast<T(*)[2][BK][PB]>(storage+2*BM*PA);
    int warp=threadIdx.x/32,lane=threadIdx.x&31;
    int warp_row=warp/(BN/WN)*WM,warp_col=warp%(BN/WN)*WN;
    float acc[WM/16][WN/8][4]={};
    unsigned ar[2][WM/16][4],br[2][WN/8][2];
    moe_mma_load_tile<T,BM,BN,BK,PA,PB,ALIGNED>(a,b,m,n,k,row0,col0,0,sa[0],sb[0]);
    if constexpr(ALIGNED)gemm_async_wait();
    __syncthreads();
    int pool=0;
    for(int base=0;base<k;base+=BK) {
        if(base+BK<k)moe_mma_load_tile<T,BM,BN,BK,PA,PB,ALIGNED>(a,b,m,n,k,row0,col0,base+BK,sa[pool^1],sb[pool^1]);
        gemm_mma_load_fragments<T,BM,BN,BK,WM,WN,PA,PB,true>(ar[0],br[0],sa[pool],sb[pool],warp_row,warp_col,0);
        #pragma unroll
        for(int p=0;p<BK;p+=16) {
            int current=(p/16)&1;
            if(p+16<BK)gemm_mma_load_fragments<T,BM,BN,BK,WM,WN,PA,PB,true>(ar[current^1],br[current^1],sa[pool],sb[pool],warp_row,warp_col,p+16);
            #pragma unroll
            for(int i=0;i<WM/16;++i) {
                #pragma unroll
                for(int j=0;j<WN/8;++j)gemm_mma_16x8x16<T>(acc[i][j],ar[current][i],br[current][j]);
            }
        }
        if(base+BK<k) {
            if constexpr(ALIGNED)gemm_async_wait();
            __syncthreads();pool^=1;
        }
    }
    #pragma unroll
    for(int i=0;i<WM/16;++i) {
        #pragma unroll
        for(int j=0;j<WN/8;++j) {
            int row=row0+warp_row+i*16+lane/4,col=col0+warp_col+j*8+(lane%4)*2;
            moe_store_pair(c,row,col,m,n,acc[i][j][0],acc[i][j][1]);
            moe_store_pair(c,row+8,col,m,n,acc[i][j][2],acc[i][j][3]);
        }
    }
}

// kernel6 low precision：一个 launch，CTA 内的 expert 分支一致，不存在分歧 barrier。
// 默认小专家 16x64，大专家 32x64；偏斜场景扩大 BN，所有配置均为 4 warps。
// PERSISTENT 用 grid-stride 分配 tile，供调参比较，不引入全局 atomic 队列。
template<class T,int BM=32,int BN=64,int WM=16,int WN=32,int SMALL_BN=64,int BK=32,bool PERSISTENT=false>
__global__ void moe_grouped_gemm_forward_kernel6_mma(
    T* __restrict__ c,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const int* counts,
    const int* offsets,
    const GroupTile* tiles,
    int tile_count,
    int n,
    int k
){
    static_assert(!std::is_same_v<T,float> && (BM/WM)*(BN/WN)==4 && SMALL_BN%32==0,"invalid grouped MMA shape");
    constexpr int PA=((BK+63)/64)*64,PB=((BN+63)/64)*64,SMALL_PB=((SMALL_BN+63)/64)*64;
    constexpr int LARGE=2*(BM*PA+BK*PB),SMALL=2*(16*PA+BK*SMALL_PB);
    __shared__ __align__(16) T storage[LARGE>SMALL?LARGE:SMALL];
    for(int index=blockIdx.x;index<tile_count;index+=gridDim.x) {
        auto t=tiles[index];int off=offsets[t.expert],m=counts[t.expert];
        T* ce=c+size_t(off)*n;
        const T* ae=a+size_t(off)*k;
        const T* be=b+size_t(t.expert)*k*n;
        bool aligned=n%8==0 && k%8==0;
        if(m<=16) {
            if(aligned)moe_mma_tile<T,16,SMALL_BN,BK,16,SMALL_BN/4,true>(ce,ae,be,m,n,k,t.row*16,t.col*SMALL_BN,storage);
            else moe_mma_tile<T,16,SMALL_BN,BK,16,SMALL_BN/4,false>(ce,ae,be,m,n,k,t.row*16,t.col*SMALL_BN,storage);
        } else {
            if(aligned)moe_mma_tile<T,BM,BN,BK,WM,WN,true>(ce,ae,be,m,n,k,t.row*BM,t.col*BN,storage);
            else moe_mma_tile<T,BM,BN,BK,WM,WN,false>(ce,ae,be,m,n,k,t.row*BM,t.col*BN,storage);
        }
        if constexpr(PERSISTENT)__syncthreads(); // 所有旧 tile 消费者读完，才能复用输入池。
        else break;
    }
}

// FP32 路径：向量化、shared XOR、global 预取双池与 register 操作数双缓冲。
// 不降低输入精度。非对齐/尾块用标量读写，不向邻居专家越界读取。
template<int BM=32,int BN=64,int BK=16,int THREADS=128>
__device__ __forceinline__ void moe_simt_prefetch(
    const float* a,
    const float* b,
    int m,
    int n,
    int k,
    int row0,
    int col0,
    int base,
    float4* av,
    float4* bv
){
    #pragma unroll
    for(int v=0;v<BM*BK/4/THREADS;++v) {
        int index=threadIdx.x+v*THREADS,row=row0+index/(BK/4),p=base+index%(BK/4)*4;
        float4 value=make_float4(0.f,0.f,0.f,0.f);
        if(row<m) {
            if(k%4==0 && p+3<k)value=*reinterpret_cast<const float4*>(a+size_t(row)*k+p);
            else {
                #pragma unroll
                for(int j=0;j<4;++j)if(p+j<k)(&value.x)[j]=a[size_t(row)*k+p+j];
            }
        }
        av[v]=value;
    }
    #pragma unroll
    for(int v=0;v<BK*BN/4/THREADS;++v) {
        int index=threadIdx.x+v*THREADS,p=base+index/(BN/4),col=col0+index%(BN/4)*4;
        float4 value=make_float4(0.f,0.f,0.f,0.f);
        if(p<k) {
            if(n%4==0 && col+3<n)value=*reinterpret_cast<const float4*>(b+size_t(p)*n+col);
            else {
                #pragma unroll
                for(int j=0;j<4;++j)if(col+j<n)(&value.x)[j]=b[size_t(p)*n+col+j];
            }
        }
        bv[v]=value;
    }
}
__global__ void moe_grouped_gemm_forward_kernel6_fp32(
    float* __restrict__ c,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const int* counts,
    const int* offsets,
    const GroupTile* tiles,
    int n,
    int k
){
    constexpr int BM=32,BN=64,BK=16,TM=4,TN=4,THREADS=128;
    __shared__ __align__(16) float sa[2][BK][BM],sb[2][BK][BN];
    auto t=tiles[blockIdx.x];int off=offsets[t.expert],m=counts[t.expert];
    const float* ae=a+size_t(off)*k;
    const float* be=b+size_t(t.expert)*k*n;
    float* ce=c+size_t(off)*n;
    int row0=t.row*BM,col0=t.col*BN,y=threadIdx.x/(BN/TN),x=threadIdx.x%(BN/TN);
    float acc[TM][TN]={};
    float4 next_a[BM*BK/4/THREADS],next_b[BK*BN/4/THREADS];
    moe_simt_prefetch(ae,be,m,n,k,row0,col0,0,next_a,next_b);
    gemm_publish_tile<BM,BN,BK,THREADS>(next_a,next_b,sa[0],sb[0]);__syncthreads();
    int pool=0;
    for(int base=0;base<k;base+=BK) {
        if(base+BK<k)moe_simt_prefetch(ae,be,m,n,k,row0,col0,base+BK,next_a,next_b);
        float4 ar[2][1],br[2][1];
        gemm_simt_load_fragments<BM,BN,BK,TM,TN>(ar[0],br[0],sa[pool],sb[pool],y,x,0);
        #pragma unroll
        for(int p=0;p<BK;++p) {
            int current=p&1;
            if(p+1<BK)gemm_simt_load_fragments<BM,BN,BK,TM,TN>(ar[current^1],br[current^1],sa[pool],sb[pool],y,x,p+1);
            #pragma unroll
            for(int i=0;i<TM;++i) {
                #pragma unroll
                for(int j=0;j<TN;++j)acc[i][j]=fmaf((&ar[current][0].x)[i],(&br[current][0].x)[j],acc[i][j]);
            }
        }
        if(base+BK<k) {gemm_publish_tile<BM,BN,BK,THREADS>(next_a,next_b,sa[pool^1],sb[pool^1]);__syncthreads();pool^=1;}
    }
    #pragma unroll
    for(int i=0;i<TM;++i) {
        int row=row0+y*4+i,col=col0+x*4;
        if(row<m) {
            if(n%4==0 && col+3<n)gemm_store4(ce+size_t(row)*n+col,make_float4(acc[i][0],acc[i][1],acc[i][2],acc[i][3]));
            else {
                #pragma unroll
                for(int j=0;j<TN;++j)if(col+j<n)ce[size_t(row)*n+col+j]=acc[i][j];
            }
        }
    }
}

// kernel7：固定 counts=[64,96,160,192]x4、FP16、N=K=1024。
// BM/BN/BK=32/128/32，warp=32x32，三层 shared 流水线；所有 M_e 都整除 BM。
inline bool moe_is_fixed_target(const std::vector<int>& counts,int n,int k) {
    constexpr int pattern[]={64,96,160,192};
    if(counts.size()!=16 || n!=1024 || k!=1024)return false;
    for(int e=0;e<16;++e)if(counts[e]!=pattern[e%4])return false;
    return true;
}
inline std::vector<GroupTile> make_fixed_tiles(const std::vector<int>& counts) {
    std::vector<GroupTile> tiles;
    // 同一专家、同一 N tile 的 M tiles 相邻，增加 B 在 L2 中的复用机会。
    for(int e=0;e<int(counts.size());++e)
        for(int col=0;col<1024/128;++col)
            for(int row=0;row<counts[e]/32;++row)tiles.push_back({e,row,col});
    return tiles;
}
// kernel8：固定 E=16、sum(M_e)=2048、N=K=1024；每个专家的 M_e 可以为零或任意尾块。
inline bool moe_is_kernel8_target(const std::vector<int>& counts,int n,int k) {
    if(counts.size()!=16 || n!=1024 || k!=1024)return false;
    int total=0;for(int m:counts)total+=m;
    return total==2048;
}
inline std::vector<GroupTile> make_kernel8_tiles(const std::vector<int>& counts) {
    std::vector<GroupTile> tiles;
    // 保持专家/N/M 次序；向上取整覆盖尾块，M_e=0 自然不会产生任务。
    for(int e=0;e<int(counts.size());++e)
        for(int col=0;col<1024/128;++col)
            for(int row=0;row<ceil_div(counts[e],32);++row)tiles.push_back({e,row,col});
    return tiles;
}
// 对线性 half 地址做 XOR：BK=32 时交换物理行，省掉 PA=64 的 padding。
// 低三位不变，16-byte cp.async 保持对齐、连续；读写使用同一个地址映射。
template<int WIDTH>
__device__ __forceinline__ int moe_compact_shared_index(int row,int col) {
    return (row*WIDTH+col)^((row&7)*8);
}
template<class T,int BM,int BN,int BK>
__device__ __forceinline__ void moe_fixed_copy(
    const T* a,const T* b,int m,int row0,int col0,int base,T* sa,T* sb
){
    for(int index=threadIdx.x;index<BM*BK/8;index+=128) {
        int row=index/(BK/8),p=index%(BK/8)*8;
        bool valid=row0+row<m;
        moe_copy_async_or_zero(sa+moe_compact_shared_index<BK>(row,p),valid?a+size_t(row0+row)*1024+base+p:a,valid);
    }
    for(int index=threadIdx.x;index<BK*BN/8;index+=128) {
        int p=index/(BN/8),col=index%(BN/8)*8;
        gemm_copy_async16(sb+moe_compact_shared_index<BN>(p,col),b+size_t(base+p)*1024+col0+col);
    }
    gemm_async_commit();
}
template<class T,int BM,int BN,int BK,int WM,int WN>
__device__ __forceinline__ void moe_fixed_fragments(
    unsigned (&ar)[WM/16][4],unsigned (&br)[WN/8][2],
    const T* sa,const T* sb,int wr,int wc,int p
){
    int lane=threadIdx.x&31;
    #pragma unroll
    for(int i=0;i<WM/16;++i)
        gemm_ldmatrix_a(ar[i],sa+moe_compact_shared_index<BK>(wr+i*16+(lane&15),p+(lane/16)*8));
    #pragma unroll
    for(int j=0;j<WN/8;j+=2) {
        // 一条 x4.trans 读取两个相邻的 16x8 B fragment，减少 ldmatrix 指令。
        unsigned x[4];unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(
            sb+moe_compact_shared_index<BN>(p+(lane&15),wc+j*8+(lane/16)*8)));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
            :"=r"(x[0]),"=r"(x[1]),"=r"(x[2]),"=r"(x[3]):"r"(address));
        br[j][0]=x[0];br[j][1]=x[1];br[j+1][0]=x[2];br[j+1][1]=x[3];
    }
}

template<int GROUPS>
__device__ __forceinline__ void moe_fixed_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(GROUPS) : "memory");
}
template<class T,int BM,int BN,int BK,int WM,int WN,int STAGES>
__device__ __forceinline__ void moe_fixed_tile(
    T* c,
    const T* a,
    const T* b,
    int m,
    int row0,
    int col0,
    T* storage
){
    constexpr int N=1024,K=1024,PA=BK,PB=BN,TILES=K/BK;
    static_assert(STAGES>=2 && STAGES<=4 && K%BK==0 && N%BN==0);
    auto& sa=*reinterpret_cast<T(*)[STAGES][BM][PA]>(storage);
    auto& sb=*reinterpret_cast<T(*)[STAGES][BK][PB]>(storage+STAGES*BM*PA);
    int warp=threadIdx.x/32,lane=threadIdx.x&31;
    int wr=warp/(BN/WN)*WM,wc=warp%(BN/WN)*WN;
    float acc[WM/16][WN/8][4]={};
    unsigned ar[WM/16][4],br[WN/8][2];
    // 预填 STAGES-1 块，等待第一块，保留最新一组拷贝在途。
    #pragma unroll
    for(int stage=0;stage<STAGES-1;++stage)
        moe_fixed_copy<T,BM,BN,BK>(a,b,m,row0,col0,stage*BK,&sa[stage][0][0],&sb[stage][0][0]);
    moe_fixed_wait<STAGES-2>();__syncthreads();
    int pool=0;
    for(int tile=0;tile<TILES;++tile) {
        int next=tile+STAGES-1,copy_pool=pool?pool-1:STAGES-1;
        if(next<TILES)
            moe_fixed_copy<T,BM,BN,BK>(a,b,m,row0,col0,next*BK,&sa[copy_pool][0][0],&sb[copy_pool][0][0]);
        // 此 warp 分块实测单组操作数寄存器更快。
        #pragma unroll
        for(int p=0;p<BK;p+=16) {
            moe_fixed_fragments<T,BM,BN,BK,WM,WN>(ar,br,&sa[pool][0][0],&sb[pool][0][0],wr,wc,p);
            #pragma unroll
            for(int i=0;i<WM/16;++i) {
                #pragma unroll
                for(int j=0;j<WN/8;++j)gemm_mma_16x8x16<T>(acc[i][j],ar[i],br[j]);
            }
        }
        if(tile+1<TILES) {
            // 收尾必须排空：没有新提交时 wait_group 1 可能留下最后一块未完成。
            if(next<TILES)moe_fixed_wait<STAGES-2>();
            else moe_fixed_wait<0>();
            __syncthreads();
        }
        if(++pool==STAGES)pool=0;
    }
    #pragma unroll
    for(int i=0;i<WM/16;++i) {
        #pragma unroll
        for(int j=0;j<WN/8;++j) {
            int row=row0+wr+i*16+lane/4,col=col0+wc+j*8+(lane%4)*2;
            if(row<m)gemm_store2(c+size_t(row)*N+col,acc[i][j][0],acc[i][j][1]);
            if(row+8<m)gemm_store2(c+size_t(row+8)*N+col,acc[i][j][2],acc[i][j][3]);
        }
    }
}
// 最终保留 32x128x32 / warp 32x32 / 3 stages；其他候选留在实验记录。
__global__ void moe_grouped_gemm_forward_kernel7(
    half* __restrict__ c,
    const half* __restrict__ a,
    const half* __restrict__ b,
    const int* counts,
    const int* offsets,
    const GroupTile* tiles
){
    extern __shared__ __align__(16) unsigned char storage[];
    auto t=tiles[blockIdx.x];int off=offsets[t.expert];
    moe_fixed_tile<half,32,128,32,32,32,3>(c+size_t(off)*1024,a+size_t(off)*1024,
        b+size_t(t.expert)*1024*1024,counts[t.expert],t.row*32,t.col*128,reinterpret_cast<half*>(storage));
}

// 复用 kernel7 的固定 N/K 计算核心；其 A 加载补零和输出行屏蔽支持任意 M 尾块。
__global__ void moe_grouped_gemm_forward_kernel8(
    half* __restrict__ c,
    const half* __restrict__ a,
    const half* __restrict__ b,
    const int* counts,
    const int* offsets,
    const GroupTile* tiles
){
    extern __shared__ __align__(16) unsigned char storage[];
    auto t=tiles[blockIdx.x];int off=offsets[t.expert];
    moe_fixed_tile<half,32,128,32,32,32,3>(c+size_t(off)*1024,a+size_t(off)*1024,
        b+size_t(t.expert)*1024*1024,counts[t.expert],t.row*32,t.col*128,reinterpret_cast<half*>(storage));
}

template<class T> bool run_benchmark(
    const GemmOptions& o,
    bool zeros=false
) {
    auto counts=make_counts(o);std::vector<int> offsets(counts.size()+1,0);
    for(size_t e=0;e<counts.size();++e)offsets[e+1]=offsets[e]+counts[e];
    const int m=offsets.back(),n=o.n,k=o.k,experts=counts.size();
    const int max_m=*std::max_element(counts.begin(),counts.end());
    if((size_t(m)*k+size_t(m)*n+size_t(experts)*k*n)*sizeof(T)>size_t(6)*1024*1024*1024)
        throw std::runtime_error("actual grouped working set exceeds 6 GiB");
    auto a=gemm_random<T>(size_t(m)*k,o.seed,zeros),b=gemm_random<T>(size_t(experts)*k*n,o.seed+1);
    DeviceBuffer<T> da(a.size()),db(b.size()),dc(size_t(m)*n),dr(size_t(m)*n);
    DeviceBuffer<int> dcounts(counts.size()),doffsets(offsets.size());
    da.upload(a);db.upload(b);dcounts.upload(counts);doffsets.upload(offsets);
    BlasHandle blas;
    auto blas_loop=[&](T* output) {
        for(int e=0;e<experts;++e)gemm_cublas(blas.handle,output+size_t(offsets[e])*n,
            da.ptr+size_t(offsets[e])*k,db.ptr+size_t(e)*k*n,counts[e],n,k);
    };
    auto cpu_check=[&](const std::vector<T>& output) {
        for(int e=0;e<experts;++e)if(!gemm_cpu_check(a,b,output,counts[e],n,k,
            size_t(offsets[e])*k,size_t(e)*k*n,size_t(offsets[e])*n))return false;
        return true;
    };
    dr.poison();blas_loop(dr.ptr);auto ref=dr.download();if(!cpu_check(ref))return false;
    printf("GROUPED dtype=%s rows=%d E=%d N=%d K=%d counts=",o.dtype.c_str(),m,experts,n,k);
    for(int e=0;e<experts;++e)printf("%s%d",e?";":"",counts[e]);puts("");
    for(int v=1;v<=8;++v) {
        if(o.kernel && o.kernel!=v)continue;
        if(v==5) {
            bool available=false;
#ifdef USE_CUTLASS
            available=!std::is_same_v<T,float> && n%8==0 && k%8==0;
#endif
            if(!available) {
                if(o.kernel==5)throw std::runtime_error("kernel5 requires USE_CUTLASS, fp16/bf16, N/K multiples of 8");
                puts("SKIP kernel5: requires USE_CUTLASS, fp16/bf16, N/K multiples of 8");continue;
            }
        }
        bool fixed=v==7 && std::is_same_v<T,half> && moe_is_fixed_target(counts,n,k);
        if(v==7)puts(fixed?"DISPATCH kernel7: fixed FP16 32x128x32, 3 stages":"DISPATCH kernel7: fallback to kernel6 for this dtype/shape");
        bool dynamic=v==8 && std::is_same_v<T,half> && moe_is_kernel8_target(counts,n,k);
        if(v==8)puts(dynamic?"DISPATCH kernel8: FP16 E=16 sum(M)=2048 N=K=1024, variable counts":"DISPATCH kernel8: fallback to kernel6 for this dtype/shape");
        bool wide=(v==6 || v==7 || v==8) && !std::is_same_v<T,float> && moe_use_wide_tiles(counts,n,k);
        auto ht=fixed ? make_fixed_tiles(counts) : dynamic ? make_kernel8_tiles(counts) : (v==6 || v==7 || v==8) ? (wide ? make_optimized_tiles<32,128,128>(counts,n,true) : make_optimized_tiles(counts,n,!std::is_same_v<T,float>)) : make_tiles(counts,n,v==3?3:4);
        DeviceBuffer<GroupTile> dt(ht.size());dt.upload(ht);
        std::function<void()> cutlass_run;
#ifdef USE_CUTLASS
        // FP32 参考保持完整 FP32 精度；此处只构造低精度 Tensor Core baseline。
        using LowT=std::conditional_t<std::is_same_v<T,float>,half,T>;
        std::unique_ptr<CutlassGrouped<LowT>> cutlass;
        if(v==5) {
            cutlass=std::make_unique<CutlassGrouped<LowT>>(reinterpret_cast<LowT*>(dc.ptr),
                reinterpret_cast<LowT*>(da.ptr),reinterpret_cast<LowT*>(db.ptr),counts,offsets,n,k);
            cutlass_run=[&]{cutlass->run();};
        }
#endif
        auto launch=[&] {
            if(!m)return;
            if(v==1)moe_grouped_gemm_forward_kernel1<<<dim3(ceil_div(n,16),ceil_div(max_m,16),experts),dim3(16,16)>>>(
                dc.ptr,da.ptr,db.ptr,dcounts.ptr,doffsets.ptr,n,k);
            else if(v==2)blas_loop(dc.ptr);
            else if(v==3)moe_grouped_gemm_forward_kernel3<<<ht.size(),dim3(16,16)>>>(
                dc.ptr,da.ptr,db.ptr,dcounts.ptr,doffsets.ptr,dt.ptr,n,k);
            else if(v==4)moe_grouped_gemm_forward_kernel4<<<ht.size(),dim3(16,16)>>>(
                dc.ptr,da.ptr,db.ptr,dcounts.ptr,doffsets.ptr,dt.ptr,n,k);
            else if(v==7 && fixed) {
                if constexpr(std::is_same_v<T,half>)moe_grouped_gemm_forward_kernel7<<<ht.size(),128,3*(32*32+32*128)*sizeof(half)>>>(dc.ptr,da.ptr,db.ptr,dcounts.ptr,doffsets.ptr,dt.ptr);
            }
            else if(v==8 && dynamic) {
                if constexpr(std::is_same_v<T,half>)moe_grouped_gemm_forward_kernel8<<<ht.size(),128,3*(32*32+32*128)*sizeof(half)>>>(dc.ptr,da.ptr,db.ptr,dcounts.ptr,doffsets.ptr,dt.ptr);
            }
            else if(v==6 || v==7 || v==8) {
                if constexpr(std::is_same_v<T,float>)moe_grouped_gemm_forward_kernel6_fp32<<<ht.size(),128>>>(dc.ptr,da.ptr,db.ptr,dcounts.ptr,doffsets.ptr,dt.ptr,n,k);
                else if(wide)moe_grouped_gemm_forward_kernel6_mma<T,32,128,16,64,128><<<ht.size(),128>>>(dc.ptr,da.ptr,db.ptr,dcounts.ptr,doffsets.ptr,dt.ptr,int(ht.size()),n,k);
                else moe_grouped_gemm_forward_kernel6_mma<T><<<ht.size(),128>>>(dc.ptr,da.ptr,db.ptr,dcounts.ptr,doffsets.ptr,dt.ptr,int(ht.size()),n,k);
            }
            else cutlass_run();
            CUDA_CHECK(cudaGetLastError());
        };
        dc.poison();launch();auto out=dc.download();std::string label="kernel"+std::to_string(v);
        if(!gemm_validate(out,ref,label.c_str()) || !cpu_check(out))return false;
        if(o.bench && m) {
            auto t=gemm_benchmark(launch,o.repeats);
            printf("RESULT grouped,%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.4f\n",o.dtype.c_str(),m,experts,n,k,v,
                t.gpu_ms,t.wall_ms,2.*m*n*k/(t.gpu_ms*1e9));
        }
    }
    return true;
}

template<class T> bool run_dtype(GemmOptions o) {
    if(!o.self_test)return run_benchmark<T>(o);
    o.bench=false;
    const char* patterns[]={"0,0,0","1","0,1,0,17,3","16,33,0,65,127","1,1,1,1,1,1,1,1","256,0,1,0,0,7,0,2"};
    for(const char* pattern:patterns) {
        o.counts=pattern;
        for(auto nk:std::vector<std::pair<int,int>>{{31,19},{64,64},{128,256}}) {
            o.n=nk.first;o.k=nk.second;
            for(int seed:{0,17}){o.seed=seed;if(!run_benchmark<T>(o))return false;}
        }
    }
    o.counts="0,17,3,1";o.n=64;o.k=32;
    if(!run_benchmark<T>(o,true))return false;
    // 覆盖 kernel6 的专家分块切换、向量对齐、M/N/K 尾块和长 K。
    o.counts="0,1,2,7,8,15,16,17,31,32,33,63,64,65,129,0";
    for(auto nk:std::vector<std::pair<int,int>>{{1,1},{8,8},{65,33},{512,512}}) {
        o.n=nk.first;o.k=nk.second;
        for(int seed:{0,17}) {o.seed=seed;if(!run_benchmark<T>(o))return false;}
    }
    o.counts="0,1,0,3,17,0";o.n=31;o.k=16383;
    if(!run_benchmark<T>(o))return false;
    if constexpr(std::is_same_v<T,half>) {
        o.counts="64,96,160,192,64,96,160,192,64,96,160,192,64,96,160,192";o.n=o.k=1024;
        for(int seed:{0,17}){o.seed=seed;if(!run_benchmark<T>(o))return false;}
        if(!run_benchmark<T>(o,true))return false;
        if(o.kernel==0 || o.kernel==8) {
            // 固定总 M 的优化路径：均匀、集中、空专家、小专家和非 32 倍数。
            std::vector<std::vector<int>> patterns(7,std::vector<int>(16,0));
            patterns[0].assign(16,128);
            patterns[1][0]=2048;patterns[2][15]=2048;
            patterns[3][0]=1;patterns[3][15]=2047;
            patterns[4].assign(16,127);patterns[4][15]=143;
            patterns[5]={0,1,2,7,8,15,16,17,31,32,33,63,64,65,129,0};
            int used=0;for(int m:patterns[5])used+=m;patterns[5][15]=2048-used;
            patterns[6][0]=1536;
            for(int e=1;e<16;++e)patterns[6][e]=512/15+(e<=512%15);
            // 随机切分而非只做均匀 multinomial，产生更多空专家和尾块。
            for(int seed:{0,17,43}) {
                std::mt19937 rng(seed);std::uniform_int_distribution<int> pick(0,2048);
                std::vector<int> cuts={0,2048};for(int e=0;e<15;++e)cuts.push_back(pick(rng));
                std::sort(cuts.begin(),cuts.end());std::vector<int> counts(16);
                for(int e=0;e<16;++e)counts[e]=cuts[e+1]-cuts[e];patterns.push_back(counts);
            }
            o.kernel=8;
            for(const auto& counts:patterns) {
                std::ostringstream text;
                for(size_t e=0;e<counts.size();++e)text<<(e?",":"")<<counts[e];
                o.counts=text.str();
                for(int seed:{0,17}){o.seed=seed;if(!run_benchmark<T>(o))return false;}
            }
            if(!run_benchmark<T>(o,true))return false;
        }
    }
    return true;
}
int main(
    int argc,
    char** argv
) {
    try {
        auto o=gemm_options(argc,argv,8);gemm_device_info();bool ok=true;
        if(o.self_test) {
            o.dtype="fp32";ok &= run_dtype<float>(o);
            o.dtype="fp16";ok &= run_dtype<half>(o);
            o.dtype="bf16";ok &= run_dtype<__nv_bfloat16>(o);
        } else if(o.dtype=="fp32")ok=run_dtype<float>(o);
        else if(o.dtype=="fp16")ok=run_dtype<half>(o);
        else ok=run_dtype<__nv_bfloat16>(o);
        puts(ok?"ALL CHECKS PASSED":"CHECK FAILED");return ok?0:1;
    } catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}
}
