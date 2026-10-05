#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <float.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstddef>

// ---- 1. 类型转换：存储类型 <-> float ----
// as_float/from_float 都标 __host__，因为 CPU 参考也要用它转换 half/bf16。
template <typename T> __host__ __device__ float as_float(T v){return (float)v;}
template <> __host__ __device__ float as_float<half>(half v){return __half2float(v);}
template <> __host__ __device__ float as_float<__nv_bfloat16>(__nv_bfloat16 v){return __bfloat162float(v);}

template <typename T> __host__ __device__ T from_float(float v){return (T)v;}
template <> __host__ __device__ half from_float<half>(float v){return __float2half_rn(v);}
template <> __host__ __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float v){return __float2bfloat16_rn(v);}

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e = (call);                                                   \
        if (e != cudaSuccess) {                                                   \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(e));                                  \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

bool is_aligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}


float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, float* h_ptr, size_t count) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
    free(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    return status;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    CUDA_CHECK(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    CUDA_CHECK(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    CUDA_CHECK(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        CUDA_CHECK(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        CUDA_CHECK(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        CUDA_CHECK(cudaEventRecord(stop, nullptr));
        CUDA_CHECK(cudaEventSynchronize(start));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float single_call;
        CUDA_CHECK(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    CUDA_CHECK(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}


// ---- GEMM 共用的分配、参考和计时工具 ----
#define CUBLAS_CHECK(call) do { auto s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s), __FILE__, __LINE__); exit(1); } } while (0)

template<class T> struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count;
    explicit DeviceBuffer(size_t n) : count(n) {
        // 空专家 / 空 token 仍分配一个元素，但不发起零 grid 的 kernel。
        CUDA_CHECK(cudaMalloc(&ptr, std::max(size_t(1), n) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const std::vector<T>& h) {
        if (h.size() != count) throw std::runtime_error("upload size mismatch");
        if (count) CUDA_CHECK(cudaMemcpy(ptr, h.data(), count*sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> h(count);
        if (count) CUDA_CHECK(cudaMemcpy(h.data(), ptr, count*sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void poison() { CUDA_CHECK(cudaMemset(ptr, 0xff, std::max(size_t(1), count)*sizeof(T))); }
};

struct BlasHandle {
    cublasHandle_t handle;
    BlasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, cublasMath_t(CUBLAS_DEFAULT_MATH |
            CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    }
    ~BlasHandle() { cublasDestroy(handle); }
};

template<class T> constexpr cudaDataType_t gemm_dtype() {
    if constexpr (std::is_same_v<T, float>) return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, half>) return CUDA_R_16F;
    else return CUDA_R_16BF;
}

// 行主序 C[M,N] = A[M,K] B[K,N]。
// cuBLAS 为列主序：计算 C^T = B^T A^T，交换 A/B 和 M/N；无需实际转置。
template<class T>
void gemm_cublas(
    cublasHandle_t handle,
    T* c,
    const T* a,
    const T* b,
    int m,
    int n,
    int k
){
    if (!m) return;
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha, b, gemm_dtype<T>(), n, a, gemm_dtype<T>(), k,
        &beta, c, gemm_dtype<T>(), n,
        std::is_same_v<T,float> ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template<class T> std::vector<T> gemm_random(
    size_t n,
    unsigned seed,
    bool zeros = false
){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<T> h(n);
    for (auto& v : h) v = from_float<T>(zeros ? 0.f : dist(rng));
    return h;
}

template<class T> bool gemm_close(
    float got,
    double ref
){
    // 拒绝 NaN/Inf，不能让 fabs(NaN) > tolerance 的 false 掩盖错误。
    double atol = 2e-4;
    double rtol = std::is_same_v<T, float> ? 2e-4 : (std::is_same_v<T, half> ? 1e-3 : 8e-3);
    return std::isfinite(got) && std::isfinite(ref) && std::abs(got-ref) <= atol + rtol*std::abs(ref);
}

template<class T> bool gemm_validate(
    const std::vector<T>& got,
    const std::vector<T>& ref,
    const char* label
){
    double max_abs = 0.; size_t failures = 0;
    if (got.size() != ref.size()) return false;
    for (size_t i=0; i<got.size(); ++i) {
        float g = as_float(got[i]), r = as_float(ref[i]);
        max_abs = std::max(max_abs, double(std::abs(g-r)));
        if (!gemm_close<T>(g, r)) {
            if (failures++ < 3) fprintf(stderr, "%s mismatch[%zu]: got=%g ref=%g\n", label, i, g, r);
        }
    }
    printf("CHECK %s elements=%zu max_abs=%.7g failures=%zu\n", label, got.size(), max_abs, failures);
    return !failures;
}

// 小矩阵全量 CPU double 参考；大矩阵每个专家固定抽查 257 个输出。
// CPU 从实际量化后的 T 输入出发，参考输出也舍入到 T。
template<class T> bool gemm_cpu_check(
    const std::vector<T>& a,
    const std::vector<T>& b,
    const std::vector<T>& c,
    int m,
    int n,
    int k,
    size_t aoff = 0,
    size_t boff = 0,
    size_t coff = 0
){
    const size_t total = size_t(m)*n;
    const size_t samples = total*size_t(k) <= 8000000 ? total : std::min(total, size_t(257));
    for (size_t s=0; s<samples; ++s) {
        size_t idx = samples == total ? s : (s * (total-1) / std::max(size_t(1), samples-1));
        size_t row=idx/n, col=idx%n;
        double acc=0.;
        for (int p=0; p<k; ++p) acc += double(as_float(a[aoff+row*k+p])) * as_float(b[boff+size_t(p)*n+col]);
        float ref = as_float(from_float<T>(float(acc)));
        if (!gemm_close<T>(as_float(c[coff+idx]), ref)) {
            fprintf(stderr, "CPU double mismatch at (%zu,%zu): got=%g ref=%g\n", row,col,as_float(c[coff+idx]),ref);
            return false;
        }
    }
    return true;
}

// 预分配、预热后计时，默认热缓存；所有版本都用同一协议。
// gpu_ms 是 CUDA Event 区间，包括全部 kernel 和 host launch 导致的 stream 空档，
// 不等于所有 kernel duration 的简单求和（cuBLAS loop 为多个 launch）；wall_ms 是
// 连续 repeats 次调用再同步的摊销时间，不是单请求的 p50/p99 延迟。
struct GemmTiming { double gpu_ms, wall_ms; };
template<class F> GemmTiming gemm_benchmark(
    F launch,
    int repeats
){
    for (int i=0; i<5; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    auto begin=std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    for (int i=0; i<repeats; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    auto end=std::chrono::steady_clock::now();
    float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return {ms/repeats, std::chrono::duration<double,std::milli>(end-begin).count()/repeats};
}

struct GemmOptions {
    int kernel=0, m=512, n=512, k=512, repeats=30, experts=8, seed=0;
    std::string dtype="fp32", counts, distribution="balanced";
    bool self_test=false, bench=true;
};
inline int gemm_integer(
    const char* text,
    int lo,
    int hi
){
    size_t pos=0; long long value=std::stoll(text,&pos);
    if (pos != std::strlen(text) || value < lo || value > hi) throw std::runtime_error("integer out of range");
    return int(value);
}
inline GemmOptions gemm_options(
    int argc,
    char** argv,
    int max_kernel = 5,
    int default_dimension = 512
){
    GemmOptions o;
    o.m = o.n = o.k = default_dimension;
    for (int i=1;i<argc;++i) {
        std::string key=argv[i];
        if (key=="--self-test") { o.self_test=true; continue; }
        if (key=="--no-bench") { o.bench=false; continue; }
        if (key=="--help") {
            printf("--kernel 0..%d (0=all) --m M --n N --k K --dtype fp32|fp16|bf16\n"
                   "--repeats R --seed S --self-test --no-bench\n"
                   "Grouped: --experts E --distribution balanced|skewed|empty --counts 0,1,17,...\n", max_kernel);
            exit(0);
        }
        if (++i==argc) throw std::runtime_error("missing option value");
        if (key=="--m") o.m=gemm_integer(argv[i],0,32768);
        else if (key=="--n") o.n=gemm_integer(argv[i],1,16384);
        else if (key=="--k") o.k=gemm_integer(argv[i],1,16384);
        else if (key=="--kernel") o.kernel=gemm_integer(argv[i],0,max_kernel);
        else if (key=="--repeats") o.repeats=gemm_integer(argv[i],1,10000);
        else if (key=="--experts") o.experts=gemm_integer(argv[i],1,256);
        else if (key=="--seed") o.seed=gemm_integer(argv[i],0,1000000);
        else if (key=="--dtype") o.dtype=argv[i];
        else if (key=="--counts") o.counts=argv[i];
        else if (key=="--distribution") o.distribution=argv[i];
        else throw std::runtime_error("unknown option: "+key);
    }
    if (o.dtype!="fp32" && o.dtype!="fp16" && o.dtype!="bf16") throw std::runtime_error("unsupported dtype");
    // 教学 harness 限制一次测试的工作集，避免参数误输导致占满共享 GPU。
    if ((size_t(o.m)*o.k + size_t(o.m)*o.n + size_t(o.experts)*o.k*o.n)*4 > size_t(6)*1024*1024*1024)
        throw std::runtime_error("test working set exceeds 6 GiB");
    return o;
}
inline void gemm_device_info() {
    int device; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p,device));
    printf("DEVICE %s sm_%d%d; row-major; FP32 accumulation; alpha=1 beta=0; warmup=5; hot-cache\n",p.name,p.major,p.minor);
}
