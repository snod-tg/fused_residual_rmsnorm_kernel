// nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/gemm_forward.cu -lcublas -o result/build/gemm_forward
// ./result/build/gemm_forward --self-test
// ./result/build/gemm_forward --m 1024 --n 1024 --k 1024 --dtype fp16 --kernel 0
#include "gemm_kernels.cuh"

template<class T> bool run_benchmark(
    const GemmOptions& o,
    bool zeros=false
){
    const int m=o.m,n=o.n,k=o.k;
    auto a=gemm_random<T>(size_t(m)*k,o.seed,zeros), b=gemm_random<T>(size_t(k)*n,o.seed+1);
    DeviceBuffer<T> da(a.size()),db(b.size()),dc(size_t(m)*n),dr(size_t(m)*n);
    da.upload(a); db.upload(b);
    BlasHandle blas;
    dr.poison(); gemm_cublas(blas.handle,dr.ptr,da.ptr,db.ptr,m,n,k);
    auto ref=dr.download();
    if(!gemm_cpu_check(a,b,ref,m,n,k)) return false;
    printf("GEMM dtype=%s M=%d N=%d K=%d seed=%d zero=%d\n",o.dtype.c_str(),m,n,k,o.seed,zeros);
    for(int v=1;v<=9;++v) {
        if(o.kernel && o.kernel!=v) continue;
        if((v>=5 && (m!=1024 || n!=1024 || k!=1024)) || (v==8 && std::is_same_v<T,float>)) {
            if(o.kernel)throw std::runtime_error("kernel5..9 require 1024^3; kernel8 requires fp16/bf16");
            printf("SKIP kernel%d: shape/dtype unsupported\n",v);continue;
        }
        auto launch=[&]{gemm_forward(v,dc.ptr,da.ptr,db.ptr,m,n,k,blas.handle);};
        dc.poison(); launch();
        auto out=dc.download();
        std::string label="kernel"+std::to_string(v);
        if(!gemm_validate(out,ref,label.c_str()) || !gemm_cpu_check(a,b,out,m,n,k)) return false;
        if(o.bench && m) {
            auto t=gemm_benchmark(launch,o.repeats);
            printf("RESULT gemm,%s,%d,%d,%d,%d,%.6f,%.6f,%.4f\n",o.dtype.c_str(),m,n,k,v,
                t.gpu_ms,t.wall_ms,2.*m*n*k/(t.gpu_ms*1e9));
        }
    }
    return true;
}
template<class T> bool run_dtype(
    GemmOptions o
){
    if(!o.self_test) return run_benchmark<T>(o);
    o.bench=false;
    const int shapes[][3]={{0,17,9},{1,1,1},{1,65,33},{17,31,19},{63,65,17},{128,128,128},{256,192,257}};
    for(auto& s:shapes) {
        o.m=s[0];o.n=s[1];o.k=s[2];
        for(int seed:{0,17}) {o.seed=seed;if(!run_benchmark<T>(o))return false;}
    }
    o.m=33;o.n=47;o.k=65;
    if(!run_benchmark<T>(o,true))return false;
    o.m=o.n=o.k=1024;
    for(int seed:{0,17}) {o.seed=seed;if(!run_benchmark<T>(o))return false;}
    return run_benchmark<T>(o,true);
}
int main(
    int argc,
    char** argv
){
    try {
        auto o=gemm_options(argc,argv,9,1024); gemm_device_info(); bool ok=true;
        if(o.self_test) {
            o.dtype="fp32";ok &= run_dtype<float>(o);
            o.dtype="fp16";ok &= run_dtype<half>(o);
            o.dtype="bf16";ok &= run_dtype<__nv_bfloat16>(o);
        } else if(o.dtype=="fp32")ok=run_dtype<float>(o);
        else if(o.dtype=="fp16")ok=run_dtype<half>(o);
        else ok=run_dtype<__nv_bfloat16>(o);
        puts(ok ? "ALL CHECKS PASSED" : "CHECK FAILED"); return ok?0:1;
    } catch(const std::exception& e) {fprintf(stderr,"%s\n",e.what());return 1;}
}
