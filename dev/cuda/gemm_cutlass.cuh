#pragma once
#include <memory>
// 可选成熟实现：编译时 -DUSE_CUTLASS -I "$CUTLASS_ROOT/include"。
// 使用 SM80 TensorOp，在 SM89 / RTX 4090D 上运行；不是 Hopper-only TMA/WGMMA。
#ifdef USE_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

template<class T> struct CutlassElement;
template<> struct CutlassElement<half> {using Type=cutlass::half_t;};
template<> struct CutlassElement<__nv_bfloat16> {using Type=cutlass::bfloat16_t;};

inline void cutlass_check(cutlass::Status s) {
    if(s != cutlass::Status::kSuccess) throw std::runtime_error(std::string("CUTLASS: ")+cutlassGetStatusString(s));
}
template<class T> struct CutlassGrouped {
    using Element=typename CutlassElement<T>::Type;
    using Kernel=typename cutlass::gemm::kernel::DefaultGemmGrouped<
        Element,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,8,
        Element,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,8,
        Element,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64,128,32>,cutlass::gemm::GemmShape<32,64,32>,cutlass::gemm::GemmShape<16,8,16>,
        cutlass::epilogue::thread::LinearCombination<Element,8,float,float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,3>::GemmKernel;
    using Gemm=cutlass::gemm::device::GemmGrouped<Kernel>;
    Gemm gemm;
    std::vector<cutlass::gemm::GemmCoord> problems;
    DeviceBuffer<cutlass::gemm::GemmCoord> sizes;
    DeviceBuffer<Element*> pa,pb,pc;
    DeviceBuffer<int64_t> lda,ldb,ldc;
    std::unique_ptr<DeviceBuffer<uint8_t>> workspace;
    bool empty;
    static int active_count(const std::vector<int>& counts) {
        return int(std::count_if(counts.begin(),counts.end(),[](int x){return x>0;}));
    }
    CutlassGrouped(T* c,T* a,T* b,const std::vector<int>& counts,const std::vector<int>& offsets,int n,int k)
        :sizes(active_count(counts)),pa(sizes.count),pb(sizes.count),pc(sizes.count),
         lda(sizes.count),ldb(sizes.count),ldc(sizes.count),empty(!sizes.count) {
        if(empty)return;
        std::vector<Element*> ha,hb,hc;
        std::vector<int64_t> hlda,hldb,hldc;
        for(size_t e=0;e<counts.size();++e) {
            if(!counts[e])continue;
            problems.emplace_back(counts[e],n,k);
            ha.push_back(reinterpret_cast<Element*>(a+size_t(offsets[e])*k));
            hb.push_back(reinterpret_cast<Element*>(b+e*size_t(k)*n));
            hc.push_back(reinterpret_cast<Element*>(c+size_t(offsets[e])*n));
            hlda.push_back(k);hldb.push_back(n);hldc.push_back(n);
        }
        sizes.upload(problems);pa.upload(ha);pb.upload(hb);pc.upload(hc);
        lda.upload(hlda);ldb.upload(hldb);ldc.upload(hldc);
        int blocks=Gemm::sufficient(problems.data(),int(problems.size()));
        if(!blocks)throw std::runtime_error("CUTLASS insufficient resources");
        typename Gemm::Arguments args(sizes.ptr,int(problems.size()),blocks,
            typename Gemm::EpilogueOutputOp::Params(1.f,0.f),
            pa.ptr,pb.ptr,pc.ptr,pc.ptr,lda.ptr,ldb.ptr,ldc.ptr,ldc.ptr,problems.data());
        workspace=std::make_unique<DeviceBuffer<uint8_t>>(Gemm::get_workspace_size(args));
        cutlass_check(gemm.initialize(args,workspace->ptr));
    }
    void run() {if(!empty)cutlass_check(gemm.run());}
};
#endif
