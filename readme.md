# My CUDA Kernels

从数学公式和参考实现出发，实践 CUDA / TileLang 算子优化。保留朴素版与各阶段 kernel，比较向量化、warp 归约、寄存器复用、共享内存布局、异步拷贝、流水线和分组调度的实际收益。

项目包含 RMSNorm、Fused Residual RMSNorm 的前向/反向，以及普通 GEMM 和 MoE Grouped GEMM 前向。手写 CUDA 支持 FP32/FP16/BF16；TileLang 示例目前使用 FP16 存储、FP32 归约。GEMM 使用 cuBLAS 参考，Grouped GEMM 另有可选 CUTLASS 对照。

## 目录

```text
dev/
  cuda/       # 手写 CUDA 算子与共用头文件
  tilelang/   # 四个 TileLang 算子示例
analysis/     # 十份算子分析报告
figures/      # 报告引用的统一风格图表
.gitignore
readme.md
```

## 分析入口

| 实现 | 算子 | 报告 |
|---|---|---|
| CUDA | RMSNorm forward | [分析](analysis/cuda_rmsnorm_forward.md) |
| CUDA | RMSNorm backward | [分析](analysis/cuda_rmsnorm_backward.md) |
| CUDA | Fused Residual RMSNorm forward | [分析](analysis/cuda_fused_residual_rmsnorm_forward.md) |
| CUDA | Fused Residual RMSNorm backward | [分析](analysis/cuda_fused_residual_rmsnorm_backward.md) |
| TileLang | RMSNorm forward | [分析](analysis/tilelang_rmsnorm_forward.md) |
| TileLang | RMSNorm backward | [分析](analysis/tilelang_rmsnorm_backward.md) |
| TileLang | Fused Residual RMSNorm forward | [分析](analysis/tilelang_fused_residual_rmsnorm_forward.md) |
| TileLang | Fused Residual RMSNorm backward | [分析](analysis/tilelang_fused_residual_rmsnorm_backward.md) |
| CUDA | GEMM，固定 1024³ | [分析](analysis/gemm.md) |
| CUDA | MoE Grouped GEMM，固定不均匀 counts | [分析](analysis/grouped_gemm.md) |

各报告注明形状、精度、硬件、计时方法、验证范围和数据日期。3060 Laptop 的历史 RMSNorm 数据与 4090D 的 GEMM / TileLang 数据分别呈现；不同测量方法或硬件的数据不直接混算加速比。

## 编译与运行

CUDA 示例使用 C++17。4090D 使用 `sm_89`，3060 Laptop 使用 `sm_86`。

```bash
mkdir -p result/build
nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/rmsnorm_forward.cu -o result/build/rmsnorm_forward
./result/build/rmsnorm_forward 10

nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/gemm_forward.cu -lcublas -o result/build/gemm_forward
./result/build/gemm_forward --dtype fp16 --kernel 9
./result/build/gemm_forward --self-test

nvcc -O3 -std=c++17 -arch=sm_89 dev/cuda/moe_grouped_gemm_forward.cu -lcublas -o result/build/moe_grouped_gemm_forward
./result/build/moe_grouped_gemm_forward --counts 64,96,160,192,64,96,160,192,64,96,160,192,64,96,160,192 --n 1024 --k 1024 --dtype fp16 --kernel 7
./result/build/moe_grouped_gemm_forward --self-test --kernel 7
```

CUTLASS 对照编译时追加 `--expt-relaxed-constexpr -DUSE_CUTLASS -I "$CUTLASS_ROOT/include"`；普通编译无需 CUTLASS。TileLang 示例需要可用的 TileLang/PyTorch/CUDA 环境及匹配的 C++ 工具链：

```bash
python dev/tilelang/rmsnorm_forward.py
python dev/tilelang/rmsnorm_backward.py
python dev/tilelang/fused_residual_rmsnorm_forward.py
python dev/tilelang/fused_residual_rmsnorm_backward.py
```

默认 TileLang backward 示例包含编译成本较高的 BLOCK_N=128；本次报告明确列出已验证的配置范围。

Grouped GEMM 输入已按专家打包，只执行一个专家线性层；端到端 MoE 的 routing、dispatch/combine、激活、通信和反向属于后续实践范围。
