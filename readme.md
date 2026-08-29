# my_cuda_kernels

从零手写 **RMSNorm** 与 **Fused Residual RMSNorm** 两个 CUDA 算子，逐步做 warp 归约、
向量化、共享内存聚合、persistent kernel 与混合精度（FP32/FP16/BF16）优化。这是一个
学习项目：每个版本都是完整、可读的实现，配套 CPU 参考、正确性校验和带宽基准。

## 目录结构

```text
dev/                 从零手写的算子（核心，可直接 nvcc 编译）
  common.h           工具：CUDA_CHECK、validate_result、benchmark_kernel、memcpy_convert
  rmsnorm_forward.cu            前向 kernel1~kernel6
  rmsnorm_backward.cu           反向 kernel1~kernel5
  fused_residual_rmsnorm_forward.cu    融合前向 kernel1~kernel6
  fused_residual_rmsnorm_backward.cu   融合反向 kernel1~kernel5
tutorial/
  tutorial/           13 章教程（数学 → CPU → CUDA → 优化 → ncu/nsys → 混合精度）
  code/               与教程配套、可单独编译的样例代码
result/20260828/      实测文本 + 带宽图 + 绘图脚本
```

## 优化版本一览

| 算子 | 方向 | 版本 |
|---|---|---|
| RMSNorm | 前向 | 1 线程一行 → 2 warp 归约 → 3 vec2 → 4 vec4 → 5 共享内存缓存 w → 6 grid-stride+__ldg |
| RMSNorm | 反向 | 1 全局原子 → 2 warp 归约 → 3 block 内共享聚合 → 4 persistent → 5 persistent+vec2 |
| Fused Residual RMSNorm | 前向 / 反向 | 与 RMSNorm 同构，前向多残差相加和 z 写回，反向多 dz 支路 |

## 编译运行（裸命令）

```bash
cd dev
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math rmsnorm_forward.cu -o rmsnorm_forward
./rmsnorm_forward 2          # 数字 1~6 选 kernel；先和 CPU 参考对拍，再测带宽

nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math rmsnorm_backward.cu -o rmsnorm_backward
./rmsnorm_backward 3         # 数字 1~5 选 kernel

nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math fused_residual_rmsnorm_forward.cu -o fused_residual_rmsnorm_forward
./fused_residual_rmsnorm_forward 4

nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math fused_residual_rmsnorm_backward.cu -o fused_residual_rmsnorm_backward
./fused_residual_rmsnorm_backward 5
```

`-arch=sm_86` 换成你 GPU 的计算能力（`nvidia-smi --query-gpu=compute_cap --format=csv`）。
`--use_fast_math` 打开快速数学（近似指令）。

## 正确性

每个 binary 都会先跑「CPU 参考 vs GPU kernel」对拍，`Mismatch` 数量非 0 即失败。CPU 参考
与 GPU kernel 在同文件内，方便逐版本对照公式。更严格的检查：

```bash
compute-sanitizer --tool memcheck    ./rmsnorm_backward 3    # 内存越界
compute-sanitizer --tool racecheck   ./rmsnorm_backward 4    # 数据竞争
compute-sanitizer --tool synccheck   ./rmsnorm_backward 4    # barrier 同步错误
```

注意：反向的 `dw` 是对 N 行求和，CPU 顺序累加 vs GPU `atomicAdd` 乱序累加的舍入不同，
所以 `dw` 的容差要放宽到 `2e-3`（参考用 `double` 累加更严谨）。这是浮点求和顺序误差，
不是 bug。

## Profiling

```bash
ncu --set full --kernel-name-base demangled --kernel-name "regex:rmsnorm_forward_kernel" \
    --launch-count 1 ./rmsnorm_forward 4

nsys profile --trace=cuda,nvtx ./rmsnorm_forward 4
nsys stats --report cuda_gpu_kern_sum report1.nsys-rep
```

详见 [教程第 11 章](tutorial/tutorial/11-profiling.md)。

## 实测结论（RTX 3060 Laptop，FP32）

- 前向（N=8192, C=768）：`warp` 是最大收益，带宽 ~68 → ~234 GB/s；`vec2/vec4` 和
  `grid-stride` 没有再涨（带宽受限，向量化不减少数据量）。
- 反向（N=8192, C=1600）：`warp` 是最大收益，~17 → ~120 GB/s；共享内存聚合 / persistent /
  vec2 三个版本基本持平（~122 GB/s），说明 FP32 下反向也是带宽受限。

结论：这类大输入的 RMSNorm 是 memory-bound 算子，先把访存做对（warp + 合并访存）收益
最大，之后的算法级优化要结合 ncu/nsys 判断是否真的触到了瓶颈。

## 学习路线

从零开始按顺序读 [教程](tutorial/tutorial/README.md)：

1. [数学推导](tutorial/tutorial/01-math.md)
2. [CPU 参考](tutorial/tutorial/02-cpu-reference.md)
3. [第一个 kernel](tutorial/tutorial/03-first-kernel.md)
4. [warp 归约](tutorial/tutorial/04-warp-reduction.md)
5. [向量化](tutorial/tutorial/05-vectorization.md)
6. [反向](tutorial/tutorial/06-backward.md)
7. [persistent](tutorial/tutorial/07-persistent.md)
8. [融合算子](tutorial/tutorial/08-fused.md)
9. [正确性](tutorial/tutorial/09-correctness.md)
10. [计时](tutorial/tutorial/10-timing.md)
11. [ncu/nsys](tutorial/tutorial/11-profiling.md)
12. [面试要点](tutorial/tutorial/12-interview.md)
13. [混合精度](tutorial/tutorial/13-mixed-precision.md)

## 混合精度

`tutorial/code/rmsnorm_mixed_precision.cu` 演示一个模板 kernel 同时支持 FP32/FP16/BF16：
低精度存储省带宽、FP32 归约保精度。编译：

```bash
cd tutorial/code
nvcc -O3 -std=c++17 -arch=sm_86 rmsnorm_mixed_precision.cu -o rmsnorm_mixed_precision
./rmsnorm_mixed_precision
```

详见 [教程第 13 章](tutorial/tutorial/13-mixed-precision.md)。
