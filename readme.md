# my_cuda_kernels：从数学公式到手写 CUDA 算子

这是一个**学习向**的 CUDA 算子项目：从数学公式出发，先写 CPU 参考，再写朴素 CUDA
kernel，逐步做 warp 归约、向量化、共享内存聚合、persistent kernel 与混合精度
（FP32/FP16/BF16），并用 `ncu`/`nsys` 解释每一步为什么快/慢。

包含两个算子（及其反向）：

- **RMSNorm**：`y = x * w / sqrt(mean(x^2) + eps)`
- **Fused Residual RMSNorm**：`z = round(x + residual); y = rmsnorm(z, w)`

## 目录结构

```text
dev/                  4 个可独立 nvcc 编译的算子程序（核心）
  common.h            工具：CUDA_CHECK、as_float/from_float、benchmark_kernel、validate_result
  rmsnorm_forward.cu            前向 kernel1~6(FP32) + kernel7~8(混合精度)
  rmsnorm_backward.cu           反向 kernel1~5(FP32) + kernel6~7(混合精度)
  fused_residual_rmsnorm_forward.cu    融合前向 kernel1~6(FP32) + kernel7~8(混合精度)
  fused_residual_rmsnorm_backward.cu   融合反向 kernel1~5(FP32) + kernel6~7(混合精度)
tutorial/
  tutorial/           13 章教程（数学 → CPU → CUDA → 优化 → ncu/nsys → 混合精度）
  code/               与教程配套、可单独编译的样例代码
result/               实测数据 + 绘图脚本
```

## 从数学到代码的完整路线

### 1. 数学公式

前向（对一行 `x`，通道数 `C`，权重 `w`）：

```text
mean2 = sum(x^2) / C
inv   = 1 / sqrt(mean2 + eps)
y     = x * w * inv
```

反向（上游梯度 `dy`）：

```text
dot  = sum(dy * w * x)
dx   = (dy*w - x*dot*inv^2/C) * inv
dw   = sum_rows(dy * x * inv)        # 跨所有行累加
```

融合算子多一步：`z = round(x + residual)`，反向多一条 `dz` 支路
`dx = dresidual = round(dnorm + dz)`。详见 [教程第 1 章](tutorial/tutorial/01-math.md)。

### 2. CPU 参考实现

每个 `.cu` 文件里都带一个 CPU 参考函数（`*_cpu`），用 `double` 归约以逼近真值，作为
对拍的「标准答案」。混合精度版本用模板 `*_cpu<T>` 复用三种 dtype。

### 3. CUDA 优化阶梯（每个版本都可读、可单独测）

| 算子 | 方向 | 版本演进 |
|---|---|---|
| RMSNorm | 前向 | 1 线程一行 → 2 warp 归约 → 3 vec2 → 4 vec4 → 5 共享内存缓存 w → 6 grid-stride+__ldg → 7 混合精度标量 → 8 混合精度 vec2 |
| RMSNorm | 反向 | 1 全局原子 → 2 warp 归约 → 3 block 共享聚合 → 4 persistent → 5 persistent+vec2 → 6 混合精度 shared → 7 混合精度 persistent+vec2 |
| Fused Residual RMSNorm | 前向/反向 | 同构，前向多残差相加与 z 写回，反向多 dz 支路 |

核心结论（实测）：

- **warp 归约是最大收益**：解决了「串行 + 访存不合并」，前向 ~68→~234 GB/s。
- **向量化不减少数据量**：大输入下 forward 是带宽受限，vec4 反而因寄存器上升略慢。
- **反向 persistent 是并行度 vs 原子竞争的取舍**：block=512 快、block=256 慢。
- **混合精度**：FP16/BF16 存储让读写字节减半，对 memory-bound 算子直接缓解带宽。

## 编译运行（裸命令，无脚本）

```bash
cd dev
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math rmsnorm_forward.cu -o rmsnorm_forward
./rmsnorm_forward 8     # 数字选 kernel：1~6 FP32，7~8 混合精度（默认 7）
```

- `rmsnorm_forward`：`./rmsnorm_forward 7|8`（7 标量 / 8 vec2，会跑 fp32+fp16+bf16）
- `rmsnorm_backward`：`./rmsnorm_backward 6|7`（6 shared / 7 persistent+vec2）
- `fused_residual_rmsnorm_forward`：`./fused_residual_rmsnorm_forward 7|8`
- `fused_residual_rmsnorm_backward`：`./fused_residual_rmsnorm_backward 6|7`

`-arch=sm_86` 换成你的计算能力：`nvidia-smi --query-gpu=compute_cap --format=csv`。

每个程序都会**先和 CPU 参考对拍，再测带宽**。对拍容差按 dtype 分档：
FP32 `3e-4`、FP16 `5e-3`、BF16 `4e-2`；反向的 `dw`（跨行求和）固定 `2e-3`。

## 正确性

```bash
compute-sanitizer --tool memcheck    ./rmsnorm_backward 6    # 内存越界
compute-sanitizer --tool racecheck   ./rmsnorm_backward 7    # 数据竞争
compute-sanitizer --tool synccheck   ./rmsnorm_backward 7    # barrier 同步错误
```

注意：反向 `dw` 是对 N 行求和，CPU 顺序累加 vs GPU `atomicAdd` 乱序累加的舍入不同，
所以参考用 `double` 累加、容差放宽到 `2e-3`。这是浮点求和顺序误差，不是 bug。

## 用 nsys / ncu 做优化

计时告诉你「快/慢」，Profiler 告诉你「为什么」。详见 [教程第 11 章](tutorial/tutorial/11-profiling.md)。

### nsys：看时间线与 kernel 数

```bash
nsys profile --trace=cuda,nvtx -o /tmp/report ./fused_residual_rmsnorm_forward 7
nsys stats --report cuda_gpu_kern_sum /tmp/report.nsys-rep
```

用 `cuda_gpu_kern_sum` 核对 kernel 调用次数和耗时，验证「融合减少启动次数」。

### ncu：看单个 kernel 的硬件指标

```bash
ncu --set full --kernel-name-base demangled \
    --kernel-name "regex:rmsnorm_forward_kernel" --launch-count 1 \
    ./rmsnorm_forward 8
ncu --import <report>.ncu-rep --page details
```

重点看：

| 指标 | 含义 |
|---|---|
| `Achieved Occupancy` | 活跃 warp 占比，太低=藏不住访存延迟 |
| `Registers Per Thread` | 寄存器数，向量化会抬高它、压低 occupancy |
| `DRAM Throughput` | 显存带宽利用率，接近峰值=带宽受限 |
| RED / ATOM 指令数 | 全局原子归约次数 |

本项目的两个关键判断（都能在 ncu 里验证）：

1. **Forward 是带宽受限**：vec128 的 DRAM 吞吐已达峰值约 90%，所以向量化无收益。
2. **Persistent 是并行度 vs 原子竞争的取舍**：block=512 时 RED 指令 12288→720 且变快；
   block=256 时 occupancy 掉到 ~17%，反而变慢。

## 学习路线

从零开始读 [教程](tutorial/tutorial/README.md)：数学 → CPU → 第一个 kernel → warp 归约
→ 向量化 → 反向 → persistent → 融合 → 正确性 → 计时 → ncu/nsys → 面试要点 → 混合精度。

## 实测结果

结果数据在 [result/](result/)，绘图用统一的 [plot_benchmark.py](result/plot_benchmark.py)：

```bash
cd result
python plot_benchmark.py 20260831/rmsnorm_forward.txt
```

数据文件命名约定：`{op}_{direction}.txt`（如 `rmsnorm_forward.txt`），
图命名 `{op}_{direction}.png`。
