# Fused Residual RMSNorm Backward：benchmark 与 block256 NCU 分析

## 1. 测试范围

- GPU：RTX 3060 Laptop，CC 8.6；shape 为 `N=8192`、`C=768`。
- NCU 固定 **Block Size = 256**；所有版本均通过 dx、dresidual 与 dw 对拍。
- 本算子读 dy、z、w，写 dx、dresidual，并跨行归约 dw；相比普通 RMSNorm backward 多一个
  大张量写回，因此必要流量和原子聚合成本都更高。

性能曲线见 [fused_residual_rmsnorm_backward.png](fused_residual_rmsnorm_backward.png)，原始
benchmark 见 [fused_residual_rmsnorm_backward.txt](fused_residual_rmsnorm_backward.txt)。

## 2. kernel 演进

| kernel | 实现 |
|---|---|
| 1 | naive：一线程一行 + 全局 dw 原子 |
| 2 | 一 warp 一行 |
| 3 | block 内 shared dw 聚合 |
| 4 | shared 聚合 + persistent grid-stride |
| 5 | persistent + vec2 |
| 6 | 混合精度 shared 聚合 |
| 7 | 混合精度 persistent + vec2 |
| 8 | C=768 三精度独立专用化：warp 寄存器累计 dw，block 末尾分层归并 |

## 3. benchmark 结果

### block256

| 版本 | Time (ms) | 有效带宽 (GB/s) |
|---|---:|---:|
| kernel1 FP32 | 4.3054 | 29.23 |
| kernel2 FP32 | 0.5226 | 240.83 |
| kernel3 FP32 | **0.5065** | **248.51** |
| kernel4 FP32 | 0.5211 | 241.55 |
| kernel5 FP32 | 0.5145 | 244.65 |
| kernel6 FP32 | 0.5109 | 246.36 |
| kernel7 FP32 | 0.5108 | 246.44 |
| kernel6 FP16 | **0.2524** | **249.39** |
| kernel6 BF16 | 0.2635 | 238.92 |
| kernel7 FP16 | 0.2567 | 245.27 |
| kernel7 BF16 | 0.2531 | 248.77 |
| kernel8 FP32 | **0.3415** | **368.53** |
| kernel8 FP16 | **0.1969** | **319.78** |
| kernel8 BF16 | **0.1753** | **359.18** |

kernel8 在 block256 将 FP32 和 BF16 分别降至 0.3415 和 0.1753 ms；FP16 本次 block256
波动为 0.1969 ms，但 block64 达到 0.1755 ms。全 block 最佳点为 kernel8 BF16 block512：
0.1751 ms / 359.57 GB/s。这个“有效带宽”
略高于 336 GB/s 理论 DRAM 峰值，并不代表物理显存超频：公式按必要字节数计算，缓存命中、
计时波动和未计入/重复流量都会使有效带宽与 NCU 的真实 DRAM 利用率不同。

## 4. block256 NCU 结果

| 版本 | Duration (us) | DRAM (%) | Compute (%) | Occupancy (%) | Registers |
|---|---:|---:|---:|---:|---:|
| kernel1 FP32 | 5330.00 | 19.36 | 4.60 | 18.63 | 38 |
| kernel2 FP32 | 488.35 | 91.97 | 21.33 | 83.55 | 39 |
| kernel3 FP32 | 499.71 | 91.12 | 22.92 | 98.91 | 38 |
| kernel4 FP32 | **478.62** | **93.22** | 22.96 | 92.68 | 39 |
| kernel5 FP32 | 495.46 | 91.86 | 17.18 | 82.33 | 44 |
| kernel6 FP32 | 492.19 | 92.63 | 19.79 | 96.95 | 38 |
| kernel7 FP32 | 497.25 | 90.83 | 18.00 | 78.60 | 44 |
| kernel6 FP16 | 238.66 | 90.98 | 41.56 | 95.93 | 39 |
| kernel6 BF16 | **234.02** | 90.75 | 41.99 | 94.44 | 39 |
| kernel7 FP16 | 248.77 | 87.81 | 31.20 | 85.13 | 47 |
| kernel7 BF16 | 246.85 | 85.19 | 31.29 | 83.59 | 46 |
| kernel8 FP32 | **331.46** | 90.94 | 12.69 | 29.69 | 128 |
| kernel8 FP16 | **162.18** | **90.30** | 30.64 | 30.32 | 114 |
| kernel8 BF16 | **168.54** | 88.06 | 30.11 | 32.69 | 114 |

报告示例：[kernel4](fused_residual_rmsnorm_backward_kernel4_b256_ncu.txt)、
[kernel6 BF16](fused_residual_rmsnorm_backward_kernel6bf16_b256_ncu.txt)、
[kernel7 FP16](fused_residual_rmsnorm_backward_kernel7fp16_b256_ncu.txt)。新增报告见
[kernel8 FP32](fused_residual_rmsnorm_backward_kernel8fp32_b256_ncu.txt)、
[kernel8 FP16](fused_residual_rmsnorm_backward_kernel8fp16_b256_ncu.txt) 和
[kernel8 BF16](fused_residual_rmsnorm_backward_kernel8bf16_b256_ncu.txt)。

## 5. 优化结论

1. **warp 化带来约 10.9× 加速。** kernel1 → kernel2 从 5330 us 降至 488.35 us，同时
   DRAM 从 19.36% 升至 91.97%。
2. **FP32 kernel2–7 已接近同一平台。** 其 NCU Duration 为 478.62–499.71 us，DRAM
   约 91–93%；主要限制是大张量流量和 dw 聚合，而非纯算术。
3. **persistent 在 NCU 中略有收益但不稳定。** kernel4 的单次 NCU 最快，但 benchmark
   block256 是 kernel3 最快。两者差异只有几个百分点，说明排序易受运行频率和采样口径影响。
4. **vec2 再次提高寄存器压力。** kernel5/7 的 registers 增至 44–47，occupancy 降低，
   没有稳定超过对应的 shared 聚合版本。
5. **低精度约 2× 加速。** kernel6 BF16 为 234.02 us，较最佳 FP32 478.62 us 快约 2.05×。
6. **kernel6 优于 kernel7。** 非 persistent shared 聚合保持约 95% occupancy；kernel7 的
   persistent+vec2 只有约 84–85% occupancy，且 DRAM 利用率下降，因此低精度也不应选它。
7. **kernel8 消除了逐行 shared atomic 热点。** 通过每 warp 24 个寄存器累计 dw，并在 block
   末尾分层归并，FP32/FP16/BF16 NCU 相对旧最佳分别加速约 30.7%/32.0%/28.0%。尽管
   registers 增至 114–128、occupancy 降至约 30%，更少的同步和原子竞争仍带来显著净收益。

## 6. 理论下界与剩余瓶颈

按五个大张量的有效流量，FP32 约 125.87 MB，336 GB/s 下界约 **374.6 us**；FP16/BF16
约 62.95 MB，下界估计约 **187.4 us**。kernel8 NCU 为 331.46 us 和 162.18 us，已经低于按
逻辑字节数直接除以 336 GB/s 得到的估计。这不表示 DRAM 超过物理峰值：w、重复读取和写入
路径存在缓存及合并，benchmark 的逻辑流量也不等同于 NCU 实际 DRAM 字节数；应以 NCU 的
约 90% DRAM Throughput 判断硬件利用率。

kernel8 已解决 block 内逐行 atomicAdd，剩余热点是每个 block 最终提交 dw 的全局原子和
两遍 dy/z 读取。继续优化可尝试 partial-dw + 第二阶段归约，但需要证明额外 workspace 与
kernel launch 能抵消当前只剩一次 block 级提交的成本。

## 7. 推荐

- 当前 `N=8192, C=768`、block256：三种精度均优先选 **kernel8**。
- block256 NCU 最快为 kernel8 FP16（162.18 us）；benchmark 最快为 kernel8 BF16
  （0.1753 ms）。FP16 block256 有单次波动，部署前建议多轮复测。
- block1024 因资源组合无法驻留专用 kernel，会自动回退到 kernel6。
