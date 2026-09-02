# Fused Residual RMSNorm Forward：benchmark 与 block256 NCU 分析

## 1. 测试范围

- GPU：RTX 3060 Laptop，CC 8.6；shape 为 `N=8192`、`C=768`。
- NCU 固定 **Block Size = 256**；所有 kernel、dtype 和 block size 均通过 y、z、mean2 对拍。
- 算子在一个 kernel 中完成 `z = round(x + residual)`、RMS 归约、y 写回，因此有效流量口径
  包含 x、residual、z、y 四个大张量。

性能曲线见 [fused_residual_rmsnorm_forward.png](fused_residual_rmsnorm_forward.png)，原始数据见
[fused_residual_rmsnorm_forward.txt](fused_residual_rmsnorm_forward.txt)。

## 2. kernel 演进

| kernel | 实现 |
|---|---|
| 1 | naive：一线程一行 |
| 2 | 一 warp 一行 + shuffle 归约 |
| 3 | float2 |
| 4 | float4 |
| 5 | shared 缓存 weight |
| 6 | grid-stride + `__ldg` + float4/尾部处理 |
| 7 | FP32 计算、FP32/FP16/BF16 存储的标量 warp |
| 8 | 混合精度 + vec2 |
| 9 | C=768 三精度独立专用化：寄存器缓存 z，归约后直接生成 y |

## 3. benchmark 结果

### block256

| 版本 | Time (ms) | 有效带宽 (GB/s) |
|---|---:|---:|
| kernel1 FP32 | 2.1124 | 47.67 |
| kernel2 FP32 | 0.4364 | 230.77 |
| kernel3 FP32 | 0.4323 | 232.93 |
| kernel4 FP32 | **0.4250** | **236.92** |
| kernel5 FP32 | 0.4381 | 229.84 |
| kernel6 FP32 | 0.4247 | 237.11 |
| kernel7 FP32 | 0.4316 | 233.31 |
| kernel8 FP32 | 0.4299 | 234.22 |
| kernel7 FP16 | 0.2131 | 236.31 |
| kernel7 BF16 | 0.2128 | 236.66 |
| kernel8 FP16 | 0.1968 | 255.90 |
| kernel8 BF16 | **0.1960** | **256.98** |
| kernel9 FP32 | **0.3382** | **297.78** |
| kernel9 FP16 | **0.1725** | **291.94** |
| kernel9 BF16 | **0.1717** | **293.38** |

kernel9 打破了 kernel2–8 的带宽平台：FP32 相对旧最佳缩短约 20%，低精度相对 kernel8
再缩短约 12%。全 block 最佳为 kernel9 BF16 block256：0.1717 ms / 293.38 GB/s；
FP16 最佳为 kernel9 block256：0.1725 ms / 291.94 GB/s。

## 4. block256 NCU 结果

| 版本 | Duration (us) | DRAM (%) | Compute (%) | Occupancy (%) | Registers |
|---|---:|---:|---:|---:|---:|
| kernel1 FP32 | 2680.00 | 16.82 | 3.43 | 18.63 | 40 |
| kernel2 FP32 | **410.72** | 91.08 | 13.85 | 86.97 | 40 |
| kernel3 FP32 | 424.06 | 88.83 | 7.28 | 86.53 | 37 |
| kernel4 FP32 | 411.01 | **91.53** | 4.00 | 89.26 | 36 |
| kernel5 FP32 | 412.83 | 90.62 | 14.48 | 87.46 | 40 |
| kernel6 FP32 | 422.40 | 89.39 | 4.24 | 83.94 | 48 |
| kernel7 FP32 | 413.47 | 91.06 | 13.91 | 89.84 | 40 |
| kernel8 FP32 | 415.33 | 90.70 | 7.21 | 86.88 | 37 |
| kernel7 FP16 | 203.52 | 86.62 | 28.43 | 90.62 | 24 |
| kernel7 BF16 | 203.01 | 86.80 | 28.27 | 90.56 | 25 |
| kernel8 FP16 | **185.79** | **89.12** | 16.42 | 78.28 | 24 |
| kernel8 BF16 | 192.22 | 87.29 | 16.14 | 78.38 | 24 |
| kernel9 FP32 | **332.38** | 90.97 | 4.43 | 58.16 | 55 |
| kernel9 FP16 | **161.25** | 91.57 | 15.64 | 90.84 | 40 |
| kernel9 BF16 | **159.94** | **92.26** | 15.93 | 89.79 | 40 |

报告示例：[kernel2](fused_residual_rmsnorm_forward_kernel2_b256_ncu.txt)、
[kernel4](fused_residual_rmsnorm_forward_kernel4_b256_ncu.txt)、
[kernel8 FP16](fused_residual_rmsnorm_forward_kernel8fp16_b256_ncu.txt)。新增报告见
[kernel9 FP32](fused_residual_rmsnorm_forward_kernel9fp32_b256_ncu.txt)、
[kernel9 FP16](fused_residual_rmsnorm_forward_kernel9fp16_b256_ncu.txt) 和
[kernel9 BF16](fused_residual_rmsnorm_forward_kernel9bf16_b256_ncu.txt)。

## 5. 优化结论

1. **warp 归约仍是最大收益。** kernel1 → kernel2 从 2680 us 降至 410.72 us，约 6.5×；
   DRAM 从 16.82% 升至 91.08%。
2. **FP32 已到带宽平台。** kernel2–8 的 NCU 时间相差不足约 3%，DRAM 大多在 89–92%；
   float2/float4、weight shared cache 和 grid-stride 都没有改变主导字节数。
3. **kernel5 的 weight 缓存无明显收益。** w 只有 768 个元素，缓存层已能较好复用；显式
   shared copy 增加同步但不消除 x/residual/z/y 的大流量。
4. **kernel6 寄存器增至 48。** grid-stride、float4 和尾部处理增加资源压力，NCU 反而慢于
   简洁的 kernel2/4。
5. **混合精度直接降低主张量流量。** kernel7 FP16/BF16 约 203 us，较 FP32 减半。
6. **这里 vec2 有效。** 与 RMSNorm forward 的 kernel8 不同，融合前向同时处理 x、residual、
   z 和 y，低精度 vec2 的指令缩减足以抵消 occupancy 从约 91% 降到约 78%，kernel8 FP16
   达到 185.79 us。
7. **kernel9 的寄存器 z 缓存直接减少一次大张量读取。** FP32/FP16/BF16 NCU 分别降至
   332.38/161.25/159.94 us，相对旧最佳加速约 19.1%/13.2%/16.8%。低精度版本仅用
   40 个寄存器并保持约 90% occupancy，因此既减少 DRAM 流量，又没有引入明显资源瓶颈。

## 6. 理论下界

按四个大张量的有效流量，FP32 约 100.70 MB，对 336 GB/s 的理想下界约 **299.7 us**；
FP16/BF16 约 50.37 MB，下界约 **149.9 us**。NCU 最佳 FP32 为 410.72 us、低精度为
159.94 us，分别只高出理论下界约 11% 和 7%。

剩余差距来自求和归约依赖、w/mean2 流量、启动与调度开销，以及硬件无法长期维持峰值带宽。
kernel9 已完成此前建议的 z 缓存。下一步继续压缩收益空间较小，应优先检查 FP32 55 个寄存器
导致 occupancy 约 58% 的影响；低精度已经非常接近当前逻辑流量 roofline。

## 7. 推荐

- 当前 `N=8192, C=768`：FP32/FP16/BF16 均优先选 **kernel9**。
- block256 的最低 benchmark 与 NCU 都是 kernel9 BF16，分别为 0.1717 ms 和 159.94 us。
- 非 C=768 shape 会自动回退到通用 kernel8。
