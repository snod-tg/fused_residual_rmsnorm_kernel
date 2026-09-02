# RMSNorm Backward：benchmark 与 block256 NCU 分析

## 1. 测试范围与口径

- GPU：RTX 3060 Laptop，Compute Capability 8.6，30 个 SM。
- Shape：`N=8192`、`C=768`；NCU 固定分析 **Block Size = 256**。
- 所有 kernel、三种 dtype、六种 block size 均通过 CPU 对拍。
- benchmark 的 Bandwidth 是按代码中的必要流量公式计算的有效带宽；NCU 的 DRAM
  Throughput 才是硬件采样的 DRAM 利用率。
- benchmark 是 100 次 CUDA event 平均值，NCU Duration 是 profiler 捕获的一次 launch；
  数值接近时不应根据亚百分比差异强行排名。

性能曲线见 [rmsnorm_backward.png](rmsnorm_backward.png)，原始 benchmark 见
[rmsnorm_backward.txt](rmsnorm_backward.txt)。

## 2. kernel 演进

| kernel | 实现 |
|---|---|
| 1 | naive：一线程一行，`dw` 直接全局 atomicAdd |
| 2 | 一 warp 一行 + shuffle 归约，`dw` 仍为全局原子 |
| 3 | block 内 shared `dw` 聚合，再写全局 `dw` |
| 4 | kernel3 + persistent grid-stride |
| 5 | kernel4 + vec2 |
| 6 | 混合精度 shared 聚合，对应 kernel3 |
| 7 | 混合精度 persistent + vec2，对应 kernel5 |
| 8 | C=768 三精度独立专用化：warp 寄存器累计 dw，block 末尾分层归并 |

## 3. benchmark 结果

### block256

| 版本 | Time (ms) | 有效带宽 (GB/s) |
|---|---:|---:|
| kernel1 FP32 | 3.2154 | 23.49 |
| kernel2 FP32 | 0.4264 | 177.17 |
| kernel3 FP32 | 0.4151 | 181.99 |
| kernel4 FP32 | 0.4305 | 175.47 |
| kernel5 FP32 | 0.4241 | 178.10 |
| kernel6 FP32 | **0.4108** | **183.90** |
| kernel7 FP32 | 0.4285 | 176.28 |
| kernel6 FP16 | **0.1985** | **190.38** |
| kernel6 BF16 | **0.1973** | **191.50** |
| kernel7 FP16 | 0.2159 | 175.01 |
| kernel7 BF16 | 0.2121 | 178.15 |
| kernel8 FP32 | **0.2591** | **291.51** |
| kernel8 FP16 | **0.1326** | **285.06** |
| kernel8 BF16 | **0.1320** | **286.29** |

block256 下，kernel8 将 FP32/FP16/BF16 分别降至 0.2591/0.1326/0.1320 ms。相对原先
最佳 kernel6，分别缩短约 37%、33% 和 33%；低精度依然约为 FP32 的一半。

### 全 block 最佳点

- FP32：kernel8 block512，0.2569 ms / 294.08 GB/s。
- FP16：kernel8 block128，0.1321 ms / 286.03 GB/s。
- BF16：kernel8 block256，0.1320 ms / 286.29 GB/s。

图中 kernel1 明显落后；kernel2 以后大部分曲线在 block64–1024 区间趋于平台，说明主要
瓶颈已经从行内串行变成数据搬运与 `dw` 原子聚合。

## 4. block256 NCU 结果

| 版本 | Duration (us) | DRAM (%) | Compute (%) | Occupancy (%) | Registers |
|---|---:|---:|---:|---:|---:|
| kernel1 FP32 | 4520.00 | 10.50 | 4.05 | 17.71 | 38 |
| kernel2 FP32 | 421.12 | 89.34 | 22.20 | 86.07 | 39 |
| kernel3 FP32 | 403.62 | 93.45 | 26.68 | 95.74 | 38 |
| kernel4 FP32 | 408.13 | 92.19 | 25.51 | 95.72 | 39 |
| kernel5 FP32 | 408.26 | 92.74 | 18.65 | 81.77 | 43 |
| kernel6 FP32 | 407.94 | 92.57 | 22.18 | 98.61 | 38 |
| kernel7 FP32 | **403.26** | **93.79** | 19.01 | 79.35 | 43 |
| kernel6 FP16 | 196.16 | 88.79 | 46.41 | 96.55 | 38 |
| kernel6 BF16 | **193.41** | 88.89 | 46.77 | 95.77 | 39 |
| kernel7 FP16 | 200.03 | 91.62 | 36.84 | 96.59 | 40 |
| kernel7 BF16 | 200.74 | 91.11 | 36.35 | 96.53 | 40 |
| kernel8 FP32 | **254.11** | 89.05 | 13.32 | 31.83 | 128 |
| kernel8 FP16 | **122.43** | 89.63 | 33.21 | 31.19 | 114 |
| kernel8 BF16 | **125.41** | 88.19 | 33.93 | 32.25 | 116 |

对应报告均位于本目录，例如
[kernel3](rmsnorm_backward_kernel3_b256_ncu.txt)、
[kernel6 BF16](rmsnorm_backward_kernel6bf16_b256_ncu.txt) 和
[kernel7 FP32](rmsnorm_backward_kernel7fp32_b256_ncu.txt)。新增专用报告见
[kernel8 FP32](rmsnorm_backward_kernel8fp32_b256_ncu.txt)、
[kernel8 FP16](rmsnorm_backward_kernel8fp16_b256_ncu.txt) 和
[kernel8 BF16](rmsnorm_backward_kernel8bf16_b256_ncu.txt)。

## 5. 逐步优化结论

1. **warp 化是决定性优化。** kernel1 → kernel2 从 4520 us 降至 421.12 us，约 10.7×；
   DRAM 从 10.5% 升至 89.34%，说明 naive 的串行行处理和非合并访存是首要问题。
2. **shared `dw` 聚合有效。** kernel3 把同一 block 内大量全局原子合并后再提交，时间继续降至
   403.62 us，DRAM 达 93.45%。
3. **persistent 在 block256 没有形成稳定收益。** kernel4 与 kernel3 基本持平；原子次数
   下降并不足以抵消固定 grid 和调度变化。
4. **vec2 不是主要矛盾。** kernel5 寄存器增至 43、occupancy 降至 81.77%，时间没有改善。
5. **低精度收益清晰。** kernel6 BF16 为 193.41 us，比最快 FP32 约快 2.08×；存储字节减半
   后 Compute 占比升至约 47%，算子由纯带宽受限向混合瓶颈移动。
6. **低精度 kernel7 仍慢于 kernel6。** persistent+vec2 增加的控制和寄存器开销没有换来更低
   Duration，因此 block256 应优先选择 shared 聚合的 kernel6。
7. **kernel8 的分层 dw 归约是第二个决定性优化。** 每个 warp 不再逐行对 shared_dw 做
   atomicAdd，而是在 24 个寄存器中累计各自通道，block 结束时一次性归并。NCU 相对旧最佳
   分别加速 FP32 37.0%、FP16 37.6%、BF16 35.2%。代价是寄存器增至 114–128、occupancy
   降至约 32%，但原子压力下降带来的收益明显更大。

## 6. 理论下界与剩余瓶颈

按 benchmark 口径，FP32 必要流量约 75.54 MB，对 336 GB/s 峰值带宽的理想下界约
**224.8 us**；FP16/BF16 约 37.79 MB，下界约 **112.5 us**。kernel8 实测最佳分别为
254.11 us 和 122.43 us，只比这一理想估计高约 13% 和 9%。

剩余差距主要来自最终跨 block 的 `dw` atomicAdd、两遍 dy/x 读取、warp 归约依赖和较低
occupancy。kernel8 已验证分层归约有效；若继续优化，可比较独立 partial-dw buffer + 第二阶段
归约是否能消除最后的全局原子，但新增 launch 和 workspace 可能抵消收益。

## 7. 推荐

- 当前 `N=8192, C=768`、block256：三种精度均优先选 **kernel8**。
- kernel8 的 FP16 NCU 最快，为 122.43 us；benchmark 中 BF16 略快，为 0.1320 ms。
- block1024 因寄存器/共享内存组合无法驻留专用 kernel，launcher 会安全回退到 kernel6。
