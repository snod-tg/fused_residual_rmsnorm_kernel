# RMSNorm Forward 逐版本 ncu 分析与优化历程（kernel1–kernel10）

本文在 20260901 的 kernel1–kernel8 分析基础上，补充 kernel9 shared-memory single-pass 与
kernel10 register-cached single-pass 版本，并结合
20260831 与 20260902 的 benchmark 结果重新判断当前最优实现。综合带宽曲线见
[rmsnorm_forward.png](rmsnorm_forward.png)。

## 0. 环境、口径与指标

- GPU：RTX 3060 Laptop，Compute Capability 8.6，30 个 SM。
- 形状：`N=8192`（行），`C=768`（通道）。ncu 对比统一使用 **Block Size = 256**。
- benchmark 图中的 Bandwidth 是按算子必要流量 `读 x + 写 y` 计算的**有效带宽**；它不等于
  ncu 统计的实际 DRAM 吞吐量。kernel1–kernel8 会再次读取 x，kernel9/10 才是 single-pass。
- `Duration (us)` 越小越好；`DRAM Throughput (%)` 越高，越接近显存带宽上限。
- `Compute (SM) Throughput (%)` 反映计算管线利用率；低 Compute、高 DRAM 通常说明带宽受限。
- `Achieved Occupancy (%)` 是实际活跃 warp 占比；共享内存或寄存器过多会限制 occupancy。
- `No Eligible (%)` 和 long scoreboard stall 可用于判断 warp 是否经常等待访存依赖。

各版本的优化点：

| kernel | 优化点 |
|---|---|
| 1 | 一线程一行（naive） |
| 2 | 一 warp 一行 + shuffle 归约 |
| 3 | vec2（每次搬 2 个 float） |
| 4 | vec4（每次搬 4 个 float） |
| 5 | 共享内存缓存 weight |
| 6 | grid-stride + `__ldg` + 尾部处理 |
| 7 | 混合精度（模板，标量） |
| 8 | 混合精度 + vec2 |
| 9 | single-pass：共享内存缓存 x，全局 x 只读一次 |
| 10 | FP32 single-pass：寄存器缓存 x + float4，全局 x 只读一次 |

---

## 1. kernel1：一线程一行（naive）

一个线程串行处理一整行：先遍历 C 个通道求平方和，再遍历一遍写 y。相邻线程访问的地址
相隔 `C=768` 个元素，既没有行内并行，也不能形成高效合并访存。

| 指标 | block256 |
|---|---:|
| Duration (us) | 1330.0 |
| DRAM Throughput (%) | 19.98 |
| Compute (SM) Throughput (%) | 3.59 |
| Achieved Occupancy (%) | 18.28 |
| Grid Size / Waves Per SM | 32 / 0.18 |

ncu 提示每个 32-byte sector 平均只使用 4 byte。低并行度与非合并访存叠加，使 kernel1 成为
最慢版本。下一步应让一个 warp 协作处理一行。

## 2. kernel2：一 warp 一行 + shuffle 归约

32 个 lane 分摊一行的通道，相邻 lane 读取相邻地址，再用 `__shfl_xor_sync` 完成行内归约。

| 指标 | block256 |
|---|---:|
| Duration (us) | 237.66 |
| DRAM Throughput (%) | 83.31 |
| Compute (SM) Throughput (%) | 17.05 |
| Achieved Occupancy (%) | 74.44 |
| Registers Per Thread | 39 |

这是收益最大的一步：1330.0 → 237.66 us，约 **5.60× 加速**。DRAM 利用率从 19.98% 升到
83.31%，说明合并访存和 warp 归约将算子推进到 memory-bound 区域。

## 3. kernel3：vec2

kernel3 用 `float2` 一次搬运两个元素，减少 load/store 指令数，但没有减少数据字节数。

| 指标 | block256 |
|---|---:|
| Duration (us) | 241.38 |
| DRAM Throughput (%) | 89.02 |
| Achieved Occupancy (%) | 84.31 |
| Registers Per Thread | 37 |

DRAM 利用率升高，但耗时略高于 kernel2。已经带宽受限后，仅减少访存指令并不能降低必须搬运
的数据量。

## 4. kernel4：vec4

kernel4 将向量宽度增至 `float4`（128 bit）。

| 指标 | block256 |
|---|---:|
| Duration (us) | 244.32 |
| DRAM Throughput (%) | 88.18 |
| Achieved Occupancy (%) | 83.63 |
| Registers Per Thread | 40 |

它没有超过 vec2 或标量 warp 版本；更宽向量提高了寄存器需求，但仍未减少总流量。

## 5. kernel5：共享内存缓存 weight

kernel5 把仅 768 个 float（约 3 KB）的 weight 放入共享内存。

| 指标 | block256 |
|---|---:|
| Duration (us) | 239.20 |
| DRAM Throughput (%) | 80.32 |
| Compute (SM) Throughput (%) | 18.95 |
| Achieved Occupancy (%) | 68.10 |

weight 本身很小，原实现已经能从缓存获益；显式共享内存缓存没有消除主要流量，反而占用资源、
降低 occupancy，因此属于该 shape 下的伪优化。

## 6. kernel6：grid-stride + `__ldg` + 尾部处理

kernel6 固定 grid，使用 grid-stride 循环、`__ldg`、float4 与标量尾部处理。

| 指标 | block256 |
|---|---:|
| Duration (us) | 227.87 |
| DRAM Throughput (%) | 87.53 |
| Achieved Occupancy (%) | 58.31 |
| Registers Per Thread | 51 |
| Grid Size / Waves Per SM | 120 / 1.0 |

它比 kernel2–kernel5 略快，但寄存器增至 51，occupancy 降到 58.31%。收益有限，说明 FP32
路径已接近两遍读取 x 的带宽上限。

## 7. kernel7：混合精度标量 warp

kernel7 用模板支持 FP32/FP16/BF16 存储，归约和核心计算仍在 FP32 中完成。

| 指标 | fp32 | fp16 | bf16 |
|---|---:|---:|---:|
| Duration (us) | 227.55 | 99.04 | 98.75 |
| DRAM Throughput (%) | 91.36 | 80.15 | 79.51 |
| Compute (SM) Throughput (%) | 17.82 | 41.68 | 40.64 |
| Achieved Occupancy (%) | 80.59 | 76.68 | 80.49 |
| Registers Per Thread | 39 | 33 | 31 |

FP16/BF16 的存储字节数减半，耗时也从约 227 us 降到约 99 us。这是继 warp 归约之后第二个
决定性优化。kernel7 也是 kernel1–kernel8 中整体最优的低精度版本。

## 8. kernel8：混合精度 + vec2

kernel8 用 `float2`、`half2` 或 `__nv_bfloat162` 一次搬运两个元素。

| 指标 | fp32 | fp16 | bf16 |
|---|---:|---:|---:|
| Duration (us) | 236.86 | 100.58 | 99.17 |
| DRAM Throughput (%) | 89.46 | 83.35 | 84.18 |
| Compute (SM) Throughput (%) | 8.91 | 20.58 | 21.05 |
| Achieved Occupancy (%) | 81.11 | 69.57 | 66.94 |
| Registers Per Thread | 37 | 31 | 30 |

vec2 提高了部分 DRAM 指标，却没有降低耗时；FP16/BF16 反而因为 occupancy 下降而略慢。
再次说明向量化只减少指令数，不改变该实现的主要字节流量。

---

## 9. kernel9：single-pass，共享内存缓存 x

### 9.1 实现逻辑

kernel1–kernel8 都在求平方和时读取一次 x，并在写 y 时再次读取 x。kernel9 为每个 warp 在
动态共享内存中保留一整行 x：

1. 每个 lane 从全局内存读取自己的元素，同时写入共享内存并累加平方和；
2. warp shuffle 归约得到 `mean2` 和 `inv`；
3. 从共享内存读取 x，结合 w 直接写 y，不再第二次访问全局 x。

在 block256 下，一个 block 有 8 个 warp，因此动态共享内存用量为：

- FP32：`8 × 768 × 4 = 24,576 byte`（ncu 显示 24.58 KB）；
- FP16/BF16：`8 × 768 × 2 = 12,288 byte`（ncu 显示 12.29 KB）。

三种精度、六种 block size 的 correctness 检查均通过。

### 9.2 block256 的 ncu 结果

报告：[kernel9_fp32](rmsnorm_forward_kernel9fp32_b256_ncu.txt) ·
[kernel9_fp16](rmsnorm_forward_kernel9fp16_b256_ncu.txt) ·
[kernel9_bf16](rmsnorm_forward_kernel9bf16_b256_ncu.txt)

| 指标 | fp32 | fp16 | bf16 |
|---|---:|---:|---:|
| Duration (us) | **185.18** | 101.66 | 102.53 |
| DRAM Throughput (%) | 81.69 | 74.26 | 75.08 |
| Memory Throughput (GB/s) | 274.14 | 249.15 | 251.95 |
| Compute (SM) Throughput (%) | 26.63 | 48.53 | 48.50 |
| Achieved Occupancy (%) | 67.19 | 97.17 | 95.61 |
| Registers Per Thread | 39 | 39 | 39 |
| Dynamic Shared Memory / Block | 24.58 KB | 12.29 KB | 12.29 KB |
| No Eligible (%) | 90.81 | 77.43 | 77.64 |
| Long scoreboard stall (cycles/warp) | 62.9 | 36.3 | 36.5 |

### 9.3 FP32：single-pass 明确有效

与 kernel7_fp32 的 227.55 us 相比，kernel9_fp32 降至 185.18 us：

- 延迟降低 **18.6%**；
- 吞吐加速约 **1.23×**；
- benchmark 在 block256 下从 kernel7 的 209.63 GB/s 提升到 kernel9 的 271.48 GB/s。

收益来自删除第二次全局 x 读取。代价是每 block 使用 24.58 KB 动态共享内存，将理论
occupancy 限制在 66.67%，实测为 67.19%。ncu 还报告 30,045 次共享内存 bank conflict，约占
共享 load wavefront 的 13.32%；因此 kernel9 尚未达到理想单遍带宽下界。

FP32 当前最明显的等待仍是 long scoreboard：每个 warp 平均有 62.9 cycles 等待 L1TEX
依赖，且 90.81% 的周期没有 eligible warp。下一步应优先减少共享内存冲突和依赖链，而不是
继续增大向量宽度。

### 9.4 FP16/BF16：single-pass 没有带来收益

与 kernel7 相比：

- FP16：99.04 → 101.66 us，kernel9 慢约 **2.6%**；
- BF16：98.75 → 102.53 us，kernel9 慢约 **3.8%**。

低精度下，一行共享内存仅 1.5 KB、block256 的 occupancy 接近 100%，所以限制因素不再是
“共享内存装不下”。更关键的是：共享内存写入/读取和类型转换增加了指令与依赖，寄存器数由
kernel7 的 33/31 增至 39；同时第二遍全局 x 读取可能已有较好的缓存命中，省下的 DRAM 流量
不足以抵消这些额外开销。因此低精度仍应使用 kernel7，而不是 kernel9。

### 9.5 不同 block size 的 benchmark

| dtype | kernel9 最快 block | 最短时间 (ms) | 有效带宽 (GB/s) | block256 时间 (ms) |
|---|---:|---:|---:|---:|
| FP32 | 32 | 0.1786 | 281.85 | 0.1854 |
| FP16 | 64 | 0.0929 | 271.00 | 0.1012 |
| BF16 | 1024 | 0.0929 | 271.04 | 0.1052 |

这说明“block256 的 ncu 结论”不等于“所有 block size 的最终选择”。kernel9 FP32 在 block32
最快；低精度即使改变 block size，也只与 kernel7/8 的最佳结果接近，没有形成稳定优势。

---

## 10. kernel10：FP32 寄存器缓存 + float4 single-pass

### 10.1 实现逻辑

kernel10 只针对当前 FP32、`C=768` 的实测 shape。每个 warp 仍负责一行，但不再像 kernel9
那样把 x 暂存在共享内存，而是让每个 lane 用 6 个 `float4` 保存自己负责的 24 个元素：

1. 6 次合并的 float4 全局读取覆盖整行 x，并在 FP32 中累加平方和；
2. warp shuffle 完成归约并计算 `inv`；
3. 从寄存器中的 `cached_x` 直接计算并写出 y。

因此 kernel10 同时消除了 kernel7 的第二次全局 x 读取，以及 kernel9 的 shared store/load、
动态共享内存和 bank conflict。代价是寄存器数从 39 增至 63。编译器报告 0 spill，说明缓存
确实保留在寄存器中，没有退化为 local-memory 流量。

### 10.2 benchmark 结果

六种 block size 全部通过正确性检查，最大绝对误差为 `2.38e-7`。

| Block Size | Time (ms) | 有效带宽 (GB/s) |
|---:|---:|---:|
| 32 | **0.1725** | **291.72** |
| 64 | 0.1735 | 290.12 |
| 128 | 0.1743 | 288.74 |
| 256 | 0.1751 | 287.45 |
| 512 | 0.1768 | 284.69 |
| 1024 | 0.1780 | 282.70 |

同一份 benchmark 数据中，kernel9_fp32 的 block256 为 0.1854 ms，kernel10 降至
0.1751 ms，延迟降低约 **5.6%**；最佳 block32 从 kernel9 的 0.1786 ms 降至 0.1725 ms。

### 10.3 block256 的 ncu 结果

报告：[kernel10_fp32](rmsnorm_forward_kernel10fp32_b256_ncu.txt)

| 指标 | kernel9_fp32 | kernel10_fp32 | 变化 |
|---|---:|---:|---:|
| Duration (us) | 185.18 | **161.31** | **-12.9%** |
| DRAM Throughput (%) | 81.69 | **91.37** | +9.68 pp |
| Memory Throughput (GB/s) | 274.14 | **306.67** | +11.9% |
| Compute (SM) Throughput (%) | 26.63 | 6.04 | -20.59 pp |
| Registers Per Thread | 39 | 63 | +24 |
| Dynamic Shared Memory / Block | 24.58 KB | **0** | -24.58 KB |
| Achieved Occupancy (%) | 67.19 | 54.36 | -12.83 pp |
| Executed Instructions | 2,785,280 | **1,073,152** | -61.5% |
| No Eligible (%) | 90.81 | 95.94 | +5.13 pp |

kernel10 的关键收益不是提高 occupancy，而是让数据路径更短：去掉共享内存往返后，执行指令
减少 61.5%，DRAM 吞吐升至 306.67 GB/s，block256 的 ncu Duration 降至 161.31 us。

63 个寄存器将理论 occupancy 限制为 66.67%，实测只有 54.36%；`No Eligible` 也升到
95.94%。这说明 kernel10 已非常接近纯带宽上限，剩余优化空间主要是“降低寄存器压力、增加
可隐藏访存延迟的 warp”，而不再是共享内存 bank conflict。

---

## 11. 更新后的最优选择

### block256（与 ncu 报告严格同口径）

| 目标 | 推荐实现 | Duration (us) | 选择理由 |
|---|---|---:|---|
| FP32 | **kernel10_fp32** | **161.31** | 寄存器 single-pass，移除共享内存往返 |
| FP16 | **kernel7_fp16** | **99.04** | kernel10 仅支持 FP32；kernel9 低精度无收益 |
| BF16 | **kernel7_bf16** | **98.75** | kernel10 仅支持 FP32；kernel7 仍是低精度最快版本 |

### 一句话结论

- 必须 FP32：使用 **kernel10**，block256 的 ncu 时间较 kernel9 再降低 12.9%，较 kernel7
  累计降低 29.1%。
- 允许 FP16/BF16：继续使用 **kernel7**；kernel10 是 FP32-only 专用版本。
- kernel9 证明 single-pass 的方向正确，kernel10 进一步证明：对 `C=768`，寄存器缓存比
  共享内存缓存更接近理论上限。

## 12. 理论上限与 kernel10 的位置

RTX 3060 Laptop 的理论显存带宽约为：

```text
(192 bit / 8) × 14 Gbps = 336 GB/s
```

忽略较小的 w 与 mean2 流量，理想 single-pass 至少读一遍 x、写一遍 y：

- FP32：`2 × 8192 × 768 × 4 = 50.33 MB`，理论下界约 **149.8 us**；
- FP16/BF16：`2 × 8192 × 768 × 2 = 25.17 MB`，理论下界约 **74.9 us**。

kernel10_fp32 的 ncu Duration 为 161.31 us，只比 FP32 理论下界高约 **11.5 us（7.7%）**；
实际内存吞吐 306.67 GB/s，达到 336 GB/s 理论峰值的约 **91.3%**。相比之下，kernel9_fp32
为 185.18 us，距离理论下界约 35.4 us。寄存器缓存已经消除了原分析中最主要的可避免流量。

benchmark 的 block256 时间为 175.1 us，与 ncu 单次采样的 161.31 us 存在测量口径差异；
前者是 2000 次运行的平均 CUDA event 时间，后者是 profiler 捕获的一次 kernel Duration。
因此同类比较应保持同一工具、同一 block size，不能直接混用两个数计算加速比。

对 kernel7 这类两遍读取 x 的实现，不能只用“3 倍数组字节数 / 峰值 DRAM 带宽”作为严格
下界，因为第二遍读取可能命中 L2。图中的“有效带宽”统一按必要的 2 倍流量计算，适合比较
端到端效率，但不代表物理 DRAM 流量。

## 13. 下一步优化建议

1. **降低 kernel10 的寄存器数**：当前 63 registers/thread 是 occupancy 的主要限制，可尝试
   分批缓存或调整 unroll/向量宽度，但必须防止重新读取 x 或产生 local-memory spill。
2. **在 48–56 registers/thread 区间做编译约束实验**，同时检查 spill、Duration 和 DRAM
   Throughput；不能只追求 occupancy 数字。
3. **比较 float2 与 float4 的寄存器生命周期**，观察能否提高 eligible warp 比例并维持
   single-pass。
4. **分别保留精度专用最优路径**：FP32 用 kernel10，FP16/BF16 用 kernel7。
5. **扩展 C 与 N 的测试矩阵**。kernel10 当前专用于 `C=768`；其它 shape 会回退到 kernel9，
   最优专用化边界仍需实测。
