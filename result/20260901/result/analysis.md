# RMSNorm Forward 逐版本 ncu 分析与优化历程

本文按阅读顺序（kernel1 → kernel2 → … → kernel8），对每个 kernel 先说明「它做了什么」，
再分析其 ncu 报告，最后得出当前最好的 kernel、仍存在的不足，以及当前环境下的理论上限。

## 0. 环境与指标速查

- GPU：RTX 3060 Laptop，Compute Capability 8.6，30 个 SM。
- 形状：N=8192（行），C=768（通道）。下面数据以 **Block Size = 256** 为主。
- 关键指标含义：
  - `Duration (us)`：kernel 耗时，越小越好。
  - `DRAM Throughput (%)`：显存带宽利用率，越接近 100% 说明越被带宽限制。
  - `Compute (SM) Throughput (%)`：SM 计算利用率。
  - `Achieved Occupancy (%)`：活跃 warp 占比，太低 = 藏不住访存延迟。
  - `Registers Per Thread`：每线程寄存器数，太高会压低 occupancy。
  - `Waves Per SM`：一个 SM 上要跑几「波」block，太低说明并行度不足。

kernels 与优化点对应关系：

| kernel | 优化点 |
|---|---|
| 1 | 一线程一行（naive） |
| 2 | 一 warp 一行 + shuffle 归约 |
| 3 | vec2（每次搬 2 个） |
| 4 | vec4（每次搬 4 个） |
| 5 | 共享内存缓存 weight |
| 6 | grid-stride + `__ldg` + 尾部处理 |
| 7 | 混合精度（模板，标量） |
| 8 | 混合精度 + vec2 |

---

## 1. kernel1：一线程一行（naive）

**做了什么**：把 CPU 循环直接搬上 GPU。`row = blockIdx.x * blockDim.x + threadIdx.x`，
**一个线程负责一整行**：先串行遍历 C 个通道求 `sum = Σx²`，算 `inv = rsqrt(sum/C+eps)`，
再串行遍历一遍写 `y[c] = x[c] * w[c] * inv`。线程之间只并行「不同行」，行内完全串行。

报告：[kernel1（block256）](rmsnorm_forward_kernel1_b256_ncu.txt) ·
[kernel1（block32）](rmsnorm_forward_kernel1_ncu.txt)

| 指标 | block256 |
|---|---|
| Duration (us) | **1330.0** |
| DRAM Throughput (%) | 19.98 |
| L1/TEX Cache Throughput (%) | 55.72 |
| Compute (SM) Throughput (%) | 3.59 |
| Achieved Occupancy (%) | 18.28 |
| Grid Size / Waves Per SM | 32 / 0.18 |

**ncu 报告分析**：

- **访存不合并**：相邻线程读的地址相隔 `C=768` 个元素，ncu 明确提示「每个 32 字节
  sector 平均只有 4 字节被用到」。带宽被严重浪费，DRAM 只有 20%。
- **并行度极低**：block256 下 Grid Size 只有 32，Waves Per SM 0.18，30 个 SM 连一波都
  填不满，GPU 大量空转。
- 结果：耗时 1330 us，Compute 才 3.6%，纯属「串行 + 乱序访存」双重低效。

→ 改进：让相邻 lane 访问相邻通道（合并访存）+ 一个 warp 并行归约一行。

---

## 2. kernel2：一 warp 一行 + shuffle 归约

**做了什么**：把「一行」交给「一个 warp（32 个线程）」。`lane = threadIdx.x % 32`，
`row = blockIdx.x * (blockDim.x/32) + threadIdx.x/32`。每个 lane 按 `c = lane, lane+32, …`
分担通道（相邻 lane 访问相邻地址 → 合并访存），再用 `__shfl_xor_sync` 做 warp 内并行
归约求出整行的 `sum`。仍是两遍循环：第一遍求 sum，第二遍写 y。

报告：[kernel2（block256）](rmsnorm_forward_kernel2_b256_ncu.txt) ·
[kernel2（block32）](rmsnorm_forward_kernel2_ncu.txt)

| 指标 | block256 |
|---|---|
| Duration (us) | 237.66 |
| DRAM Throughput (%) | 83.31 |
| Compute (SM) Throughput (%) | 17.05 |
| Achieved Occupancy (%) | 74.44 |
| Registers Per Thread | 39 |

**ncu 报告分析**：这是**收益最大的一步**，1330 → 238 us（快 5.6 倍）。DRAM 从 20% 飙升
到 83%，说明合并访存 + warp 并行把带宽真正喂饱了。此时 Compute 只有 17%、DRAM 已 83%，
**算子进入带宽受限（memory-bound）区**。

→ 改进：带宽还没到 100%，试试减少访存指令条数（向量化）。

---

## 3. kernel3：vec2（一次搬 2 个 float）

**做了什么**：在 kernel2 基础上，把访存改成 `float2`（一次搬 2 个元素）。循环步长从
`32` 变 `32*2`，起始从 `lane` 变 `lane*2`，`sum` 累加 `v.x² + v.y²`。归约、两遍循环
逻辑不变。

报告：[kernel3（block256）](rmsnorm_forward_kernel3_b256_ncu.txt)

| 指标 | block256 |
|---|---|
| Duration (us) | 241.38 |
| DRAM Throughput (%) | 89.02 |
| Achieved Occupancy (%) | 84.31 |
| Registers Per Thread | 37 |

**ncu 报告分析**：DRAM 升到 89%，但 Duration 反而略慢（241 vs 238 us）。因为**向量化
只减少访存指令条数，不减少要搬的字节数**。已经带宽受限时减少指令无济于事，反而因
调度开销略慢。

→ 改进：进一步加宽到 128-bit（vec4）看是否有变化。

---

## 4. kernel4：vec4（一次搬 4 个 float）

**做了什么**：把 kernel3 的 `float2` 换成 `float4`，一次搬 4 个 float（128-bit）。循环
步长 `32*4`、起始 `lane*4`，`sum` 累加 `v.x²+v.y²+v.z²+v.w²`。

报告：[kernel4（block256）](rmsnorm_forward_kernel4_b256_ncu.txt)

| 指标 | block256 |
|---|---|
| Duration (us) | 244.32 |
| DRAM Throughput (%) | 88.18 |
| Achieved Occupancy (%) | 83.63 |
| Registers Per Thread | 40 |

**ncu 报告分析**：和 vec2 基本持平（244 us），DRAM 88%。寄存器升到 40，occupancy 略降。
再次验证：**带宽受限下，向量化不带来收益**。

→ 改进：换假设——把 weight 缓存进共享内存，减少重复读。

---

## 5. kernel5：共享内存缓存 weight

**做了什么**：在 kernel2 基础上，把 weight 缓存到共享内存（`extern __shared__ float
shared_w[]`），先 `__syncthreads()` 保证加载完成，写回时用 `shared_w[c]` 替代全局的
`w[c]`。

报告：[kernel5（block256）](rmsnorm_forward_kernel5_b256_ncu.txt)

| 指标 | block256 |
|---|---|
| Duration (us) | 239.20 |
| DRAM Throughput (%) | 80.32 |
| Compute (SM) Throughput (%) | 18.95 |
| Achieved Occupancy (%) | 68.10 |

**ncu 报告分析**：缓存 weight 后 DRAM 反而从 89% 掉到 80%，occupancy 从 84% 掉到 68%，
Duration 没变快。原因：**weight 只有 C=768 个 float（3KB），本来就被 L2 缓存，不是
瓶颈**；共享内存反而挤占了资源、降低了 occupancy。这个方向是伪优化。

→ 改进：换 grid-stride + `__ldg` + 尾部处理。

---

## 6. kernel6：grid-stride + `__ldg` + 尾部处理

**做了什么**：外层加 grid-stride 循环（固定 grid，每个 warp 跨多行），weight 缓存到
共享内存，读用 `__ldg`（只读缓存），并对通道做 float4 向量化 + 尾部标量处理。

报告：[kernel6（block256）](rmsnorm_forward_kernel6_b256_ncu.txt)

| 指标 | block256 |
|---|---|
| Duration (us) | 227.87 |
| DRAM Throughput (%) | 87.53 |
| Achieved Occupancy (%) | 58.31 |
| Registers Per Thread | 51 |
| Grid Size / Waves Per SM | 120 / 1.0 |

**ncu 报告分析**：时间 227.87 us，比 kernel2~5 快一点。但它把 grid 固定成 120 个 block
（Waves 1.0），occupancy 反而掉到 58%、寄存器升到 51——靠 `__ldg`+向量化+尾部处理压了
一点时间，却**牺牲了并行度**，收益有限。

→ 改进：到这里 fp32 已贴带宽墙，唯一能大跃进的只剩「降低数据精度」。

---

## 7. kernel7：混合精度（模板，标量 warp）

**做了什么**：把 kernel2 改成 `template <typename T>`，用 `as_float()`/`from_float()` 在
「低精度存储」和「FP32 计算」之间转换：load 后 `as_float` 转 float 算平方与归约，写回
前 `from_float<T>` 转回 half/bf16。**存储用 2 字节省带宽，归约仍用 FP32 保精度**。

报告：[kernel7_fp32](rmsnorm_forward_kernel7fp32_b256_ncu.txt) ·
[kernel7_fp16](rmsnorm_forward_kernel7fp16_b256_ncu.txt) ·
[kernel7_bf16](rmsnorm_forward_kernel7bf16_b256_ncu.txt)

| 指标 | fp32 | fp16 | bf16 |
|---|---|---|---|
| Duration (us) | 227.55 | **99.04** | **98.75** |
| DRAM Throughput (%) | 91.36 | 80.15 | 79.51 |
| Compute (SM) Throughput (%) | 17.82 | 41.68 | 40.64 |
| Achieved Occupancy (%) | 80.59 | 76.68 | 80.49 |
| Registers Per Thread | 39 | 33 | 31 |

**ncu 报告分析**：这是整个项目的**核心收益**。

- fp32 版（227.55 us）和 kernel2/6 一样，DRAM 91.36%——**已顶到带宽墙**。
- fp16/bf16 版把每个元素从 4 字节压到 2 字节，**读写字节减半**，Duration 直接砍半：
  227 → 99 us。DRAM% 反而「只有」80%，因为数据量减半后绝对带宽需求下降。
- Compute 升到 41%（数据量减半后计算占比相对上升，且 fp16 算术吞吐更高）。

→ 改进：低精度下再试试向量化。

---

## 8. kernel8：混合精度 + vec2

**做了什么**：在 kernel7 基础上加向量化，用 `vec2_of<T>` 把 `T` 映射到 `float2`/`half2`/
`__nv_bfloat162`，一次搬 2 个元素（half/bf16 是 4 字节，fp32 是 8 字节）。

报告：[kernel8_fp32](rmsnorm_forward_kernel8fp32_b256_ncu.txt) ·
[kernel8_fp16](rmsnorm_forward_kernel8fp16_b256_ncu.txt) ·
[kernel8_bf16](rmsnorm_forward_kernel8bf16_b256_ncu.txt)

| 指标 | fp32 | fp16 | bf16 |
|---|---|---|---|
| Duration (us) | 236.86 | 100.58 | 99.17 |
| DRAM Throughput (%) | 89.46 | 83.35 | 84.18 |
| Compute (SM) Throughput (%) | 8.91 | 20.58 | 21.05 |
| Achieved Occupancy (%) | 81.11 | 69.57 | 66.94 |
| Registers Per Thread | 37 | 31 | 30 |

**ncu 报告分析**：vec2 在 fp16/bf16 下 DRAM 升到 83~84%，但 Duration（100.58/99.17 us）
和 kernel7 标量版（99.04/98.75 us）**基本一样甚至略慢**。因为 fp16 下数据量已减半，
vec2 收益更小，还让 occupancy 从 ~78% 掉到 ~67%。**向量化在低精度下也没收益。**

---

## 9. 结论：当前最好的 kernel

| 排名 | kernel | Duration (us) | 说明 |
|---|---|---|---|
| 🥇 最快 | **kernel7_bf16** | **98.75** | 低精度，精度约 2 位有效数字 |
| 🥈 | kernel7_fp16 | 99.04 | 低精度，精度约 3 位有效数字 |
| 🥉 fp32 最快 | kernel7_fp32 | 227.55 | DRAM 91.36%，已到带宽墙 |

**一句话结论**：

- **要速度、能接受低精度** → 用 **kernel7 fp16 / bf16**（~99 us），比 fp32 快 ~2.3 倍。
- **必须 fp32** → 用 **kernel7 fp32**（227.55 us），已经到带宽墙，再优化 fp32 收益很小。
- 向量化（kernel3/4/8）和共享内存缓存 w（kernel5）在这个 shape 上都**没有收益**，
  因为它们不减少「要搬的字节数」，而瓶颈就是带宽。

---

## 10. 仍存在的不足

1. **两遍循环导致 x 读两次**：第一遍求 `sum`、第二遍写 `y`，两次都读 `x`，实际 DRAM
   流量约 3×N×C（而非理想 2×N×C），多出约 50% 带宽浪费。这是**当前最大可优化点**。
2. **没有做 single-pass**：若把一行 `x` 缓存进寄存器/共享内存（C=768 时一行 fp32 仅
   3KB，共享内存装得下），就能只读一次 x，fp32 理论上还能从 227 us 降到 ~150 us。
3. **低精度有精度损失**：bf16 尾数 7 bit（~2 位有效数字）、fp16 10 bit，敏感场景不可用。
4. **kernel6 的 grid-stride 是负优化**：固定 grid=120 导致 Waves 1.0、occupancy 58%。
5. **kernel5 缓存 weight 是伪优化**：weight 太小，L2 早已缓存。
6. **只测了固定 shape**：结论只对 N=8192、C=768、block=256 有效，其它 shape 最优 kernel
   可能不同。

---

## 11. 当前环境下的理论上限怎么算、是多少

### 11.1 先判断是「带宽受限」还是「计算受限」

看 ncu：所有 fp32 版本 `Compute (SM) Throughput` 只有 5~18%，而 `DRAM Throughput` 高达
80~91%。**计算几乎闲着、带宽几乎打满 → memory-bound（带宽受限）算子**。所以理论上限
由显存带宽决定，而不是算力。

### 11.2 峰值带宽是多少

RTX 3060 Laptop：显存位宽 **192-bit**，GDDR6 等效速率 **14 Gbps**。

```
峰值带宽 = 位宽(byte) × 数据率
         = (192 / 8) byte × 14e9 /s
         = 24 × 14e9 = 336e9 byte/s ≈ 336 GB/s
```

（ncu 报告里 `DRAM Frequency 6.99 Ghz`，即 7 GHz × 2 = 14 Gbps，与上面一致。）

### 11.3 理论最短耗时（带宽下界）

```text
最短耗时 = 总搬运字节数 / 峰值带宽
```

RMSNorm 前向最少读一遍 x、写一遍 y（weight/mean2 很小忽略）：

- **fp32**：`2 × N × C × 4 = 2 × 8192 × 768 × 4 = 50.33 MB`
  → `50.33e6 / 336e9 ≈ 149.8 us ≈ 150 us`
- **fp16/bf16**：`2 × N × C × 2 = 25.17 MB`
  → `25.17e6 / 336e9 ≈ 74.9 us ≈ 75 us`

### 11.4 为什么实测比下界慢

实测 fp32 最快 227.55 us，比理想下界 150 us 慢约 1.5 倍，原因是两遍循环把 x 读了两次，
实际最小流量是 `3 × N × C`：

```
fp32: 3 × 8192 × 768 × 4 = 75.5 MB → 75.5e6 / 336e9 ≈ 224.7 us
```

实测 227.55 us ≈ 这个「三遍流量」下界的 99%，说明 **fp32 已经真正顶到带宽墙**。再想快
只有两条路：

1. **消除 x 的二次读（single-pass）**：下界从 3×N×C 降到 2×N×C，fp32 理论能到 ~150 us。
2. **降低精度（fp16/bf16）**：每个元素 4 字节降到 2 字节，下界减半到 ~75 us，实测 ~99 us
   （受两遍循环影响，实际下界 ~112 us，已接近）。

### 11.5 一句话总结理论上限

> 峰值带宽 **336 GB/s**；理想单遍读写理论最短耗时 fp32 ≈ **150 us**、fp16/bf16 ≈
> **75 us**；当前两遍循环的实际带宽下界 fp32 ≈ **225 us**（已基本触顶）、fp16 ≈ **112 us**
> （实测 99 us，接近）。
