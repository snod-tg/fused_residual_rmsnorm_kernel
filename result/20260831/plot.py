import re
import matplotlib.pyplot as plt
import numpy as np

# 假设数据保存在文件 'rmsnorm_forward.txt' 中
filename = 'rmsnorm_forward.txt'

# 读取文件内容
with open(filename, 'r') as f:
    lines = f.readlines()

# 数据结构：data[(kernel, precision, block_size)] = (time_ms, bandwidth_gbs)
data = {}
current_kernel = None

for line in lines:
    line = line.strip()
    # 检测 kernel 头部
    m = re.match(r'Using kernel (\d+)', line)
    if m:
        current_kernel = int(m.group(1))
        continue

    # 检测精度和性能行，如：
    # [fp32] block_size   32 | time 0.7514 ms | bandwidth 66.98 GB/s
    m = re.match(r'\[(fp32|fp16|bf16)\]\s+block_size\s+(\d+)\s+\|\s+time\s+([\d.]+)\s+ms\s+\|\s+bandwidth\s+([\d.]+)\s+GB/s', line)
    if m and current_kernel is not None:
        prec = m.group(1)
        bs = int(m.group(2))
        time_ms = float(m.group(3))
        bw = float(m.group(4))
        data[(current_kernel, prec, bs)] = (time_ms, bw)

# 现在整理成绘图所需的结构
# 按精度分组，每个精度内 kernel -> {block_size: bandwidth}
precisions = ['fp32', 'fp16', 'bf16']
kernel_list = sorted({k for (k, p, bs) in data.keys()})

# 构建每个精度的数据
plot_data = {p: {} for p in precisions}
for (k, p, bs), (_, bw) in data.items():
    if p not in plot_data:
        continue
    if k not in plot_data[p]:
        plot_data[p][k] = {}
    plot_data[p][k][bs] = bw

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('RMSNorm Forward Kernel Bandwidth Comparison', fontsize=16)

# 颜色循环
colors = plt.cm.tab10(np.linspace(0, 1, len(kernel_list)))

for idx, (prec, ax) in enumerate(zip(precisions, axes)):
    # 获取该精度下所有 kernel
    kernels = sorted(plot_data[prec].keys())
    if not kernels:
        ax.set_title(f'{prec.upper()} (no data)')
        continue
    ax.set_title(f'{prec.upper()}')
    ax.set_xlabel('Block Size')
    if idx == 0:
        ax.set_ylabel('Bandwidth (GB/s)')
    
    for i, k in enumerate(kernels):
        bs_list = sorted(plot_data[prec][k].keys())
        bw_list = [plot_data[prec][k][bs] for bs in bs_list]
        ax.plot(bs_list, bw_list, marker='o', label=f'Kernel {k}', color=colors[i % len(colors)])
    ax.set_xscale('log', base=2)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(loc='best')

plt.tight_layout()
plt.savefig('rmsnorm_bandwidth.png', dpi=300)
plt.show()