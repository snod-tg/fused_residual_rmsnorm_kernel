import re
import matplotlib.pyplot as plt

def parse_benchmark(filepath):
    """
    解析 rmsnorm_forward.txt 或 rmsnorm_backward.txt 格式的性能数据。
    自动检测 kernel 分割，提取 block_size 和 bandwidth。
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    kernels = []          # 每个元素为 {'id': kernel_id, 'data': [(size, bw), ...]}
    current_kernel = None

    for line in lines:
        line = line.strip()
        # 检测新 kernel
        if line.startswith('Using kernel'):
            kernel_id = line.split()[-1]
            current_kernel = {'id': kernel_id, 'data': []}
            kernels.append(current_kernel)
        # 如果当前在某个 kernel 上下文中，且行包含 block_size 和 bandwidth
        elif current_kernel is not None:
            if 'block_size' in line and 'bandwidth' in line:
                match = re.search(r'block_size\s+(\d+)\s+.*bandwidth\s+([\d.]+)', line)
                if match:
                    size = int(match.group(1))
                    bw = float(match.group(2))
                    current_kernel['data'].append((size, bw))

    # 只保留有数据的 kernel
    return [k for k in kernels if k['data']]

def plot_bandwidth(kernels, title='RMSNorm Kernel Bandwidth Comparison', save_path=None):
    plt.figure(figsize=(10, 6))
    for k in kernels:
        sizes = [d[0] for d in k['data']]
        bws = [d[1] for d in k['data']]
        plt.plot(sizes, bws, marker='o', label=f'Kernel {k["id"]}')

    plt.xlabel('Block Size')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图表已保存为 {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    # 请根据需要修改文件名
    filename = 'fused_residual_rmsnorm_forward.txt'   # 可改为 'rmsnorm_forward.txt'
    data = parse_benchmark(filename)
    if data:
        plot_bandwidth(data, title=f'Bandwidth Comparison ({filename})', save_path=f'{filename}.png')
    else:
        print("未找到性能数据，请检查文件格式或路径。")