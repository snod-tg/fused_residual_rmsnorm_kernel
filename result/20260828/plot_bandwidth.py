import re
import matplotlib.pyplot as plt

def parse_benchmark(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    kernels = []
    current_kernel = None
    recording = False

    for line in lines:
        line = line.strip()
        if line.startswith('Using kernel'):
            kernel_id = line.split()[-1]
            kernels.append({'id': kernel_id, 'data': []})
            recording = False  # 新 kernel 开始，停止记录旧数据
        elif 'Starting benchmarks.' in line:
            recording = True   # 从此行开始记录性能数据
        elif recording and 'block_size' in line and 'bandwidth' in line:
            match = re.search(r'block_size\s+(\d+)\s+.*bandwidth\s+([\d.]+)', line)
            if match and kernels:
                size = int(match.group(1))
                bw = float(match.group(2))
                kernels[-1]['data'].append((size, bw))

    return kernels

def plot_bandwidth(kernels):
    plt.figure(figsize=(10, 6))
    for k in kernels:
        sizes = [d[0] for d in k['data']]
        bws = [d[1] for d in k['data']]
        plt.plot(sizes, bws, marker='o', label=f'Kernel {k["id"]}')

    plt.xlabel('Block Size')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('RMSNorm Forward Kernel Bandwidth Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('bandwidth.png')
    plt.show()

if __name__ == '__main__':
    data = parse_benchmark('rmsnorm_forward.txt')
    if data and all(d['data'] for d in data):
        plot_bandwidth(data)
    else:
        print("未找到性能数据，请检查文件格式或路径。")