#!/usr/bin/env python3
"""统一绘图脚本：解析 benchmark 输出并画带宽对比图。

用法：
    python plot_benchmark.py <数据1.txt> [数据2.txt ...] [-o 输出.png]

统一命名约定：
    数据文件：{op}_{direction}.txt   例如 rmsnorm_forward.txt
    图  文件：{op}_{direction}.png   默认取第一个输入文件同名 .png

支持两种 benchmark 输出格式：
    1) 旧 FP32 格式：
         block_size   32 | time 0.7514 ms | bandwidth 68.24 GB/s
    2) 新混合精度格式：
         [fp32] block_size   32 | time 0.7514 ms | bandwidth 68.24 GB/s
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")  # 无 GUI 也能保存
import matplotlib.pyplot as plt
import numpy as np

DTYPES = ["fp32", "fp16", "bf16"]
LINESTYLE = {"fp32": "-", "fp16": "--", "bf16": "-."}
MARKER = {"fp32": "o", "fp16": "s", "bf16": "^"}


def parse(path):
    """返回 {(kernel_id, dtype): {block_size: bandwidth}}"""
    series = {}
    cur_kernel = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Using kernel (\d+)", line)
            if m:
                cur_kernel = int(m.group(1))
                continue
            # 混合精度格式：[fp32|fp16|bf16] block_size .. | bandwidth .. GB/s
            m = re.match(
                r"\[(fp32|fp16|bf16)\]\s+block_size\s+(\d+)\s+\|\s+"
                r"time\s+[\d.]+\s+ms\s+\|\s+bandwidth\s+([\d.]+)\s+GB/s",
                line,
            )
            if m and cur_kernel is not None:
                dtype, bs, bw = m.group(1), int(m.group(2)), float(m.group(3))
                series.setdefault((cur_kernel, dtype), {})[bs] = bw
                continue
            # 旧 FP32 格式：block_size .. | bandwidth .. GB/s
            m = re.match(
                r"block_size\s+(\d+)\s+\|\s+time\s+[\d.]+\s+ms\s+\|\s+"
                r"bandwidth\s+([\d.]+)\s+GB/s",
                line,
            )
            if m and cur_kernel is not None:
                bs, bw = int(m.group(1)), float(m.group(2))
                series.setdefault((cur_kernel, "fp32"), {})[bs] = bw
    return series


def main():
    ap = argparse.ArgumentParser(description="统一的 benchmark 带宽绘图")
    ap.add_argument("files", nargs="+", help="benchmark 输出 .txt（可多个）")
    ap.add_argument("-o", "--output", default=None, help="输出 .png 路径")
    args = ap.parse_args()

    merged = {}
    for path in args.files:
        for key, val in parse(path).items():
            merged.setdefault(key, {}).update(val)

    kernels = sorted({k for (k, _) in merged})
    colors = plt.cm.tab10(np.linspace(0, 1, len(kernels)))

    # 统一风格
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, k in enumerate(kernels):
        for dtype in DTYPES:
            if (k, dtype) not in merged:
                continue
            bs_list = sorted(merged[(k, dtype)])
            bw_list = [merged[(k, dtype)][b] for b in bs_list]
            ax.plot(
                bs_list,
                bw_list,
                linestyle=LINESTYLE[dtype],
                marker=MARKER[dtype],
                color=colors[i % len(colors)],
                linewidth=2,
                markersize=7,
                label=f"kernel {k} [{dtype}]",
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Block Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Kernel Bandwidth Comparison")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    out = args.output or args.files[0].replace(".txt", ".png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
