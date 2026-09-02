#!/usr/bin/env python3
"""
用法（在本目录下执行）：
    python3 plot_ncu_metrics.py                          # 自动为每个 Block Size 各画一张图
    python3 plot_ncu_metrics.py --block 256              # 只画 Block Size = 256
    python3 plot_ncu_metrics.py --block 256 --metrics "DRAM Throughput (%),Achieved Occupancy (%)"
    python3 plot_ncu_metrics.py --dir /path/to/result --out out.png

功能：
    解析 ncu `--page details` 输出，对指定 Block Size（batch size）画分组柱状图：
    x 轴 = 不同 kernel，每个子图 = 一个硬件指标，用于对比同一 Block Size 下不同 kernel。

    Duration 单位在 ncu 里不统一（个别是 ms，多数是 us）。脚本会自动统计多数单位，
    并把所有 Duration 统一换算到该多数单位后再画图，子图标题会带上单位。

依赖：matplotlib、numpy。
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # 无显示环境也能保存
import matplotlib.pyplot as plt
import numpy as np

METRICS = [
    ("Memory Throughput (%)", "Memory Throughput"),
    ("DRAM Throughput (%)", "DRAM Throughput"),
    ("L1/TEX Cache Throughput (%)", "L1/TEX Cache Throughput"),
    ("L2 Cache Throughput (%)", "L2 Cache Throughput"),
    ("Compute (SM) Throughput (%)", "Compute (SM) Throughput"),
    ("Achieved Occupancy (%)", "Achieved Occupancy"),
    ("Theoretical Occupancy (%)", "Theoretical Occupancy"),
    ("Registers Per Thread", "Registers Per Thread"),
    ("Grid Size", "Grid Size"),
    ("Waves Per SM", "Waves Per SM"),
]

DTYPE_MAP = {"float": "fp32", "__half": "fp16", "__nv_bfloat16": "bf16"}

TIME_SCALE_S = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def normalize_unit(unit):
    u = unit.lower()
    if u.startswith("n"):
        return "ns"
    if u.startswith("u") or u.startswith("\u00b5"):
        return "us"
    if u.startswith("m"):
        return "ms"
    if u.startswith("s"):
        return "s"
    return u


def get_metric(text, name):
    m = re.search(r"^\s*" + re.escape(name) + r"\b.*\s(\S+)\s*$", text, re.MULTILINE)
    if not m:
        return None
    value = m.group(1)
    try:
        return float(value)
    except ValueError:
        return None


def get_duration_raw(text):
    m = re.search(r"^\s*Duration\s+(\S+)\s+([\d.]+)\s*$", text, re.MULTILINE)
    if not m:
        return None
    return float(m.group(2)), normalize_unit(m.group(1))


def detect_majority_unit(files):
    counts = defaultdict(int)
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw = get_duration_raw(f.read())
        if raw:
            counts[raw[1]] += 1
    return max(counts, key=counts.get) if counts else "us"


def parse_file(path, target_unit):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    header = text.splitlines()[1] if len(text.splitlines()) > 1 else ""

    km = re.search(r"rmsnorm_forward_kernel(\d+)", header)
    kernel_num = int(km.group(1)) if km else None
    dm = re.search(r"kernel\d+<([^>]+)>", header)
    dtype = DTYPE_MAP.get(dm.group(1), dm.group(1)) if dm else "fp32"

    block_size = get_metric(text, "Block Size")
    block_size = int(block_size) if block_size is not None else None

    label = f"kernel{kernel_num}" if kernel_num and kernel_num <= 6 else f"kernel{kernel_num}_{dtype}"

    metrics = {out_name: get_metric(text, metric_name) for out_name, metric_name in METRICS}
    metrics["Block Size"] = block_size
    metrics["_kernel_num"] = kernel_num
    metrics["_dtype"] = dtype

    duration_col = f"Duration ({target_unit})"
    dur = get_duration_raw(text)
    if dur:
        seconds = dur[0] * TIME_SCALE_S[dur[1]]
        metrics[duration_col] = seconds / TIME_SCALE_S[target_unit]
    else:
        metrics[duration_col] = None
    return label, block_size, metrics, duration_col


def plot_block(block_size, rows, metric_names, duration_col, out):
    dtype_order = {"fp32": 0, "fp16": 1, "bf16": 2}
    rows = sorted(rows, key=lambda r: (r[1]["_kernel_num"] or 0, dtype_order.get(r[1]["_dtype"], 9)))
    labels = [r[0] for r in rows]

    n = len(metric_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"RMSNorm Forward Kernel Metrics (Block Size = {block_size})", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, name in enumerate(metric_names):
        ax = axes[i // ncols][i % ncols]
        values = [r[1].get(name) for r in rows]
        numeric = [(l, v) for l, v in zip(labels, values) if isinstance(v, (int, float))]
        if not numeric:
            ax.set_title(name)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            continue
        xs = range(len(numeric))
        bars = ax.bar(xs, [v for _, v in numeric], color=[colors[labels.index(l)] for l, _ in numeric])
        ax.set_xticks(list(xs))
        ax.set_xticklabels([l for l, _ in numeric], rotation=45, ha="right", fontsize=8)
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, (_, v) in zip(bars, numeric):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:g}", ha="center", va="bottom", fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存 {out}")


def main():
    ap = argparse.ArgumentParser(description="同 Block Size 下不同 kernel 的 ncu 指标对比图（Duration 统一到多数单位）")
    ap.add_argument("--dir", default=".", help="ncu .txt 所在目录，默认当前目录")
    ap.add_argument("--block", type=int, default=None, help="只画指定 Block Size，默认所有")
    ap.add_argument("--metrics", default=None, help="要画的指标，逗号分隔，默认全部")
    ap.add_argument("--out", default=None, help="输出 png 路径（只画单个 block 时有效）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "rmsnorm_forward_kernel*.txt")))
    if not files:
        print("没找到 rmsnorm_forward_kernel*.txt 文件，请用 --dir 指定目录")
        return

    target_unit = detect_majority_unit(files)
    duration_col = f"Duration ({target_unit})"
    print(f"Duration 多数单位是 {target_unit}，统一换算到 {target_unit}")

    metric_names = [duration_col] + [name for name, _ in METRICS]
    if args.metrics:
        metric_names = [s.strip() for s in args.metrics.split(",") if s.strip()]

    groups = defaultdict(list)
    for path in files:
        label, block_size, metrics, _ = parse_file(path, target_unit)
        if block_size is None:
            continue
        groups[block_size].append((label, metrics))

    if args.block is not None:
        if args.block not in groups:
            print(f"没有 Block Size = {args.block} 的数据，可选：{sorted(groups)}")
            return
        out = args.out or f"ncu_block{args.block}.png"
        plot_block(args.block, groups[args.block], metric_names, duration_col, out)
    else:
        for block_size in sorted(groups):
            out = f"ncu_block{block_size}.png"
            plot_block(block_size, groups[block_size], metric_names, duration_col, out)


if __name__ == "__main__":
    main()
