#!/usr/bin/env python3
"""
用法（在本目录下执行）：
    python3 process_ncu_csv.py
    python3 process_ncu_csv.py --dir /path/to/result --out ncu_summary

功能：
    解析当前目录（或 --dir 指定目录）下所有 rmsnorm_forward_kernel*.txt 的
    ncu `--page details` 输出，按 Block Size（batch size）分组，把同一个 Block Size
    下的所有 kernel 汇总成一个 CSV。

    注意：ncu 里 Duration 的单位不统一（个别 kernel 是 ms，多数是 us）。
    脚本会先读每个文件的 Duration 单位，统计出「多数单位」，再把所有 Duration
    统一换算到该多数单位写进 CSV。列名会自动带上单位，例如 Duration (us)。

依赖：仅标准库（re/glob/csv/argparse/collections），无第三方依赖。
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

# 除 Duration 外的指标（Duration 单独处理，因为它的单位不统一）
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

# 时间单位 -> 秒 的换算系数
TIME_SCALE_S = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def normalize_unit(unit):
    """把 ncu 里可能的单位名（ns/us/ms/s 或全称）归一成短名。"""
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
    """在 ncu details 文本里找 Metric Name，返回它所在行最后一个 token（数值）。"""
    m = re.search(r"^\s*" + re.escape(name) + r"\b.*\s(\S+)\s*$", text, re.MULTILINE)
    if not m:
        return None
    value = m.group(1)
    try:
        return float(value)
    except ValueError:
        return value


def get_duration_raw(text):
    """返回 Duration 的 (数值, 单位)，例如 (1.33, 'ms')；找不到返回 None。"""
    m = re.search(r"^\s*Duration\s+(\S+)\s+([\d.]+)\s*$", text, re.MULTILINE)
    if not m:
        return None
    return float(m.group(2)), normalize_unit(m.group(1))


def detect_majority_unit(files):
    """统计所有文件 Duration 的单位，返回出现次数最多的单位。"""
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

    # Duration：按实际单位换算到多数单位 target_unit
    duration_col = f"Duration ({target_unit})"
    dur = get_duration_raw(text)
    if dur:
        seconds = dur[0] * TIME_SCALE_S[dur[1]]
        metrics[duration_col] = round(seconds / TIME_SCALE_S[target_unit], 4)
    else:
        metrics[duration_col] = None
    return label, block_size, metrics, duration_col


def main():
    ap = argparse.ArgumentParser(description="把同 Block Size 的 ncu 数据汇总成 CSV（Duration 统一到多数单位）")
    ap.add_argument("--dir", default=".", help="ncu .txt 所在目录，默认当前目录")
    ap.add_argument("--out", default="ncu_summary", help="输出 CSV 前缀，默认 ncu_summary")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "rmsnorm_forward_kernel*.txt")))
    if not files:
        print("没找到 rmsnorm_forward_kernel*.txt 文件，请用 --dir 指定目录")
        return

    target_unit = detect_majority_unit(files)
    print(f"Duration 多数单位是 {target_unit}，统一换算到 {target_unit}")

    groups = defaultdict(list)  # block_size -> [(label, metrics)]
    for path in files:
        label, block_size, metrics, _ = parse_file(path, target_unit)
        if block_size is None:
            continue
        groups[block_size].append((label, metrics))

    dtype_order = {"fp32": 0, "fp16": 1, "bf16": 2}
    duration_col = f"Duration ({target_unit})"
    for block_size in sorted(groups):
        rows = groups[block_size]
        rows.sort(key=lambda r: (r[1]["_kernel_num"] or 0, dtype_order.get(r[1]["_dtype"], 9)))
        out = f"{args.out}_block{block_size}.csv"
        fieldnames = ["kernel", duration_col] + [name for name, _ in METRICS] + ["Block Size"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for label, m in rows:
                row = {"kernel": label}
                row.update(m)
                writer.writerow(row)
        print(f"写入 {out}：{len(rows)} 个 kernel")


if __name__ == "__main__":
    main()
