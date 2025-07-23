#!/usr/bin/env python3
"""
Batch MLIR Gate-Level Benchmark: Toffoli Reduction Metrics

This script scans a directory for original and optimized gate-level MLIR files
and computes per-file and summary metrics:
- Circuit depth, width, gate count
- Toffoli (ccx) gate count and depth

Usage:
    python compare_mlir_metrics.py [--dir <mlir_directory>]
"""

import sys
import os
import re
from pathlib import Path
from collections import defaultdict

def parse_mlir(filename: Path):
    """Parse MLIR file and extract quantum operations and involved qubits."""
    ops = []
    # match quantum operations like q.x, q.cx, q.ccx etc.
    op_pattern = re.compile(r"\s*q\.([a-zA-Z0-9_]+)\s+([%\w\[\], ]+)")
    qubit_pattern = re.compile(r"%q(\d+)")
    with open(filename, 'r') as f:
        for line in f:
            m = op_pattern.search(line)
            if m:
                opname = m.group(1).lower()
                args = m.group(2)
                # extract all qubit indices
                qubits = [int(q) for q in qubit_pattern.findall(args)]
                ops.append({'op': opname, 'qubits': qubits})
    return ops


def compute_metrics(ops):
    """Compute circuit and Toffoli metrics from parsed ops."""
    total_ops = len(ops)
    toffoli_ops = sum(1 for o in ops if o['op'] in ('ccx', 'toffoli'))

    # circuit width = highest qubit index +1
    max_q = max((q for o in ops for q in o['qubits']), default=-1)
    width = max_q + 1

    # compute general circuit depth
    qubit_depth = defaultdict(int)
    depth = 0
    for o in ops:
        involved = o['qubits']
        layer = (max(qubit_depth[q] for q in involved) + 1) if involved else depth + 1
        for q in involved:
            qubit_depth[q] = layer
        depth = max(depth, layer)

    # compute Toffoli-specific depth
    tof_depth_map = defaultdict(int)
    tof_depth = 0
    for o in ops:
        if o['op'] in ('ccx', 'toffoli'):
            involved = o['qubits']
            layer = (max(tof_depth_map[q] for q in involved) + 1) if involved else tof_depth + 1
            for q in involved:
                tof_depth_map[q] = layer
            tof_depth = max(tof_depth, layer)

    return {
        'Circuit Depth': depth,
        'Circuit Width': width,
        'Gate Count': total_ops,
        'Toffoli Count': toffoli_ops,
        'Toffoli Depth': tof_depth
    }


def find_pairs(directory: Path):
    """Locate pairs of original and optimized MLIR files in a directory."""
    original_files = sorted(directory.glob('*_gate_opt.mlir'))
    pairs = []
    for orig in original_files:
        if orig.name.startswith('optimized_'):
            continue
        opt_name = 'optimized_' + orig.name
        opt_path = directory / opt_name
        if opt_path.exists():
            pairs.append((orig, opt_path))
        else:
            print(f"⚠️  Missing optimized file for {orig.name}")
    return pairs


def print_metrics(orig: Path, opt: Path, m1: dict, m2: dict):
    header = f"Metrics for {orig.name} vs {opt.name}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"Metric{'':15} | Original | Optimized | Saving %")
    print('-' * 60)
    for key in ['Circuit Depth', 'Circuit Width', 'Gate Count', 'Toffoli Count', 'Toffoli Depth']:
        v1, v2 = m1.get(key, 0), m2.get(key, 0)
        saving = ((v1 - v2) / v1 * 100) if v1 > 0 else 0.0
        print(f"{key:18} | {v1:8} | {v2:8} | {saving:8.2f}%")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare gate-level MLIR metrics')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing MLIR files')
    args = parser.parse_args()

    work_dir = Path(args.dir)
    if not work_dir.is_dir():
        print(f"Error: {work_dir} is not a directory.")
        sys.exit(1)

    pairs = find_pairs(work_dir)
    if not pairs:
        print("No valid file pairs found in directory.")
        sys.exit(0)

    summary = defaultdict(lambda: {'orig': 0, 'opt': 0})
    count = 0

    for orig, opt in pairs:
        ops1 = parse_mlir(orig)
        ops2 = parse_mlir(opt)
        m1 = compute_metrics(ops1)
        m2 = compute_metrics(ops2)
        print_metrics(orig, opt, m1, m2)

        for k in m1:
            summary[k]['orig'] += m1[k]
            summary[k]['opt'] += m2[k]
        count += 1

    if count > 1:
        print("Overall Averages Across Files:")
        print('-' * 40)
        for key, vals in summary.items():
            avg_orig = vals['orig'] / count
            avg_opt = vals['opt'] / count
            save_pct = ((avg_orig - avg_opt) / avg_orig * 100) if avg_orig > 0 else 0.0
            print(f"{key:18} | {avg_orig:8.2f} | {avg_opt:8.2f} | {save_pct:8.2f}%")

if __name__ == '__main__':
    main()

