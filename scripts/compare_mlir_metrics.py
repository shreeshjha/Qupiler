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
    """Compute comprehensive quantum circuit metrics from parsed ops."""
    total_ops = len(ops)
    
    # Count different gate types
    toffoli_ops = sum(1 for o in ops if o['op'] in ('ccx', 'toffoli'))
    t_ops = sum(1 for o in ops if o['op'] in ('t', 'tdg'))
    h_ops = sum(1 for o in ops if o['op'] == 'h')
    s_ops = sum(1 for o in ops if o['op'] in ('s', 'sdg'))
    z_ops = sum(1 for o in ops if o['op'] == 'z')
    x_ops = sum(1 for o in ops if o['op'] == 'x')
    y_ops = sum(1 for o in ops if o['op'] == 'y')
    cx_ops = sum(1 for o in ops if o['op'] == 'cx')

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

    # compute T-gate specific depth
    t_depth_map = defaultdict(int)
    t_depth = 0
    for o in ops:
        if o['op'] in ('t', 'tdg'):
            involved = o['qubits']
            layer = (max(t_depth_map[q] for q in involved) + 1) if involved else t_depth + 1
            for q in involved:
                t_depth_map[q] = layer
            t_depth = max(t_depth, layer)

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
        'T-Gate Count': t_ops,
        'T-Gate Depth': t_depth,
        'Toffoli Count': toffoli_ops,
        'Toffoli Depth': tof_depth,
        'H-Gate Count': h_ops,
        'S-Gate Count': s_ops,
        'Z-Gate Count': z_ops,
        'X-Gate Count': x_ops,
        'Y-Gate Count': y_ops,
        'CX-Gate Count': cx_ops
    }


def find_pairs(directory: Path):
    """Locate pairs of original and enhanced optimized MLIR files in a directory."""
    original_files = sorted(directory.glob('*_gate_opt.mlir'))
    pairs = []
    for orig in original_files:
        # Get base name without _gate_opt.mlir suffix
        base_name = orig.name.replace('_gate_opt.mlir', '')
        opt_name = base_name + '_enhanced_opt.mlir'
        opt_path = directory / opt_name
        if opt_path.exists():
            pairs.append((orig, opt_path))
        else:
            print(f"⚠️  Missing enhanced optimized file for {orig.name}: expected {opt_name}")
    return pairs


def print_metrics(orig: Path, opt: Path, m1: dict, m2: dict):
    header = f"Quantum Circuit Optimization Metrics: {orig.name} → {opt.name}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"{'Metric':<20} | {'Original':<10} | {'Optimized':<10} | {'Reduction':<12} | {'% Saved':<8}")
    print('-' * 75)
    
    # Core circuit metrics
    core_metrics = ['Circuit Depth', 'Circuit Width', 'Gate Count']
    for key in core_metrics:
        v1, v2 = m1.get(key, 0), m2.get(key, 0)
        reduction = v1 - v2
        saving = ((v1 - v2) / v1 * 100) if v1 > 0 else 0.0
        print(f"{key:<20} | {v1:<10} | {v2:<10} | {reduction:<12} | {saving:<8.1f}%")
    
    print()
    print("Gate-Specific Metrics:")
    print('-' * 75)
    
    # Gate-specific metrics
    gate_metrics = ['T-Gate Count', 'T-Gate Depth', 'Toffoli Count', 'Toffoli Depth', 
                   'H-Gate Count', 'S-Gate Count', 'Z-Gate Count', 'X-Gate Count', 
                   'Y-Gate Count', 'CX-Gate Count']
    for key in gate_metrics:
        v1, v2 = m1.get(key, 0), m2.get(key, 0)
        reduction = v1 - v2
        saving = ((v1 - v2) / v1 * 100) if v1 > 0 else 0.0
        print(f"{key:<20} | {v1:<10} | {v2:<10} | {reduction:<12} | {saving:<8.1f}%")
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
        print("\n" + "=" * 75)
        print(f"SUMMARY: Average Optimization Results Across {count} Test Cases")
        print("=" * 75)
        print(f"{'Metric':<20} | {'Original':<10} | {'Optimized':<10} | {'Avg Reduction':<14} | {'% Saved':<8}")
        print('-' * 75)
        
        # Core metrics first
        core_metrics = ['Circuit Depth', 'Circuit Width', 'Gate Count']
        for key in core_metrics:
            if key in summary:
                vals = summary[key]
                avg_orig = vals['orig'] / count
                avg_opt = vals['opt'] / count
                avg_reduction = avg_orig - avg_opt
                save_pct = ((avg_orig - avg_opt) / avg_orig * 100) if avg_orig > 0 else 0.0
                print(f"{key:<20} | {avg_orig:<10.1f} | {avg_opt:<10.1f} | {avg_reduction:<14.1f} | {save_pct:<8.1f}%")
        
        print()
        print("Gate-Specific Average Optimizations:")
        print('-' * 75)
        
        # Gate-specific metrics
        gate_metrics = ['T-Gate Count', 'T-Gate Depth', 'Toffoli Count', 'Toffoli Depth', 
                       'H-Gate Count', 'S-Gate Count', 'Z-Gate Count', 'X-Gate Count', 
                       'Y-Gate Count', 'CX-Gate Count']
        for key in gate_metrics:
            if key in summary:
                vals = summary[key]
                avg_orig = vals['orig'] / count
                avg_opt = vals['opt'] / count
                avg_reduction = avg_orig - avg_opt
                save_pct = ((avg_orig - avg_opt) / avg_orig * 100) if avg_orig > 0 else 0.0
                print(f"{key:<20} | {avg_orig:<10.1f} | {avg_opt:<10.1f} | {avg_reduction:<14.1f} | {save_pct:<8.1f}%")

if __name__ == '__main__':
    main()

