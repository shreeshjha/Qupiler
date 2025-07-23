#!/usr/bin/env python3
"""
Simple MLIR Metrics Analyzer

Quick analysis tool for existing MLIR files to extract circuit metrics.
Can be used independently of the full benchmark system.

Usage: python simple_metrics_analyzer.py <mlir_file>
       python simple_metrics_analyzer.py --dir <directory>
"""

import re
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

@dataclass
class SimpleCircuitMetrics:
    """Simplified circuit metrics"""
    file_name: str
    circuit_depth: int = 0
    circuit_width: int = 0  # Number of qubits
    gate_count: int = 0
    
    # Gate type breakdown
    cx_gates: int = 0
    ccx_gates: int = 0  # Toffoli/T-gates
    x_gates: int = 0
    init_ops: int = 0
    measure_ops: int = 0
    
    # High-level operations (if present)
    add_circuits: int = 0
    sub_circuits: int = 0
    mul_circuits: int = 0
    div_circuits: int = 0
    mod_circuits: int = 0
    and_circuits: int = 0
    or_circuits: int = 0
    xor_circuits: int = 0
    
    # File properties
    file_size_bytes: int = 0
    total_lines: int = 0

class SimpleMLIRAnalyzer:
    def __init__(self):
        self.gate_patterns = {
            'cx': r'q\.cx\s+',
            'ccx': r'q\.ccx\s+',
            'x': r'q\.x\s+',
            'init': r'q\.init\s+',
            'measure': r'q\.measure\s+',
            'alloc': r'q\.alloc\s*:'
        }
        
        self.circuit_patterns = {
            'add_circuit': r'q\.add_circuit\s+',
            'sub_circuit': r'q\.sub_circuit\s+', 
            'mul_circuit': r'q\.mul_circuit\s+',
            'div_circuit': r'q\.div_circuit\s+',
            'mod_circuit': r'q\.mod_circuit\s+',
            'and_circuit': r'q\.and_circuit\s+',
            'or_circuit': r'q\.or_circuit\s+',
            'xor_circuit': r'q\.xor_circuit\s+'
        }
    
    def analyze_mlir_file(self, file_path: Path) -> SimpleCircuitMetrics:
        """Analyze a single MLIR file"""
        
        metrics = SimpleCircuitMetrics(file_name=str(file_path))
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return metrics
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            metrics.file_size_bytes = len(content.encode('utf-8'))
            metrics.total_lines = len(lines)
            
            # Extract qubits (circuit width)
            qubit_pattern = r'%q(\d+)'
            qubits = set()
            for match in re.finditer(qubit_pattern, content):
                qubits.add(int(match.group(1)))
            metrics.circuit_width = len(qubits)
            
            # Count gate operations
            gate_counts = {}
            for gate_type, pattern in self.gate_patterns.items():
                count = len(re.findall(pattern, content))
                gate_counts[gate_type] = count
            
            metrics.cx_gates = gate_counts.get('cx', 0)
            metrics.ccx_gates = gate_counts.get('ccx', 0)
            metrics.x_gates = gate_counts.get('x', 0)
            metrics.init_ops = gate_counts.get('init', 0)
            metrics.measure_ops = gate_counts.get('measure', 0)
            
            # Total gate count (excluding alloc, init, measure)
            metrics.gate_count = metrics.cx_gates + metrics.ccx_gates + metrics.x_gates
            
            # Count high-level circuit operations
            circuit_counts = {}
            for circuit_type, pattern in self.circuit_patterns.items():
                count = len(re.findall(pattern, content))
                circuit_counts[circuit_type] = count
            
            metrics.add_circuits = circuit_counts.get('add_circuit', 0)
            metrics.sub_circuits = circuit_counts.get('sub_circuit', 0)
            metrics.mul_circuits = circuit_counts.get('mul_circuit', 0)
            metrics.div_circuits = circuit_counts.get('div_circuit', 0)
            metrics.mod_circuits = circuit_counts.get('mod_circuit', 0)
            metrics.and_circuits = circuit_counts.get('and_circuit', 0)
            metrics.or_circuits = circuit_counts.get('or_circuit', 0)
            metrics.xor_circuits = circuit_counts.get('xor_circuit', 0)
            
            # Estimate circuit depth (simplified approach)
            # Count maximum operations per qubit
            qubit_operations = defaultdict(int)
            
            for line in lines:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('//'):
                    continue
                
                # Look for quantum operations
                if any(op in line for op in ['q.cx', 'q.ccx', 'q.x']):
                    # Extract qubits involved in this operation
                    qubits_in_op = re.findall(r'%q(\d+)', line)
                    for qubit in qubits_in_op:
                        qubit_operations[int(qubit)] += 1
            
            # Circuit depth is the maximum operations on any qubit
            metrics.circuit_depth = max(qubit_operations.values()) if qubit_operations else 0
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return metrics
    
    def analyze_directory(self, dir_path: Path) -> List[SimpleCircuitMetrics]:
        """Analyze all MLIR files in directory"""
        
        mlir_files = list(dir_path.glob('*.mlir'))
        
        if not mlir_files:
            print(f"❌ No MLIR files found in {dir_path}")
            return []
        
        print(f"📁 Found {len(mlir_files)} MLIR files in {dir_path}")
        
        results = []
        for mlir_file in mlir_files:
            print(f"🔍 Analyzing: {mlir_file.name}")
            metrics = self.analyze_mlir_file(mlir_file)
            results.append(metrics)
        
        return results
    
    def print_metrics(self, metrics: SimpleCircuitMetrics):
        """Print metrics in a readable format"""
        
        print(f"\n📊 Metrics for: {metrics.file_name}")
        print("-" * 50)
        print(f"Circuit Properties:")
        print(f"  • Circuit Width (Qubits): {metrics.circuit_width}")
        print(f"  • Circuit Depth: {metrics.circuit_depth}")
        print(f"  • Total Gates: {metrics.gate_count}")
        
        print(f"\nGate Breakdown:")
        print(f"  • CX Gates: {metrics.cx_gates}")
        print(f"  • CCX Gates (T-gates): {metrics.ccx_gates}")
        print(f"  • X Gates: {metrics.x_gates}")
        print(f"  • Init Operations: {metrics.init_ops}")
        print(f"  • Measure Operations: {metrics.measure_ops}")
        
        # Show high-level operations if present
        high_level_total = (metrics.add_circuits + metrics.sub_circuits + 
                           metrics.mul_circuits + metrics.div_circuits +
                           metrics.mod_circuits + metrics.and_circuits +
                           metrics.or_circuits + metrics.xor_circuits)
        
        if high_level_total > 0:
            print(f"\nHigh-Level Operations:")
            if metrics.add_circuits > 0: print(f"  • Add Circuits: {metrics.add_circuits}")
            if metrics.sub_circuits > 0: print(f"  • Sub Circuits: {metrics.sub_circuits}")
            if metrics.mul_circuits > 0: print(f"  • Mul Circuits: {metrics.mul_circuits}")
            if metrics.div_circuits > 0: print(f"  • Div Circuits: {metrics.div_circuits}")
            if metrics.mod_circuits > 0: print(f"  • Mod Circuits: {metrics.mod_circuits}")
            if metrics.and_circuits > 0: print(f"  • And Circuits: {metrics.and_circuits}")
            if metrics.or_circuits > 0: print(f"  • Or Circuits: {metrics.or_circuits}")
            if metrics.xor_circuits > 0: print(f"  • Xor Circuits: {metrics.xor_circuits}")
        
        print(f"\nFile Properties:")
        print(f"  • File Size: {metrics.file_size_bytes} bytes")
        print(f"  • Total Lines: {metrics.total_lines}")
    
    def compare_metrics(self, metrics_list: List[SimpleCircuitMetrics]):
        """Compare multiple metrics and show summary"""
        
        if len(metrics_list) < 2:
            print("❌ Need at least 2 files to compare")
            return
        
        print(f"\n📊 COMPARISON SUMMARY ({len(metrics_list)} files)")
        print("=" * 60)
        
        # Create comparison table
        print(f"{'File':<25} {'Width':<6} {'Depth':<6} {'Gates':<6} {'CX':<4} {'CCX':<4}")
        print("-" * 60)
        
        total_gates = 0
        total_ccx = 0
        
        for metrics in metrics_list:
            file_name = Path(metrics.file_name).name[:24]  # Truncate long names
            total_gates += metrics.gate_count
            total_ccx += metrics.ccx_gates
            
            print(f"{file_name:<25} {metrics.circuit_width:<6} {metrics.circuit_depth:<6} "
                  f"{metrics.gate_count:<6} {metrics.cx_gates:<4} {metrics.ccx_gates:<4}")
        
        print("-" * 60)
        print(f"{'TOTALS':<25} {'':<6} {'':<6} {total_gates:<6} {'':<4} {total_ccx:<4}")
        
        # Calculate averages
        avg_width = sum(m.circuit_width for m in metrics_list) / len(metrics_list)
        avg_depth = sum(m.circuit_depth for m in metrics_list) / len(metrics_list)
        avg_gates = sum(m.gate_count for m in metrics_list) / len(metrics_list)
        
        print(f"\n📈 Averages:")
        print(f"  • Average Width: {avg_width:.1f} qubits")
        print(f"  • Average Depth: {avg_depth:.1f}")
        print(f"  • Average Gates: {avg_gates:.1f}")
    
    def save_results(self, metrics_list: List[SimpleCircuitMetrics], output_file: str):
        """Save results to JSON file"""
        
        output_path = Path(output_file)
        
        # Convert to JSON-serializable format
        json_data = {
            'summary': {
                'total_files': len(metrics_list),
                'total_gates': sum(m.gate_count for m in metrics_list),
                'total_ccx_gates': sum(m.ccx_gates for m in metrics_list),
                'average_width': sum(m.circuit_width for m in metrics_list) / len(metrics_list) if metrics_list else 0,
                'average_depth': sum(m.circuit_depth for m in metrics_list) / len(metrics_list) if metrics_list else 0
            },
            'files': [asdict(m) for m in metrics_list]
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Simple MLIR Metrics Analyzer')
    parser.add_argument('file_or_dir', nargs='?', help='MLIR file or directory to analyze')
    parser.add_argument('--dir', '-d', type=str, help='Directory containing MLIR files')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file for results')
    parser.add_argument('--compare', action='store_true', help='Show comparison table')
    
    args = parser.parse_args()
    
    analyzer = SimpleMLIRAnalyzer()
    
    # Determine what to analyze
    if args.dir:
        target_path = Path(args.dir)
        if not target_path.is_dir():
            print(f"❌ Directory not found: {target_path}")
            return 1
        
        metrics_list = analyzer.analyze_directory(target_path)
        
    elif args.file_or_dir:
        target_path = Path(args.file_or_dir)
        
        if target_path.is_file():
            # Single file
            metrics = analyzer.analyze_mlir_file(target_path)
            analyzer.print_metrics(metrics)
            metrics_list = [metrics]
            
        elif target_path.is_dir():
            # Directory
            metrics_list = analyzer.analyze_directory(target_path)
            
        else:
            print(f"❌ File or directory not found: {target_path}")
            return 1
    else:
        # Default: analyze current directory
        print("🔍 No input specified, analyzing current directory...")
        metrics_list = analyzer.analyze_directory(Path('.'))
    
    # Show results
    if len(metrics_list) > 1:
        if args.compare:
            analyzer.compare_metrics(metrics_list)
        else:
            # Print individual metrics
            for metrics in metrics_list:
                analyzer.print_metrics(metrics)
    
    # Save results if requested
    if args.output:
        analyzer.save_results(metrics_list, args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
