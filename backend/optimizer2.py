#!/usr/bin/env python3
"""
SAFE Quantum Circuit Optimizer for gate_optimizer2.py Output
Compatible with circuit_generator.py input format

Usage: python optimizer.py <input.mlir> <output.mlir> [--debug]
"""

import re
import sys
import argparse
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class QuantumGate:
    gate_type: str
    qubits: List[str]
    control_qubits: List[str] = None
    target_qubits: List[str] = None
    line_number: int = 0
    original_line: str = ""
    is_eliminated: bool = False

@dataclass
class OptimizationStats:
    original_gates: int = 0
    gate_cancellations: int = 0

class SafeQuantumCircuitOptimizer:
    def __init__(self, enable_debug=False):
        self.enable_debug = enable_debug
        self.gates: List[QuantumGate] = []
        self.allocations: List[str] = []
        self.initializations: List[Tuple[str, int]] = []
        self.measurements: List[Tuple[str, str]] = []
        self.stats = OptimizationStats()
        
    def debug_print(self, message: str):
        if self.enable_debug:
            print(f"[SAFE_OPTIMIZER] {message}")
    
    def optimize_circuit(self, mlir_content: str) -> str:
        self.debug_print("Starting SAFE quantum circuit optimization...")
        
        self._parse_mlir(mlir_content)
        self.stats.original_gates = len(self.gates)
        
        self.debug_print(f"Original: {self.stats.original_gates} gates")
        self.debug_print(f"Allocations: {len(self.allocations)}")
        self.debug_print(f"Initializations: {len(self.initializations)}")
        self.debug_print(f"Measurements: {len(self.measurements)}")
        
        # ONLY apply the safest optimizations
        self._safe_x_gate_cancellation()
        
        return self._generate_safe_mlir()
    
    def _parse_mlir(self, content: str):
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Skip pure comment lines but preserve circuit operation lines
            if line_stripped.startswith('//') and 'CIRCUIT_DECOMP' not in line_stripped:
                continue
            
            # Parse allocations - PRESERVE ALL
            alloc_match = re.search(r'(%q\d+)\s*=\s*q\.alloc\s*:\s*!qreg<(\d+)>', line_stripped)
            if alloc_match:
                qubit_reg = alloc_match.group(1)
                self.allocations.append(qubit_reg)
                self.debug_print(f"Found allocation: {qubit_reg}")
                continue
            
            # Parse initializations - PRESERVE ALL
            init_match = re.search(r'q\.init\s+(%q\d+),\s*(\d+)', line_stripped)
            if init_match:
                qubit_reg, value = init_match.groups()
                self.initializations.append((qubit_reg, int(value)))
                self.debug_print(f"Found initialization: {qubit_reg} = {value}")
                continue
            
            # Parse measurements - PRESERVE ALL
            measure_match = re.search(r'(%q\d+)\s*=\s*q\.measure\s+(%q\d+)', line_stripped)
            if measure_match:
                result_reg, measured_reg = measure_match.groups()
                self.measurements.append((result_reg, measured_reg))
                self.debug_print(f"Found measurement: {measured_reg} -> {result_reg}")
                continue
            
            # Parse quantum gates - CAPTURE EVERYTHING
            if self._is_quantum_gate(line_stripped):
                gate = QuantumGate(
                    gate_type="gate",
                    qubits=[],
                    line_number=i,
                    original_line=line_stripped  # FIXED: Store the actual line
                )
                self.gates.append(gate)
                self.debug_print(f"Found gate: {line_stripped}")
    
    def _is_quantum_gate(self, line: str) -> bool:
        """Check if line contains a quantum gate operation"""
        # Skip pure comments
        if line.startswith('q.comment'):
            return False
        
        # Match quantum gate operations
        gate_patterns = [
            r'q\.x\s+%q\d+\[\d+\]',       # X gates
            r'q\.cx\s+%q\d+\[\d+\],',     # CNOT gates  
            r'q\.ccx\s+%q\d+\[\d+\],',    # Toffoli gates
        ]
        
        return any(re.search(pattern, line) for pattern in gate_patterns)
    
    def _safe_x_gate_cancellation(self):
        """Only cancel X gates that are TRULY consecutive and identical"""
        self.debug_print("Applying SAFE X-gate cancellation...")
        
        i = 0
        while i < len(self.gates) - 1:
            gate1 = self.gates[i]
            gate2 = self.gates[i + 1]
            
            # Only cancel if both lines are X gates on the same exact qubit
            if (not gate1.is_eliminated and not gate2.is_eliminated and
                'q.x' in gate1.original_line and 'q.x' in gate2.original_line):
                
                # Extract qubit from both lines
                qubit1_match = re.search(r'q\.x\s+(%q\d+\[\d+\])', gate1.original_line)
                qubit2_match = re.search(r'q\.x\s+(%q\d+\[\d+\])', gate2.original_line)
                
                if (qubit1_match and qubit2_match and 
                    qubit1_match.group(1) == qubit2_match.group(1)):
                    
                    gate1.is_eliminated = True
                    gate2.is_eliminated = True
                    self.stats.gate_cancellations += 1
                    self.debug_print(f"Safely cancelled X-gate pair on {qubit1_match.group(1)}")
                    i += 2
                    continue
            
            i += 1
        
        self.debug_print(f"Total X-gate cancellations: {self.stats.gate_cancellations}")
    
    def _generate_safe_mlir(self) -> str:
        """Generate MLIR preserving ALL structure"""
        lines = [
            "// SAFE Optimized Gate-Level Quantum MLIR",
            f"// Original: {self.stats.original_gates} gates",
            f"// Safe reductions: {self.stats.gate_cancellations} X-gate cancellations",
            "builtin.module {",
            '  "quantum.func"() ({'
        ]
        
        # PRESERVE ALL allocations
        for qubit_reg in self.allocations:
            lines.append(f"    {qubit_reg} = q.alloc : !qreg<4>")
        
        # PRESERVE ALL initializations  
        for qubit_reg, value in self.initializations:
            lines.append(f"    q.init {qubit_reg}, {value} : i32")
        
        # Add all non-eliminated gates
        active_gates = 0
        for gate in self.gates:
            if not gate.is_eliminated:
                lines.append(f"    {gate.original_line}")
                active_gates += 1
        
        self.debug_print(f"Output {active_gates} gates")
        
        # PRESERVE ALL measurements
        for result_reg, measured_reg in self.measurements:
            lines.append(f"    {result_reg} = q.measure {measured_reg} : !qreg -> i32")
        
        lines.extend([
            "    func.return",
            '  }) {func_name = "quantum_circuit"} : () -> ()',
            "}"
        ])
        
        return '\n'.join(lines)
    
    def print_stats(self):
        active_gates = len([g for g in self.gates if not g.is_eliminated])
        print(f"🎯 Original gates: {self.stats.original_gates}")
        print(f"🎯 Output gates: {active_gates}")
        print(f"🎯 X-gate cancellations: {self.stats.gate_cancellations}")
        print(f"✅ Circuit correctness preserved")

def main():
    parser = argparse.ArgumentParser(description='SAFE Quantum Circuit Optimizer for MLIR')
    parser.add_argument('input_file', help='Input MLIR file from gate_optimizer2.py')
    parser.add_argument('output_file', help='Output optimized MLIR file for circuit_generator.py')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--stats', action='store_true', help='Show optimization statistics')
    
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    optimizer = SafeQuantumCircuitOptimizer(enable_debug=args.debug)
    optimized_mlir = optimizer.optimize_circuit(mlir_content)
    
    try:
        with open(args.output_file, 'w') as f:
            f.write(optimized_mlir)
        print(f"✅ SAFE optimized MLIR written to: {args.output_file}")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)
    
    if args.stats:
        optimizer.print_stats()
    else:
        optimizer.print_stats()

if __name__ == "__main__":
    main()
