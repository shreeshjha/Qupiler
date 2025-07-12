#!/usr/bin/env python3
"""
Quantum Circuit Optimizer for gate_optimizer2.py Output
Compatible with circuit_generator.py input format

Usage: python quantum_optimizer_clean.py <input.mlir> <output.mlir> [--debug]
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
class QubitLifetime:
    first_use: int = -1
    last_use: int = -1
    is_ancilla: bool = False
    can_reuse: bool = False

@dataclass
class OptimizationStats:
    original_gates: int = 0
    optimized_gates: int = 0
    original_width: int = 0
    optimized_width: int = 0
    cnot_reduction: int = 0
    gate_cancellations: int = 0
    qubit_reuse_count: int = 0
    dead_gates_removed: int = 0

class QuantumCircuitOptimizer:
    def __init__(self, enable_debug=False):
        self.enable_debug = enable_debug
        self.gates: List[QuantumGate] = []
        self.allocations: List[str] = []
        self.initializations: List[Tuple[str, int]] = []
        self.measurements: List[Tuple[str, str]] = []
        self.qubit_lifetimes: Dict[str, QubitLifetime] = {}
        self.stats = OptimizationStats()
        
    def debug_print(self, message: str):
        if self.enable_debug:
            print(f"[OPTIMIZER] {message}")
    
    def optimize_circuit(self, mlir_content: str) -> str:
        self.debug_print("Starting quantum circuit optimization...")
        
        self._parse_mlir(mlir_content)
        self.stats.original_gates = len(self.gates)
        self.stats.original_width = len(self.allocations)
        
        self.debug_print(f"Original: {self.stats.original_gates} gates, {self.stats.original_width} qubits")
        
        self._analyze_qubit_lifetimes()
        self._optimization_pass_1_gate_cancellation()
        self._optimization_pass_2_cnot_optimization()
        self._optimization_pass_3_qubit_coalescing()
        self._optimization_pass_5_peephole_optimization()
        
        optimized_mlir = self._generate_optimized_mlir()
        
        active_gates = [g for g in self.gates if not g.is_eliminated]
        self.stats.optimized_gates = len(active_gates)
        
        self.debug_print(f"Optimized: {self.stats.optimized_gates} gates")
        
        return optimized_mlir
    
    def _parse_mlir(self, content: str):
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('//'):
                continue
            
            alloc_match = re.search(r'(%q\d+)\s*=\s*q\.alloc\s*:\s*!qreg<(\d+)>', line_stripped)
            if alloc_match:
                qubit_reg = alloc_match.group(1)
                self.allocations.append(qubit_reg)
                self.qubit_lifetimes[qubit_reg] = QubitLifetime()
                continue
            
            init_match = re.search(r'q\.init\s+(%q\d+),\s*(\d+)', line_stripped)
            if init_match:
                qubit_reg, value = init_match.groups()
                self.initializations.append((qubit_reg, int(value)))
                continue
            
            measure_match = re.search(r'(%q\d+)\s*=\s*q\.measure\s+(%q\d+)', line_stripped)
            if measure_match:
                result_reg, measured_reg = measure_match.groups()
                self.measurements.append((result_reg, measured_reg))
                continue
            
            gate = self._parse_gate(line_stripped, i)
            if gate:
                self.gates.append(gate)
    
    def _parse_gate(self, line: str, line_num: int) -> Optional[QuantumGate]:
        if line.strip().startswith('//') or 'q.comment' in line:
            return None
        
        cnot_match = re.search(r'q\.cx\s+(%q\d+)\[(\d+)\],\s*(%q\d+)\[(\d+)\]', line)
        if cnot_match:
            control = f"{cnot_match.group(1)}[{cnot_match.group(2)}]"
            target = f"{cnot_match.group(3)}[{cnot_match.group(4)}]"
            return QuantumGate("cx", [control, target], [control], [target], line_num, line)
        
        ccx_match = re.search(r'q\.ccx\s+(%q\d+)\[(\d+)\],\s*(%q\d+)\[(\d+)\],\s*(%q\d+)\[(\d+)\]', line)
        if ccx_match:
            ctrl1 = f"{ccx_match.group(1)}[{ccx_match.group(2)}]"
            ctrl2 = f"{ccx_match.group(3)}[{ccx_match.group(4)}]"
            target = f"{ccx_match.group(5)}[{ccx_match.group(6)}]"
            return QuantumGate("ccx", [ctrl1, ctrl2, target], [ctrl1, ctrl2], [target], line_num, line)
        
        x_match = re.search(r'q\.x\s+(%q\d+)\[(\d+)\]', line)
        if x_match:
            target = f"{x_match.group(1)}[{x_match.group(2)}]"
            return QuantumGate("x", [target], [], [target], line_num, line)
        
        circuit_match = re.search(r'q\.(\w+_circuit)\s+((?:%q\d+(?:,\s*)?)+)', line)
        if circuit_match:
            circuit_type, operands_str = circuit_match.groups()
            operands = [op.strip() for op in operands_str.split(',')]
            return QuantumGate(circuit_type, operands, [], [], line_num, line)
        
        return None
    
    def _analyze_qubit_lifetimes(self):
        self.debug_print("Analyzing qubit lifetimes...")
        
        for i, gate in enumerate(self.gates):
            for qubit in gate.qubits:
                base_reg = qubit.split('[')[0] if '[' in qubit else qubit
                
                if base_reg in self.qubit_lifetimes:
                    lifetime = self.qubit_lifetimes[base_reg]
                    if lifetime.first_use == -1:
                        lifetime.first_use = i
                    lifetime.last_use = i
    
    def _optimization_pass_1_gate_cancellation(self):
        self.debug_print("Pass 1: Gate cancellation...")
        
        self._cancel_consecutive_x_gates()
        self._cancel_consecutive_identical_gates()
    
    def _cancel_consecutive_x_gates(self):
        i = 0
        while i < len(self.gates) - 1:
            gate1 = self.gates[i]
            gate2 = self.gates[i + 1]
            
            if (gate1.gate_type == "x" and gate2.gate_type == "x" and 
                not gate1.is_eliminated and not gate2.is_eliminated and
                gate1.qubits == gate2.qubits):
                
                gate1.is_eliminated = True
                gate2.is_eliminated = True
                self.stats.gate_cancellations += 1
                self.debug_print(f"Cancelled consecutive X-gate pair on {gate1.qubits[0]}")
                i += 2
            else:
                i += 1
    
    def _cancel_consecutive_identical_gates(self):
        i = 0
        while i < len(self.gates) - 1:
            gate1 = self.gates[i]
            gate2 = self.gates[i + 1]
            
            if (gate1.gate_type == gate2.gate_type and 
                gate1.qubits == gate2.qubits and
                not gate1.is_eliminated and not gate2.is_eliminated and
                gate1.gate_type in ["cx", "ccx"]):
                
                gate1.is_eliminated = True
                gate2.is_eliminated = True
                self.stats.gate_cancellations += 1
                self.debug_print(f"Cancelled consecutive {gate1.gate_type} pair: {gate1.qubits}")
                i += 2
            else:
                i += 1
    
    def _optimization_pass_2_cnot_optimization(self):
        self.debug_print("Pass 2: CNOT optimization...")
        
        cnot_count_before = sum(1 for g in self.gates if g.gate_type == "cx" and not g.is_eliminated)
        
        self._optimize_ccx_to_cx_patterns()
        
        cnot_count_after = sum(1 for g in self.gates if g.gate_type == "cx" and not g.is_eliminated)
        self.stats.cnot_reduction = cnot_count_before - cnot_count_after
        
        ccx_reductions = sum(1 for g in self.gates if g.gate_type == "ccx" and g.is_eliminated)
        self.stats.cnot_reduction += ccx_reductions
        
        self.debug_print(f"CNOT-equivalent reduction: {self.stats.cnot_reduction}")
    
    def _optimize_ccx_to_cx_patterns(self):
        for gate in self.gates:
            if gate.is_eliminated or gate.gate_type != "ccx":
                continue
            
            if len(gate.qubits) >= 3:
                ctrl1, ctrl2, target = gate.qubits[:3]
                
                if ctrl1 == ctrl2:
                    gate.is_eliminated = True
                    new_gate = QuantumGate("cx", [ctrl1, target], [ctrl1], [target], 
                                         gate.line_number, f"    q.cx {ctrl1}, {target}  // Optimized from CCX")
                    idx = self.gates.index(gate)
                    self.gates.insert(idx + 1, new_gate)
                    self.debug_print(f"Converted CCX to CX: {ctrl1} == {ctrl2}")
    
    def _optimization_pass_3_qubit_coalescing(self):
        self.debug_print("Pass 3: Qubit coalescing...")
        
        coalescing_map = {}
        available_qubits = set(self.qubit_lifetimes.keys())
        
        for qubit1 in list(available_qubits):
            if qubit1 in coalescing_map:
                continue
                
            lifetime1 = self.qubit_lifetimes[qubit1]
            
            for qubit2 in list(available_qubits):
                if qubit1 == qubit2 or qubit2 in coalescing_map:
                    continue
                
                lifetime2 = self.qubit_lifetimes[qubit2]
                
                if (lifetime1.last_use < lifetime2.first_use or 
                    lifetime2.last_use < lifetime1.first_use):
                    coalescing_map[qubit2] = qubit1
                    self.stats.qubit_reuse_count += 1
                    self.debug_print(f"Coalescing {qubit2} into {qubit1}")
                    break
        
        for gate in self.gates:
            if gate.is_eliminated:
                continue
            
            new_qubits = []
            for qubit in gate.qubits:
                base_reg = qubit.split('[')[0] if '[' in qubit else qubit
                if base_reg in coalescing_map:
                    if '[' in qubit:
                        index = qubit.split('[')[1]
                        new_qubit = f"{coalescing_map[base_reg]}[{index}"
                    else:
                        new_qubit = coalescing_map[base_reg]
                    new_qubits.append(new_qubit)
                else:
                    new_qubits.append(qubit)
            gate.qubits = new_qubits
        
        self.allocations = [q for q in self.allocations if q not in coalescing_map]
        self.stats.optimized_width = len(self.allocations)
    
    def _optimization_pass_5_peephole_optimization(self):
        self.debug_print("Pass 5: Peephole optimization...")
        
        self._optimize_multiplication_patterns()
    
    def _optimize_multiplication_patterns(self):
        for i, gate1 in enumerate(self.gates):
            if gate1.is_eliminated or gate1.gate_type != "ccx":
                continue
            
            for j in range(i + 2, min(i + 10, len(self.gates))):
                gate2 = self.gates[j]
                if gate2.is_eliminated or gate2.gate_type != "ccx":
                    continue
                
                if gate1.qubits == gate2.qubits:
                    interfering = False
                    for k in range(i + 1, j):
                        intermediate = self.gates[k]
                        if (not intermediate.is_eliminated and 
                            any(qubit in intermediate.qubits for qubit in gate1.qubits)):
                            interfering = True
                            break
                    
                    if not interfering:
                        gate1.is_eliminated = True
                        gate2.is_eliminated = True
                        self.stats.gate_cancellations += 1
                        self.debug_print(f"Cancelled distant identical CCX gates: {gate1.qubits}")
                        break
    
    def _generate_optimized_mlir(self) -> str:
        lines = [
            "// Optimized Gate-Level Quantum MLIR",
            f"// Original: {self.stats.original_gates} gates, {self.stats.original_width} qubits",
            f"// Optimized: {len([g for g in self.gates if not g.is_eliminated])} gates, {len(self.allocations)} qubits",
            f"// Reductions: {self.stats.gate_cancellations} cancellations, {self.stats.cnot_reduction} CNOT-equiv",
            "builtin.module {",
            '  "quantum.func"() ({'
        ]
        
        used_qubits = set()
        for gate in self.gates:
            if not gate.is_eliminated:
                for qubit in gate.qubits:
                    base_reg = qubit.split('[')[0] if '[' in qubit else qubit
                    used_qubits.add(base_reg)
        
        for _, measured_qubit in self.measurements:
            base_reg = measured_qubit.split('[')[0] if '[' in measured_qubit else measured_qubit
            used_qubits.add(base_reg)
        
        final_allocations = [q for q in self.allocations if q in used_qubits]
        
        for qubit_reg in final_allocations:
            lines.append(f"    {qubit_reg} = q.alloc : !qreg<4>")
        
        for qubit_reg, value in self.initializations:
            if qubit_reg in final_allocations:
                lines.append(f"    q.init {qubit_reg}, {value} : i32")
        
        for gate in self.gates:
            if not gate.is_eliminated:
                lines.append(f"    {gate.original_line.strip()}")
        
        for result_reg, measured_reg in self.measurements:
            measured_base = measured_reg.split('[')[0] if '[' in measured_reg else measured_reg
            if measured_base in final_allocations:
                lines.append(f"    {result_reg} = q.measure {measured_reg} : !qreg -> i32")
        
        lines.extend([
            "    func.return",
            '  }) {func_name = "quantum_circuit"} : () -> ()',
            "}"
        ])
        
        self.stats.optimized_width = len(final_allocations)
        return '\n'.join(lines)
    
    def print_optimization_stats(self):
        print("\n" + "="*60)
        print("QUANTUM CIRCUIT OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Original Gates:        {self.stats.original_gates}")
        optimized_gates = len([g for g in self.gates if not g.is_eliminated])
        print(f"Optimized Gates:       {optimized_gates}")
        print(f"Gates Reduced:         {self.stats.original_gates - optimized_gates}")
        if self.stats.original_gates > 0:
            print(f"Reduction Percentage:  {((self.stats.original_gates - optimized_gates) / self.stats.original_gates * 100):.1f}%")
        print()
        print(f"Original Qubits:       {self.stats.original_width}")
        print(f"Optimized Qubits:      {self.stats.optimized_width}")
        print(f"Qubits Reduced:        {self.stats.original_width - self.stats.optimized_width}")
        print()
        print("Optimization Breakdown:")
        print(f"  Gate Cancellations:  {self.stats.gate_cancellations}")
        print(f"  CNOT-equiv Reduced:  {self.stats.cnot_reduction}")
        print(f"  Qubit Reuse:         {self.stats.qubit_reuse_count}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Quantum Circuit Optimizer for MLIR')
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
    
    optimizer = QuantumCircuitOptimizer(enable_debug=args.debug)
    optimized_mlir = optimizer.optimize_circuit(mlir_content)
    
    try:
        with open(args.output_file, 'w') as f:
            f.write(optimized_mlir)
        print(f"✅ Optimized MLIR written to: {args.output_file}")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)
    
    if args.stats:
        optimizer.print_optimization_stats()
    
    optimized_gates = len([g for g in optimizer.gates if not g.is_eliminated])
    print(f"🎯 Gate reduction: {optimizer.stats.original_gates - optimized_gates}")
    print(f"🎯 Qubit reduction: {optimizer.stats.original_width - optimizer.stats.optimized_width}")

if __name__ == "__main__":
    main()