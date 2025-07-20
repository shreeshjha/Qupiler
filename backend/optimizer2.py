#!/usr/bin/env python3
"""
Quantum Circuit Optimizer for Gate-Level MLIR
Usage: python optimizer.py <input.mlir> <output.mlir> [--debug] [--stats]
"""

import re
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class QuantumGate:
    gate_type: str
    qubits: List[str]
    control_qubits: List[str] = field(default_factory=list)
    target_qubits: List[str] = field(default_factory=list)
    is_eliminated: bool = False

@dataclass
class QubitLifetime:
    first_use: int = -1
    last_use: int = -1

@dataclass
class OptimizationStats:
    original_gates: int = 0
    optimized_gates: int = 0
    original_width: int = 0
    optimized_width: int = 0
    cnot_reduction: int = 0
    gate_cancellations: int = 0
    qubit_reuse_count: int = 0

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
        self.stats.original_width = len(self.qubit_lifetimes)
        
        self.debug_print(f"Original: {self.stats.original_gates} gates, {self.stats.original_width} qubits")

        # Iteratively apply optimizations until a fixed point is reached
        for pass_num in range(5): # Loop to catch cascading optimizations
            self.debug_print(f"--- Running Optimization Suite: Pass {pass_num + 1} ---")
            initial_gate_count = len([g for g in self.gates if not g.is_eliminated])
            
            self._pass_sanitize_and_simplify()
            self._pass_gate_cancellation()
            
            final_gate_count = len([g for g in self.gates if not g.is_eliminated])
            if initial_gate_count == final_gate_count:
                self.debug_print("No further gate reductions found. Moving to final passes.")
                break
        
        self._pass_qubit_coalescing()
        
        optimized_mlir = self._generate_optimized_mlir()
        
        self.stats.optimized_gates = len([g for g in self.gates if not g.is_eliminated])
        self.debug_print(f"Optimized: {self.stats.optimized_gates} gates, {self.stats.optimized_width} qubits")
        
        return optimized_mlir

    def _discover_qubits(self, qubits: List[str]):
        """Dynamically discovers and registers qubit registers from gate operands."""
        for qubit in qubits:
            base_reg = qubit.split('[')[0]
            if base_reg not in self.qubit_lifetimes:
                self.debug_print(f"Discovered new qubit register: {base_reg}")
                self.qubit_lifetimes[base_reg] = QubitLifetime()
                if base_reg not in self.allocations:
                    self.allocations.append(base_reg)

    def _parse_mlir(self, content: str):
        lines = content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('//') or "q.comment" in line_stripped:
                continue
            
            alloc_match = re.search(r'(%q\d+)\s*=\s*q\.alloc', line_stripped)
            if alloc_match:
                qubit_reg = alloc_match.group(1)
                if qubit_reg not in self.allocations: self.allocations.append(qubit_reg)
                if qubit_reg not in self.qubit_lifetimes: self.qubit_lifetimes[qubit_reg] = QubitLifetime()
                continue
            
            init_match = re.search(r'q\.init\s+(%q\d+),\s*(\d+)', line_stripped)
            if init_match:
                qubit_reg, value = init_match.groups()
                self._discover_qubits([qubit_reg])
                self.initializations.append((qubit_reg, int(value)))
                continue
            
            measure_match = re.search(r'(%q\d+)\s*=\s*q\.measure\s+(%q\d+)', line_stripped)
            if measure_match:
                result_reg, measured_reg = measure_match.groups()
                self._discover_qubits([measured_reg])
                self.measurements.append((result_reg, measured_reg))
                continue
            
            gate = self._parse_gate(line_stripped)
            if gate:
                self.gates.append(gate)

    def _parse_gate(self, line: str) -> Optional[QuantumGate]:
        gate_match = re.search(r'q\.(\w+)\s+(.+)', line)
        if not gate_match:
            return None

        gate_type = gate_match.group(1)
        operands_str = gate_match.group(2).split('//')[0].strip() # Ignore comments
        operands = [op.strip() for op in operands_str.split(',')]
        
        self._discover_qubits(operands)
        
        control_qubits, target_qubits = [], []
        if gate_type in ["cx", "ccx"]:
            control_qubits = operands[:-1]
            target_qubits = [operands[-1]]
        else:
            target_qubits = operands
            
        return QuantumGate(gate_type, operands, control_qubits, target_qubits)

    def _pass_sanitize_and_simplify(self):
        """Eliminates invalid gates and simplifies complex gates."""
        self.debug_print("Pass: Sanitize and Simplify...")
        for gate in self.gates:
            if gate.is_eliminated:
                continue

            # Rule: Eliminate gates where control and target are the same
            if set(gate.control_qubits) & set(gate.target_qubits):
                self.debug_print(f"Sanitizing: Removing gate with overlapping control/target: {gate.qubits}")
                gate.is_eliminated = True
                self.stats.gate_cancellations += 1
                continue

            # Rule: Simplify CCX(a, a, b) -> CX(a, b)
            if gate.gate_type == "ccx" and len(set(gate.control_qubits)) < len(gate.control_qubits):
                unique_controls = sorted(list(set(gate.control_qubits)))
                self.debug_print(f"Simplifying CCX on {gate.qubits} to CX")
                gate.gate_type = "cx"
                gate.qubits = [unique_controls[0], gate.target_qubits[0]]
                gate.control_qubits = [unique_controls[0]]
                self.stats.cnot_reduction += 1
                
    def _pass_gate_cancellation(self):
        """Cancels consecutive self-inverse gates."""
        self.debug_print("Pass: Gate Cancellation...")
        i = 0
        while i < len(self.gates) - 1:
            gate1 = self.gates[i]
            if gate1.is_eliminated:
                i += 1
                continue

            next_gate_idx = next((j for j, g in enumerate(self.gates[i+1:]) if not g.is_eliminated), -1)
            if next_gate_idx == -1: break
            next_gate_idx += i + 1
            
            gate2 = self.gates[next_gate_idx]
            
            if (gate1.gate_type == gate2.gate_type and gate1.qubits == gate2.qubits 
                and gate1.gate_type in ["x", "h", "cx", "ccx"]):
                
                # Check for interfering gates between the pair
                interfering = False
                for k in range(i + 1, next_gate_idx):
                    interim_gate = self.gates[k]
                    if not interim_gate.is_eliminated and set(gate1.qubits) & set(interim_gate.qubits):
                        interfering = True
                        break
                
                if not interfering:
                    self.debug_print(f"Cancelling {gate1.gate_type} pair on {gate1.qubits}")
                    gate1.is_eliminated = True
                    gate2.is_eliminated = True
                    self.stats.gate_cancellations += 1
                    i = next_gate_idx + 1
                    continue
            i += 1
            
    def _pass_qubit_coalescing(self):
        """Reuses qubit registers whose lifetimes do not overlap."""
        self.debug_print("Pass: Qubit Coalescing...")
        
        # 1. Analyze lifetimes based on the current active gates
        for q in self.qubit_lifetimes: self.qubit_lifetimes[q] = QubitLifetime()
        active_gates = [g for g in self.gates if not g.is_eliminated]
        for i, gate in enumerate(active_gates):
            for qubit in gate.qubits:
                base_reg = qubit.split('[')[0]
                if self.qubit_lifetimes[base_reg].first_use == -1:
                    self.qubit_lifetimes[base_reg].first_use = i
                self.qubit_lifetimes[base_reg].last_use = i

        # 2. Find coalescing opportunities
        coalescing_map = {}
        sorted_qubits = sorted(self.qubit_lifetimes.items(), key=lambda item: item[1].first_use)
        
        for i in range(len(sorted_qubits)):
            q1, life1 = sorted_qubits[i]
            if life1.last_use == -1: continue

            for j in range(i + 1, len(sorted_qubits)):
                q2, life2 = sorted_qubits[j]
                
                # Find the root of the coalescing chain for q1
                target_q = q1
                while target_q in coalescing_map:
                    target_q = coalescing_map[target_q]

                if life1.last_use < life2.first_use and q2 not in coalescing_map:
                    self.debug_print(f"Coalescing {q2} -> {target_q} (lifetimes end:{life1.last_use} < start:{life2.first_use})")
                    coalescing_map[q2] = target_q
                    self.stats.qubit_reuse_count += 1
        
        if not coalescing_map: return

        # 3. Apply the coalescing map to all gates
        for gate in self.gates:
            if gate.is_eliminated: continue
            
            new_qubits = []
            for q_str in gate.qubits:
                base, *rest = q_str.split('[')
                new_base = coalescing_map.get(base, base)
                new_qubits.append(f"{new_base}{'[' + rest[0] if rest else ''}")
            
            gate.qubits = new_qubits
            if gate.gate_type in ["cx", "ccx"]:
                gate.control_qubits = new_qubits[:-1]
                gate.target_qubits = [new_qubits[-1]]
            else:
                gate.target_qubits = new_qubits

    def _reconstruct_gate_line(self, gate: QuantumGate) -> str:
        qubit_str = ", ".join(gate.qubits)
        return f"    q.{gate.gate_type} {qubit_str}"

    def _generate_optimized_mlir(self) -> str:
        active_gates = [g for g in self.gates if not g.is_eliminated]
        
        final_used_qubits = set()
        for gate in active_gates:
            for q in gate.qubits: final_used_qubits.add(q.split('[')[0])
        for _, measured_reg in self.measurements: final_used_qubits.add(measured_reg.split('[')[0])
        for reg, _ in self.initializations: final_used_qubits.add(reg)

        self.stats.optimized_width = len(final_used_qubits)
        final_allocations = sorted(list(final_used_qubits))

        header = [
            "// Optimized Gate-Level Quantum MLIR",
            f"// Original: {self.stats.original_gates} gates, {self.stats.original_width} qubits",
            f"// Optimized: {len(active_gates)} gates, {self.stats.optimized_width} qubits",
            f"// Reductions: {self.stats.gate_cancellations} cancellations, {self.stats.cnot_reduction} CNOT-equiv, {self.stats.qubit_reuse_count} qubit reuses",
            "builtin.module {",
            '  "quantum.func"() ({'
        ]
        
        body = []
        if final_allocations:
            for qubit_reg in final_allocations:
                body.append(f"    {qubit_reg} = q.alloc : !qreg<4>")
            body.append("")
        
        for qubit_reg, value in sorted(self.initializations):
            if qubit_reg in final_used_qubits:
                body.append(f"    q.init {qubit_reg}, {value} : i32")
        if any(init[0] in final_used_qubits for init in self.initializations): body.append("")

        for gate in active_gates: body.append(self._reconstruct_gate_line(gate))
        if active_gates: body.append("")
        
        for result_reg, measured_reg in self.measurements:
            if measured_reg.split('[')[0] in final_used_qubits:
                body.append(f"    {result_reg} = q.measure {measured_reg} : !qreg -> i32")
        
        footer = ["    func.return", '  }) {func_name = "quantum_circuit"} : () -> ()', "}"]
        return '\n'.join(header + body + footer)

    def print_optimization_stats(self):
        print("\n" + "="*60)
        print("QUANTUM CIRCUIT OPTIMIZATION RESULTS")
        print("="*60)
        gate_reduction = self.stats.original_gates - self.stats.optimized_gates
        qubit_reduction = self.stats.original_width - self.stats.optimized_width
        
        print(f"Original Gates:        {self.stats.original_gates}")
        print(f"Optimized Gates:       {self.stats.optimized_gates}")
        print(f"Gates Reduced:         {gate_reduction}")
        if self.stats.original_gates > 0:
            print(f"Reduction Percentage:  {(gate_reduction / self.stats.original_gates * 100):.1f}%")
        print("-" * 25)
        print(f"Original Qubits:       {self.stats.original_width}")
        print(f"Optimized Qubits:      {self.stats.optimized_width}")
        print(f"Qubits Reduced:        {qubit_reduction}")
        print("-" * 25)
        print("Optimization Breakdown:")
        print(f"  Gate Cancellations:  {self.stats.gate_cancellations}")
        print(f"  CNOT-equiv Reduced:  {self.stats.cnot_reduction}")
        print(f"  Qubit Reuse:         {self.stats.qubit_reuse_count}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Quantum Circuit Optimizer for MLIR')
    parser.add_argument('input_file', help='Input MLIR file')
    parser.add_argument('output_file', help='Output optimized MLIR file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--stats', action='store_true', help='Show optimization statistics')
    
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    optimizer = QuantumCircuitOptimizer(enable_debug=args.debug)
    optimized_mlir = optimizer.optimize_circuit(mlir_content)
    
    try:
        with open(args.output_file, 'w') as f:
            f.write(optimized_mlir)
        print(f"Optimized MLIR written to: {args.output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    
    if args.stats:
        optimizer.print_optimization_stats()

if __name__ == "__main__":
    main()
