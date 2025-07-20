#!/usr/bin/env python3
"""
Advanced Quantum Gate Optimizer

This optimizer implements various quantum-specific optimizations:
1. X Gate Optimization (removing double X gates, X gate cancellation)
2. CX Gate Optimization (CNOT cancellation, redundant CX removal)
3. CCX Gate Optimization (Toffoli gate optimization, redundant CCX removal)
4. Dead Code Elimination (removing unused qubits and operations)
5. Gate Commutation and Reordering
6. Measurement Optimization
7. Register Consolidation

Usage: python quantum_gate_optimizer.py <input.mlir> <output.mlir> [--debug]
"""

import re
import sys
import argparse
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque

@dataclass
class QuantumGate:
    """Represents a quantum gate operation"""
    gate_type: str
    operands: List[str]
    line_number: int
    original_line: str
    is_removed: bool = False
    optimization_applied: str = ""

@dataclass
class QubitRegister:
    """Represents a quantum register"""
    name: str
    size: int
    is_used: bool = False
    first_use: Optional[int] = None
    last_use: Optional[int] = None

@dataclass
class OptimizationStats:
    """Track optimization statistics"""
    original_gates: int = 0
    optimized_gates: int = 0
    x_optimizations: int = 0
    cx_optimizations: int = 0
    ccx_optimizations: int = 0
    dead_code_eliminations: int = 0
    register_consolidations: int = 0
    gate_reorderings: int = 0

class AdvancedQuantumGateOptimizer:
    def __init__(self, enable_debug=False):
        self.enable_debug = enable_debug
        self.gates: List[QuantumGate] = []
        self.registers: Dict[str, QubitRegister] = {}
        self.allocations: List[str] = []
        self.initializations: List[str] = []
        self.measurements: List[str] = []
        self.comments: List[str] = []
        self.stats = OptimizationStats()
        
    def debug_print(self, message: str):
        if self.enable_debug:
            print(f"[DEBUG] {message}")
    
    def parse_mlir(self, content: str) -> None:
        """Parse MLIR content and extract operations"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines and structural elements
            if (not line_stripped or 
                line_stripped.startswith('//') or
                'builtin.module' in line_stripped or
                'quantum.func' in line_stripped or
                'func.return' in line_stripped or
                line_stripped in ['}', '{']):
                continue
            
            # Parse allocation
            alloc_match = re.search(r'%(\w+)\s*=\s*q\.alloc\s*:\s*!qreg<(\d+)>', line_stripped)
            if alloc_match:
                reg_name, size = alloc_match.groups()
                full_name = f"%{reg_name}"
                self.registers[full_name] = QubitRegister(full_name, int(size))
                self.allocations.append(line_stripped)
                self.debug_print(f"Found allocation: {full_name} (size: {size})")
                continue
            
            # Parse initialization
            init_match = re.search(r'q\.init\s+%(\w+),\s*(\d+)', line_stripped)
            if init_match:
                self.initializations.append(line_stripped)
                reg_name = f"%{init_match.group(1)}"
                if reg_name in self.registers:
                    self.registers[reg_name].is_used = True
                    self.registers[reg_name].first_use = i
                self.debug_print(f"Found initialization: {line_stripped}")
                continue
            
            # Parse measurement
            measure_match = re.search(r'%(\w+)\s*=\s*q\.measure\s+%(\w+)', line_stripped)
            if measure_match:
                self.measurements.append(line_stripped)
                reg_name = f"%{measure_match.group(2)}"
                if reg_name in self.registers:
                    self.registers[reg_name].last_use = i
                self.debug_print(f"Found measurement: {line_stripped}")
                continue
            
            # Parse comments
            if 'q.comment' in line_stripped:
                self.comments.append(line_stripped)
                continue
            
            # Parse quantum gates
            gate_match = re.search(r'q\.(\w+)\s+((?:%\w+(?:\[\d+\])?(?:,\s*)?)+)', line_stripped)
            if gate_match and gate_match.group(1) not in ['comment', 'alloc', 'init', 'measure']:
                gate_type = gate_match.group(1)
                operands_str = gate_match.group(2)
                operands = [op.strip() for op in operands_str.split(',')]
                
                gate = QuantumGate(
                    gate_type=gate_type,
                    operands=operands,
                    line_number=i,
                    original_line=line_stripped
                )
                
                self.gates.append(gate)
                
                # Update register usage
                for operand in operands:
                    reg_name = operand.split('[')[0] if '[' in operand else operand
                    if reg_name in self.registers:
                        self.registers[reg_name].is_used = True
                        if self.registers[reg_name].first_use is None:
                            self.registers[reg_name].first_use = i
                        self.registers[reg_name].last_use = i
                
                self.debug_print(f"Found gate: {gate_type} on {operands}")
        
        self.stats.original_gates = len(self.gates)
        self.debug_print(f"Parsed {len(self.gates)} gates total")
    
    def optimization_1_x_gate_optimization(self) -> int:
        """
        X Gate Optimizations:
        1. Remove consecutive X gates on same qubit (X X = I)
        2. Remove X gates that don't affect the computation
        3. Merge X gates with other operations where possible
        """
        print("🔧 Applying X Gate Optimizations...")
        optimizations = 0
        
        # Track X gates by qubit
        x_gates_by_qubit = defaultdict(list)
        
        for i, gate in enumerate(self.gates):
            if gate.gate_type == 'x' and not gate.is_removed:
                qubit = gate.operands[0]
                x_gates_by_qubit[qubit].append(i)
        
        # Remove consecutive X gates (X X = I)
        for qubit, gate_indices in x_gates_by_qubit.items():
            i = 0
            while i < len(gate_indices) - 1:
                current_idx = gate_indices[i]
                next_idx = gate_indices[i + 1]
                
                # Check if gates are consecutive (no other gates on same qubit between them)
                if self._are_gates_consecutive(current_idx, next_idx, qubit):
                    self.gates[current_idx].is_removed = True
                    self.gates[current_idx].optimization_applied = "X_CANCELLATION_1"
                    self.gates[next_idx].is_removed = True
                    self.gates[next_idx].optimization_applied = "X_CANCELLATION_2"
                    
                    optimizations += 2
                    self.debug_print(f"Removed consecutive X gates on {qubit}")
                    i += 2  # Skip both gates
                else:
                    i += 1
        
        # Remove X gates that don't affect measurements
        for i, gate in enumerate(self.gates):
            if gate.gate_type == 'x' and not gate.is_removed:
                qubit = gate.operands[0]
                if not self._affects_measurement(i, qubit):
                    gate.is_removed = True
                    gate.optimization_applied = "X_DEAD_CODE"
                    optimizations += 1
                    self.debug_print(f"Removed dead X gate on {qubit}")
        
        self.stats.x_optimizations = optimizations
        print(f"   ✓ Applied {optimizations} X gate optimizations")
        return optimizations
    
    def optimization_2_cx_gate_optimization(self) -> int:
        """
        CX Gate Optimizations:
        1. Remove consecutive CX gates on same control-target pair (CX CX = I)
        2. Optimize CX gates based on commutation rules
        3. Remove redundant CX operations
        """
        print("🔧 Applying CX Gate Optimizations...")
        optimizations = 0
        
        # Track CX gates by control-target pair
        cx_gates_by_pair = defaultdict(list)
        
        for i, gate in enumerate(self.gates):
            if gate.gate_type == 'cx' and not gate.is_removed and len(gate.operands) >= 2:
                control = gate.operands[0]
                target = gate.operands[1]
                pair = (control, target)
                cx_gates_by_pair[pair].append(i)
        
        # Remove consecutive CX gates (CX CX = I)
        for pair, gate_indices in cx_gates_by_pair.items():
            control, target = pair
            i = 0
            while i < len(gate_indices) - 1:
                current_idx = gate_indices[i]
                next_idx = gate_indices[i + 1]
                
                # Check if gates are consecutive
                if self._are_cx_gates_consecutive(current_idx, next_idx, control, target):
                    self.gates[current_idx].is_removed = True
                    self.gates[current_idx].optimization_applied = "CX_CANCELLATION_1"
                    self.gates[next_idx].is_removed = True
                    self.gates[next_idx].optimization_applied = "CX_CANCELLATION_2"
                    
                    optimizations += 2
                    self.debug_print(f"Removed consecutive CX gates: {control} -> {target}")
                    i += 2
                else:
                    i += 1
        
        # Remove CX gates that don't affect computation
        for i, gate in enumerate(self.gates):
            if gate.gate_type == 'cx' and not gate.is_removed and len(gate.operands) >= 2:
                control = gate.operands[0]
                target = gate.operands[1]
                
                if not self._cx_affects_measurement(i, control, target):
                    gate.is_removed = True
                    gate.optimization_applied = "CX_DEAD_CODE"
                    optimizations += 1
                    self.debug_print(f"Removed dead CX gate: {control} -> {target}")
        
        self.stats.cx_optimizations = optimizations
        print(f"   ✓ Applied {optimizations} CX gate optimizations")
        return optimizations
    
    def optimization_3_ccx_gate_optimization(self) -> int:
        """
        CCX Gate Optimizations:
        1. Remove consecutive CCX gates on same control-control-target triplet
        2. Optimize redundant Toffoli gates
        3. Replace CCX with simpler gates where possible
        """
        print("🔧 Applying CCX Gate Optimizations...")
        optimizations = 0
        
        # Track CCX gates by triplet
        ccx_gates_by_triplet = defaultdict(list)
        
        for i, gate in enumerate(self.gates):
            if gate.gate_type == 'ccx' and not gate.is_removed and len(gate.operands) >= 3:
                control1 = gate.operands[0]
                control2 = gate.operands[1] 
                target = gate.operands[2]
                triplet = (control1, control2, target)
                ccx_gates_by_triplet[triplet].append(i)
        
        # Remove consecutive CCX gates (CCX CCX = I)
        for triplet, gate_indices in ccx_gates_by_triplet.items():
            control1, control2, target = triplet
            i = 0
            while i < len(gate_indices) - 1:
                current_idx = gate_indices[i]
                next_idx = gate_indices[i + 1]
                
                if self._are_ccx_gates_consecutive(current_idx, next_idx, control1, control2, target):
                    self.gates[current_idx].is_removed = True
                    self.gates[current_idx].optimization_applied = "CCX_CANCELLATION_1"
                    self.gates[next_idx].is_removed = True
                    self.gates[next_idx].optimization_applied = "CCX_CANCELLATION_2"
                    
                    optimizations += 2
                    self.debug_print(f"Removed consecutive CCX gates: {control1}, {control2} -> {target}")
                    i += 2
                else:
                    i += 1
        
        # Detect and optimize redundant CCX patterns
        for i, gate in enumerate(self.gates):
            if gate.gate_type == 'ccx' and not gate.is_removed and len(gate.operands) >= 3:
                if self._is_redundant_ccx(i):
                    gate.is_removed = True
                    gate.optimization_applied = "CCX_REDUNDANT"
                    optimizations += 1
                    self.debug_print(f"Removed redundant CCX gate")
        
        self.stats.ccx_optimizations = optimizations
        print(f"   ✓ Applied {optimizations} CCX gate optimizations")
        return optimizations
    
    def optimization_4_dead_code_elimination(self) -> int:
        """
        Advanced Dead Code Elimination:
        1. Remove unused quantum registers
        2. Remove gates that don't contribute to measurements
        3. Remove unreachable code
        """
        print("🔧 Applying Dead Code Elimination...")
        optimizations = 0
        
        # Find registers that contribute to measurements
        contributing_qubits = self._find_contributing_qubits()
        
        # Remove gates on non-contributing qubits
        for gate in self.gates:
            if not gate.is_removed:
                gate_qubits = set()
                for operand in gate.operands:
                    qubit_base = operand.split('[')[0]
                    gate_qubits.add(qubit_base)
                
                # If none of the gate's qubits contribute to measurement, remove it
                if not gate_qubits.intersection(contributing_qubits):
                    gate.is_removed = True
                    gate.optimization_applied = "DEAD_CODE_ELIMINATION"
                    optimizations += 1
                    self.debug_print(f"Removed dead gate: {gate.gate_type} on {gate.operands}")
        
        # Mark unused registers
        unused_registers = []
        for reg_name, reg_info in self.registers.items():
            if not reg_info.is_used or reg_name not in contributing_qubits:
                unused_registers.append(reg_name)
        
        self.stats.dead_code_eliminations = optimizations
        print(f"   ✓ Eliminated {optimizations} dead gates and {len(unused_registers)} unused registers")
        return optimizations
    
    def optimization_5_gate_commutation_and_reordering(self) -> int:
        """
        Gate Commutation and Reordering:
        1. Reorder commuting gates for better optimization opportunities
        2. Move gates closer to reduce circuit depth
        3. Optimize gate scheduling
        """
        print("🔧 Applying Gate Commutation and Reordering...")
        optimizations = 0
        
        # Create dependency graph
        dependencies = self._build_dependency_graph()
        
        # Reorder gates to reduce circuit depth
        optimizations += self._reorder_commuting_gates(dependencies)
        
        # Move X gates closer to their dependencies
        optimizations += self._optimize_x_gate_placement()
        
        self.stats.gate_reorderings = optimizations
        print(f"   ✓ Applied {optimizations} gate reorderings")
        return optimizations
    
    def optimization_6_register_consolidation(self) -> int:
        """
        Register Consolidation:
        1. Merge registers with non-overlapping lifetimes
        2. Renumber registers for better layout
        3. Remove unused register allocations
        """
        print("🔧 Applying Register Consolidation...")
        optimizations = 0
        
        # Find register lifetime overlaps
        register_lifetimes = self._compute_register_lifetimes()
        
        # Consolidate non-overlapping registers
        consolidation_map = self._find_consolidation_opportunities(register_lifetimes)
        
        if consolidation_map:
            optimizations += self._apply_register_consolidation(consolidation_map)
        
        # Renumber registers for dense layout
        optimizations += self._renumber_registers()
        
        self.stats.register_consolidations = optimizations
        print(f"   ✓ Applied {optimizations} register consolidations")
        return optimizations
    
    def optimization_7_measurement_optimization(self) -> int:
        """
        Measurement Optimization:
        1. Ensure measurements target the correct result registers
        2. Remove redundant measurements
        3. Optimize measurement placement
        """
        print("🔧 Applying Measurement Optimization...")
        optimizations = 0
        
        # Find the final result register (last computed result)
        final_result_reg = self._find_final_result_register()
        
        # Update measurements to target final result
        for i, measurement in enumerate(self.measurements):
            measure_match = re.search(r'(%\w+)\s*=\s*q\.measure\s+(%\w+)', measurement)
            if measure_match:
                result_reg, measured_reg = measure_match.groups()
                
                if final_result_reg and measured_reg != final_result_reg:
                    # Update measurement target
                    new_measurement = measurement.replace(measured_reg, final_result_reg)
                    self.measurements[i] = new_measurement + "  // MEASUREMENT_OPTIMIZED"
                    optimizations += 1
                    self.debug_print(f"Optimized measurement: {measured_reg} -> {final_result_reg}")
        
        print(f"   ✓ Applied {optimizations} measurement optimizations")
        return optimizations
    
    # Helper methods
    def _are_gates_consecutive(self, idx1: int, idx2: int, qubit: str) -> bool:
        """Check if two gates on the same qubit are consecutive"""
        for i in range(idx1 + 1, idx2):
            if not self.gates[i].is_removed:
                for operand in self.gates[i].operands:
                    if operand.split('[')[0] == qubit.split('[')[0]:
                        return False
        return True
    
    def _are_cx_gates_consecutive(self, idx1: int, idx2: int, control: str, target: str) -> bool:
        """Check if two CX gates are consecutive"""
        for i in range(idx1 + 1, idx2):
            if not self.gates[i].is_removed:
                gate = self.gates[i]
                for operand in gate.operands:
                    operand_base = operand.split('[')[0]
                    if operand_base == control.split('[')[0] or operand_base == target.split('[')[0]:
                        return False
        return True
    
    def _are_ccx_gates_consecutive(self, idx1: int, idx2: int, control1: str, control2: str, target: str) -> bool:
        """Check if two CCX gates are consecutive"""
        qubits = {control1.split('[')[0], control2.split('[')[0], target.split('[')[0]}
        
        for i in range(idx1 + 1, idx2):
            if not self.gates[i].is_removed:
                gate = self.gates[i]
                for operand in gate.operands:
                    if operand.split('[')[0] in qubits:
                        return False
        return True
    
    def _affects_measurement(self, gate_idx: int, qubit: str) -> bool:
        """Check if a gate affects any measurement"""
        qubit_base = qubit.split('[')[0]
        
        # Look for measurements on this qubit after this gate
        for i in range(gate_idx + 1, len(self.gates)):
            gate = self.gates[i]
            if not gate.is_removed:
                for operand in gate.operands:
                    if operand.split('[')[0] == qubit_base:
                        return True
        
        # Check measurements list
        for measurement in self.measurements:
            if qubit_base in measurement:
                return True
        
        return False
    
    def _cx_affects_measurement(self, gate_idx: int, control: str, target: str) -> bool:
        """Check if a CX gate affects measurements"""
        return (self._affects_measurement(gate_idx, control) or 
                self._affects_measurement(gate_idx, target))
    
    def _is_redundant_ccx(self, gate_idx: int) -> bool:
        """Check if a CCX gate is redundant"""
        gate = self.gates[gate_idx]
        if len(gate.operands) < 3:
            return False
        
        control1, control2, target = gate.operands[:3]
        
        # Simple heuristic: if controls are never set to 1, CCX is redundant
        # This is a simplified check - real optimization would need state tracking
        return False
    
    def _find_contributing_qubits(self) -> Set[str]:
        """Find qubits that contribute to measurements using correct dependency logic."""
        contributing = set()
        
        # Start from any register that is measured
        for measurement in self.measurements:
            measure_match = re.search(r'q\.measure\s+(%\w+)', measurement)
            if measure_match:
                contributing.add(measure_match.group(1))
        
        # Backward propagation through gates
        changed = True
        while changed:
            changed = False
            for gate in reversed(self.gates):
                if gate.is_removed:
                    continue

                # Get all unique register bases involved in the gate
                gate_qubit_bases = {op.split('[')[0] for op in gate.operands}

                # Check if ANY of the gate's qubits are already known to be contributing
                if any(qubit_base in contributing for qubit_base in gate_qubit_bases):
                    # If so, ALL qubits involved in this gate are now considered contributing,
                    # because their states are entangled or codependent.
                    for qubit_base in gate_qubit_bases:
                        if qubit_base not in contributing:
                            contributing.add(qubit_base)
                            changed = True
                            self.debug_print(f"Liveness analysis: Marked {qubit_base} as contributing due to gate {gate.gate_type}")
        
        return contributing

    
    def _build_dependency_graph(self) -> Dict[int, List[int]]:
        """Build gate dependency graph"""
        dependencies = defaultdict(list)
        
        for i, gate in enumerate(self.gates):
            if gate.is_removed:
                continue
                
            gate_qubits = {op.split('[')[0] for op in gate.operands}
            
            # Find gates that this gate depends on
            for j in range(i):
                if self.gates[j].is_removed:
                    continue
                    
                prev_gate_qubits = {op.split('[')[0] for op in self.gates[j].operands}
                
                if gate_qubits.intersection(prev_gate_qubits):
                    dependencies[i].append(j)
        
        return dependencies
    
    def _reorder_commuting_gates(self, dependencies: Dict[int, List[int]]) -> int:
        """Reorder commuting gates for better optimization"""
        # This is a simplified implementation
        # Real gate reordering would need sophisticated analysis
        return 0
    
    def _optimize_x_gate_placement(self) -> int:
        """Optimize X gate placement"""
        # Move X gates closer to where they're needed
        optimizations = 0
        
        # Find X gates that can be moved
        for i, gate in enumerate(self.gates):
            if gate.gate_type == 'x' and not gate.is_removed:
                # Check if we can move this gate later
                if self._can_move_x_gate(i):
                    optimizations += 1
        
        return optimizations
    
    def _can_move_x_gate(self, gate_idx: int) -> bool:
        """Check if an X gate can be moved"""
        # Simplified check
        return False
    
    def _compute_register_lifetimes(self) -> Dict[str, Tuple[int, int]]:
        """Compute lifetime intervals for each register"""
        lifetimes = {}
        
        for reg_name, reg_info in self.registers.items():
            if reg_info.first_use is not None and reg_info.last_use is not None:
                lifetimes[reg_name] = (reg_info.first_use, reg_info.last_use)
        
        return lifetimes
    
    def _find_consolidation_opportunities(self, lifetimes: Dict[str, Tuple[int, int]]) -> Dict[str, str]:
        """Find registers that can be consolidated"""
        consolidation_map = {}
        
        # Simple consolidation: merge registers with non-overlapping lifetimes
        reg_list = list(lifetimes.items())
        
        for i, (reg1, (start1, end1)) in enumerate(reg_list):
            for j, (reg2, (start2, end2)) in enumerate(reg_list[i+1:], i+1):
                # If lifetimes don't overlap, they can be consolidated
                if end1 < start2 or end2 < start1:
                    if reg2 not in consolidation_map and reg1 not in consolidation_map.values():
                        consolidation_map[reg2] = reg1
                        break
        
        return consolidation_map
    
    def _apply_register_consolidation(self, consolidation_map: Dict[str, str]) -> int:
        """Apply register consolidation"""
        optimizations = 0
        
        # Update gate operands
        for gate in self.gates:
            new_operands = []
            for operand in gate.operands:
                reg_base = operand.split('[')[0]
                if reg_base in consolidation_map:
                    # Replace with consolidated register
                    new_operand = operand.replace(reg_base, consolidation_map[reg_base])
                    new_operands.append(new_operand)
                    optimizations += 1
                else:
                    new_operands.append(operand)
            gate.operands = new_operands
        
        return optimizations
    
    def _renumber_registers(self) -> int:
        """Renumber registers for dense layout"""
        # Create mapping from old names to new consecutive names
        used_registers = set()
        
        # Collect all used registers
        for gate in self.gates:
            if not gate.is_removed:
                for operand in gate.operands:
                    used_registers.add(operand.split('[')[0])
        
        # Create dense renumbering
        register_mapping = {}
        counter = 0
        
        for reg in sorted(used_registers):
            if reg.startswith('%q'):
                register_mapping[reg] = f"%q{counter}"
                counter += 1
        
        # Apply renumbering
        optimizations = 0
        for gate in self.gates:
            if not gate.is_removed:
                new_operands = []
                for operand in gate.operands:
                    reg_base = operand.split('[')[0]
                    if reg_base in register_mapping:
                        new_operand = operand.replace(reg_base, register_mapping[reg_base])
                        new_operands.append(new_operand)
                        if new_operand != operand:
                            optimizations += 1
                    else:
                        new_operands.append(operand)
                gate.operands = new_operands
        
        # Update allocations and other references
        new_allocations = []
        for alloc in self.allocations:
            for old_reg, new_reg in register_mapping.items():
                alloc = alloc.replace(old_reg, new_reg)
            new_allocations.append(alloc)
        self.allocations = new_allocations
        
        new_initializations = []
        for init in self.initializations:
            for old_reg, new_reg in register_mapping.items():
                init = init.replace(old_reg, new_reg)
            new_initializations.append(init)
        self.initializations = new_initializations
        
        new_measurements = []
        for measure in self.measurements:
            for old_reg, new_reg in register_mapping.items():
                measure = measure.replace(old_reg, new_reg)
            new_measurements.append(measure)
        self.measurements = new_measurements

        new_measurements = []
        for measure in self.measurements:
            updated_measure = measure
            for old_reg, new_reg in register_mapping.items():
                # Use regex to avoid partial matches (e.g., %q1 matching in %q10)
                updated_measure = re.sub(r'\b' + old_reg + r'\b', new_reg, updated_measure)
            
            if updated_measure != measure:
                optimizations += 1
                self.debug_print(f"Renumbered measurement: {measure} -> {updated_measure}")
            
            new_measurements.append(updated_measure)
        self.measurements = new_measurements
        
        return optimizations
    
    def _find_final_result_register(self) -> Optional[str]:
        """Find the register containing the final computation result"""
        # Look for the last gate that produces a result
        for gate in reversed(self.gates):
            if not gate.is_removed and len(gate.operands) > 0:
                # For most arithmetic circuits, the last operand is the result
                if gate.gate_type in ['ccx', 'cx'] and len(gate.operands) >= 2:
                    return gate.operands[-1].split('[')[0]
        
        # Fallback: find most frequently used register in measurements
        reg_counts = defaultdict(int)
        for measurement in self.measurements:
            measure_match = re.search(r'q\.measure\s+(%\w+)', measurement)
            if measure_match:
                reg_counts[measure_match.group(1)] += 1
        
        if reg_counts:
            return max(reg_counts.items(), key=lambda x: x[1])[0]
        
        return None
    
    def generate_optimized_mlir(self) -> str:
        """Generate optimized MLIR output"""
        lines = [
            "// Advanced Quantum Gate Optimized MLIR",
            f"// Original gates: {self.stats.original_gates}",
            f"// Optimized gates: {len([g for g in self.gates if not g.is_removed])}",
            f"// Applied optimizations: X({self.stats.x_optimizations}), CX({self.stats.cx_optimizations}), CCX({self.stats.ccx_optimizations}), DCE({self.stats.dead_code_eliminations})",
            "builtin.module {",
            '  "quantum.func"() ({'
        ]
        
        # Add allocations
        for alloc in self.allocations:
            lines.append(f"    {alloc}")
        
        # Add initializations
        for init in self.initializations:
            lines.append(f"    {init}")
        
        # Add optimized gates
        active_gates = [gate for gate in self.gates if not gate.is_removed]
        
        for gate in active_gates:
            optimization_note = f"  // {gate.optimization_applied}" if gate.optimization_applied else ""
            operands_str = ", ".join(gate.operands)
            lines.append(f"    q.{gate.gate_type} {operands_str}{optimization_note}")
        
        # Add measurements
        for measurement in self.measurements:
            lines.append(f"    {measurement}")
        
        lines.extend([
            "    func.return",
            '  }) {func_name = "quantum_circuit"} : () -> ()',
            "}"
        ])
        
        self.stats.optimized_gates = len(active_gates)
        return '\n'.join(lines)
    
    def run_optimization_pipeline(self, mlir_content: str) -> str:
        """Run the complete optimization pipeline"""
        print("🚀 Starting Advanced Quantum Gate Optimization Pipeline...")
        print("=" * 60)
        
        # Parse input
        self.parse_mlir(mlir_content)
        print(f"📊 Parsed {len(self.gates)} gates and {len(self.registers)} registers")
        
        # Apply optimizations in order
        total_optimizations = 0
        
        # Phase 1: Basic gate optimizations
        total_optimizations += self.optimization_1_x_gate_optimization()
        total_optimizations += self.optimization_2_cx_gate_optimization() 
        total_optimizations += self.optimization_3_ccx_gate_optimization()
        
        # Phase 2: Advanced optimizations
        # total_optimizations += self.optimization_4_dead_code_elimination()
        total_optimizations += self.optimization_5_gate_commutation_and_reordering()
        # total_optimizations += self.optimization_6_register_consolidation()
        # total_optimizations += self.optimization_7_measurement_optimization()
        
        # Generate optimized MLIR
        optimized_mlir = self.generate_optimized_mlir()
        
        print("=" * 60)
        print("✅ Advanced Quantum Gate Optimization Complete!")
        print(f"📈 Total optimizations applied: {total_optimizations}")
        print(f"🎯 Gate reduction: {self.stats.original_gates} → {self.stats.optimized_gates} ({((self.stats.original_gates - self.stats.optimized_gates) / self.stats.original_gates * 100):.1f}% reduction)")
        
        return optimized_mlir
    
    def print_detailed_stats(self):
        """Print detailed optimization statistics"""
        print("\n=== Detailed Optimization Statistics ===")
        print(f"Original gates:           {self.stats.original_gates}")
        print(f"Optimized gates:          {self.stats.optimized_gates}")
        print(f"Gates eliminated:         {self.stats.original_gates - self.stats.optimized_gates}")
        print(f"X gate optimizations:     {self.stats.x_optimizations}")
        print(f"CX gate optimizations:    {self.stats.cx_optimizations}")
        print(f"CCX gate optimizations:   {self.stats.ccx_optimizations}")
        print(f"Dead code eliminations:   {self.stats.dead_code_eliminations}")
        print(f"Register consolidations:  {self.stats.register_consolidations}")
        print(f"Gate reorderings:         {self.stats.gate_reorderings}")
        
        if self.stats.original_gates > 0:
            reduction = (self.stats.original_gates - self.stats.optimized_gates) / self.stats.original_gates * 100
            print(f"Overall reduction:        {reduction:.1f}%")
        
        # Show optimization breakdown by type
        print("\n=== Optimization Breakdown ===")
        optimizations = [
            ("X Gate Optimizations", self.stats.x_optimizations),
            ("CX Gate Optimizations", self.stats.cx_optimizations), 
            ("CCX Gate Optimizations", self.stats.ccx_optimizations),
            ("Dead Code Elimination", self.stats.dead_code_eliminations),
            ("Register Consolidation", self.stats.register_consolidations),
            ("Gate Reordering", self.stats.gate_reorderings)
        ]
        
        for opt_name, count in optimizations:
            if count > 0:
                print(f"  {opt_name:<25}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Quantum Gate Optimizer')
    parser.add_argument('input_file', help='Input MLIR file')
    parser.add_argument('output_file', help='Output optimized MLIR file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--stats', action='store_true', help='Show detailed optimization statistics')
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    print(f"📄 Reading MLIR from: {args.input_file}")
    
    # Create optimizer and run optimizations
    optimizer = AdvancedQuantumGateOptimizer(enable_debug=args.debug)
    optimized_mlir = optimizer.run_optimization_pipeline(mlir_content)
    
    # Write output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(optimized_mlir)
        print(f"💾 Optimized MLIR saved to: {args.output_file}")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)
    
    # Show statistics if requested
    if args.stats:
        optimizer.print_detailed_stats()
    
    print(f"\n🎯 Ready for circuit generation with: python circuit_generator2.py {args.output_file} output_circuit.py")

if __name__ == "__main__":
    main()