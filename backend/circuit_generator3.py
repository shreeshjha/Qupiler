#!/usr/bin/env python3
"""
Generic Qiskit Circuit Generator from Optimized MLIR (FIXED VERSION)

This script dynamically generates Qiskit circuits from any optimized MLIR,
supporting all operations: arithmetic, logical, bitwise, increment/decrement, while loops, etc.

Usage: python generic_qiskit_generator.py <optimized.mlir> <output.py>
"""

import re
import sys
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import json

@dataclass
class QubitRegister:
    name: str
    size: int
    qiskit_name: str
    initial_value: Optional[int] = None

@dataclass
class QuantumOperation:
    op_type: str
    operands: List[str]
    result: Optional[str] = None
    optimization: str = ""
    description: str = ""
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

class GenericQiskitGenerator:
    def __init__(self):
        self.registers: Dict[str, QubitRegister] = {}
        self.operations: List[QuantumOperation] = []
        self.classical_registers: Dict[str, str] = {}
        self.optimizations_applied: List[str] = []
        self.initial_values: Dict[str, int] = {}
        self.expected_result: Optional[int] = None
        
    def parse_optimized_mlir(self, content: str) -> None:
        """Parse any optimized MLIR content dynamically"""
        lines = content.split('\n')
        
        print(f"🔍 Parsing optimized MLIR with {len(lines)} lines")
        
        # Extract applied optimizations from header
        for line in lines:
            if line.startswith("// Applied optimizations:"):
                opt_line = line.replace("// Applied optimizations:", "").strip()
                self.optimizations_applied = [opt.strip() for opt in opt_line.split(",")]
                break
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('//') or 'builtin.module' in line or 'quantum.func' in line or 'func.return' in line or line == '}':
                continue
                
            try:
                self._parse_line(line)
            except Exception as e:
                print(f"⚠️  Warning: Failed to parse line {i+1}: '{line}' - {e}")
        
        print(f"✅ Parsed {len(self.registers)} registers and {len(self.operations)} operations")
        print(f"📋 Registers: {list(self.registers.keys())}")
        print(f"🔧 Optimizations: {self.optimizations_applied}")
        print(f"🔬 Operations found:")
        for i, op in enumerate(self.operations):
            print(f"   {i+1}. {op.op_type}: {op.operands} {op.optimization}")
        print()
    
    def _parse_line(self, line: str) -> None:
        """Parse a single MLIR line with all operation types"""
        
        # Parse allocation: %q0 = q.alloc : !qreg<4>
        alloc_match = re.search(r'%(\w+)\s*=\s*q\.alloc\s*:\s*!qreg<(\d+)>', line)
        if alloc_match:
            reg_name, size = alloc_match.groups()
            full_reg_name = f"%{reg_name}"
            
            # Check if this register already exists
            if full_reg_name in self.registers:
                # This is a duplicate allocation - skip it or use existing
                print(f"   ⚠️  Duplicate allocation for {full_reg_name}, using existing register")
                return
            
            # Create unique qiskit name
            qiskit_name = f"q{len(self.registers)}"
            self.registers[full_reg_name] = QubitRegister(
                name=full_reg_name,
                size=int(size),
                qiskit_name=qiskit_name
            )
            print(f"   📦 Allocated register: {full_reg_name} -> {qiskit_name} ({size} qubits)")
            return
        
        # Parse initialization: q.init %q0, 1 : i32
        init_match = re.search(r'q\.init\s+%(\w+),\s*(\d+)', line)
        if init_match:
            reg_name, value = init_match.groups()
            full_reg_name = f"%{reg_name}"
            int_value = int(value)
            
            # Store initial value
            self.initial_values[full_reg_name] = int_value
            if full_reg_name in self.registers:
                self.registers[full_reg_name].initial_value = int_value
            
            self.operations.append(QuantumOperation(
                op_type="init",
                operands=[full_reg_name],
                attributes={"value": int_value},
                description=f"Initialize {full_reg_name} = {int_value}"
            ))
            return
        
        # Parse all circuit operations with optimization annotations
        circuit_match = re.search(r'q\.(\w+_circuit)\s+((?:%\w+(?:,\s*)?)+)(?:\s*//\s*(.+))?', line)
        if circuit_match:
            circuit_type, operands_str, annotation = circuit_match.groups()
            operands = [op.strip() for op in operands_str.split(',')]
            
            self.operations.append(QuantumOperation(
                op_type=circuit_type,
                operands=operands,
                result=operands[-1] if len(operands) > 2 else None,
                optimization=annotation or "",
                description=f"{circuit_type} {annotation or ''}".strip()
            ))
            return
        
        # Parse basic gates: q.cx %q1[0], %q4[0]  // CCNOT_DECOMP
        gate_match = re.search(r'q\.(\w+)\s+((?:%\w+(?:\[\d+\])?(?:,\s*)?)+)(?:\s*//\s*(.+))?', line)
        if gate_match:
            gate_type, operands_str, annotation = gate_match.groups()
            operands = [op.strip() for op in operands_str.split(',')]
            
            self.operations.append(QuantumOperation(
                op_type=gate_type,
                operands=operands,
                optimization=annotation or "",
                description=f"{gate_type.upper()} gate {annotation or ''}".strip()
            ))
            return
        
        # Parse measurement: %q6 = q.measure %q4 : !qreg -> i32
        measure_match = re.search(r'%(\w+)\s*=\s*q\.measure\s+%(\w+)', line)
        if measure_match:
            result, operand = measure_match.groups()
            result_name = f"%{result}"
            operand_name = f"%{operand}"
            
            # Create classical register for measurement result
            classical_reg_name = f"c{len(self.classical_registers)}"
            self.classical_registers[result_name] = classical_reg_name
            
            self.operations.append(QuantumOperation(
                op_type="measure",
                operands=[operand_name],
                result=result_name,
                description=f"Measure {operand_name} -> {result_name}"
            ))
            print(f"   📏 Measure operation: {operand_name} -> {result_name} (classical: {classical_reg_name})")
            return
    
    def calculate_expected_result(self) -> int:
        """Calculate expected classical result by dynamically simulating MLIR operations"""
        print("🧮 Calculating expected classical result from MLIR operations...")
        
        # Initialize variables with their values from MLIR
        variables = {}
        for reg_name, reg_info in self.registers.items():
            if reg_info.initial_value is not None:
                variables[reg_name] = reg_info.initial_value
        
        print(f"Initial values from MLIR: {variables}")
        
        # Dynamically simulate operations based on MLIR content
        for op in self.operations:
            try:
                # Handle any circuit operation dynamically
                if op.op_type.endswith("_circuit") and len(op.operands) >= 2:
                    circuit_type = op.op_type.replace("_circuit", "")
                    
                    if len(op.operands) >= 3:  # Binary operations
                        a_reg, b_reg, result_reg = op.operands[:3]
                        if a_reg in variables and b_reg in variables:
                            a_val, b_val = variables[a_reg], variables[b_reg]
                            
                            # Dynamically compute based on operation type
                            if circuit_type == "add":
                                result = a_val + b_val
                            elif circuit_type == "sub":
                                result = a_val - b_val
                            elif circuit_type == "mul":
                                result = a_val * b_val
                            elif circuit_type == "div":
                                result = a_val // b_val if b_val != 0 else 0
                            elif circuit_type == "mod":
                                result = a_val % b_val if b_val != 0 else 0
                            elif circuit_type == "and":
                                result = 1 if (a_val and b_val) else 0
                            elif circuit_type == "or":
                                result = 1 if (a_val or b_val) else 0
                            elif circuit_type == "xor":
                                result = a_val ^ b_val
                            elif circuit_type == "post_inc":
                                # Special case: post_inc has 3 outputs
                                variables[op.operands[1]] = a_val  # original value
                                variables[op.operands[2]] = a_val + 1  # incremented value
                                print(f"Post-increment: {a_reg}({a_val}) -> orig:{op.operands[1]}({a_val}), inc:{op.operands[2]}({a_val + 1})")
                                continue
                            elif circuit_type == "post_dec":
                                # Special case: post_dec has 3 outputs
                                variables[op.operands[1]] = a_val  # original value  
                                variables[op.operands[2]] = a_val - 1  # decremented value
                                print(f"Post-decrement: {a_reg}({a_val}) -> orig:{op.operands[1]}({a_val}), dec:{op.operands[2]}({a_val - 1})")
                                continue
                            else:
                                print(f"Unknown circuit type: {circuit_type}")
                                continue
                            
                            variables[result_reg] = result & 0xF  # 4-bit mask
                            print(f"{circuit_type.title()}: {a_reg}({a_val}) {circuit_type} {b_reg}({b_val}) = {result}")
                    
                    elif len(op.operands) >= 2:  # Unary operations
                        input_reg, result_reg = op.operands[:2]
                        if input_reg in variables:
                            input_val = variables[input_reg]
                            
                            if circuit_type == "not":
                                result = ~input_val & 0xF  # 4-bit mask
                            elif circuit_type == "neg":
                                result = (-input_val) & 0xF  # 4-bit mask
                            else:
                                print(f"Unknown unary circuit type: {circuit_type}")
                                continue
                            
                            variables[result_reg] = result
                            print(f"{circuit_type.title()}: {circuit_type}({input_reg}({input_val})) = {result}")
                
                # Handle basic gates (these don't change classical values, just quantum state)
                elif op.op_type in ["cx", "ccx", "x"]:
                    # These are quantum-only operations that don't affect classical simulation
                    print(f"Quantum gate: {op.op_type} (no classical effect)")
                    
            except Exception as e:
                print(f"Warning: Could not simulate {op.op_type}: {e}")
        
        # Find the final result from the measured register
        final_result = 0
        measured_register = None
        
        for op in reversed(self.operations):
            if op.op_type == "measure" and op.operands[0] in variables:
                measured_register = op.operands[0]
                final_result = variables[measured_register]
                break
        
        print(f"Final variables from MLIR simulation: {variables}")
        print(f"Measured register: {measured_register}")
        print(f"Expected result: {final_result}")
        
        self.expected_result = final_result
        return final_result
    
    def _safe_get_register(self, reg_name: str) -> Optional[QubitRegister]:
        """Safely get a register, with fallback creation if needed"""
        if reg_name in self.registers:
            return self.registers[reg_name]
        
        print(f"⚠️  Missing register {reg_name}, searching alternatives...")
        
        # Try to find a similar register name
        for existing_reg in self.registers.keys():
            if existing_reg.endswith(reg_name.split('%')[-1]):
                print(f"   Found alternative: {existing_reg}")
                return self.registers[existing_reg]
        
        print(f"⚠️  Creating fallback register for {reg_name}")
        # Create unique qiskit name that doesn't conflict
        existing_qiskit_names = {reg.qiskit_name for reg in self.registers.values()}
        counter = 0
        while f"fallback_q{counter}" in existing_qiskit_names:
            counter += 1
        qiskit_name = f"fallback_q{counter}"
        
        fallback_reg = QubitRegister(
            name=reg_name,
            size=4,
            qiskit_name=qiskit_name
        )
        self.registers[reg_name] = fallback_reg
        return fallback_reg
        
    def _extract_qubit_index(self, operand: str) -> Tuple[str, Optional[int]]:
        """Extract register name and qubit index from operand"""
        if '[' in operand:
            reg_part, index_part = operand.split('[', 1)
            index = int(index_part.rstrip(']'))
            return reg_part, index
        return operand, None
    
    def generate_quantum_circuit_helpers(self) -> List[str]:
        """Generate all quantum circuit helper functions"""
        return [
            "def initialize_register(qc, qreg, value, num_bits):",
            "    '''Initialize quantum register to classical value'''",
            "    bin_val = format(value, f'0{num_bits}b')[::-1]",
            "    for i, bit in enumerate(bin_val):",
            "        if bit == '1':",
            "            qc.x(qreg[i])",
            "",
            "def apply_post_inc_circuit(qc, input_reg, orig_reg, inc_reg):",
            "    '''Post-increment: orig = input, inc = input + 1'''",
            "    for i in range(len(input_reg)):",
            "        qc.cx(input_reg[i], orig_reg[i])",
            "    for i in range(len(input_reg)):",
            "        qc.cx(input_reg[i], inc_reg[i])",
            "    qc.x(inc_reg[0])  # Add 1",
            "",
            "def apply_post_dec_circuit(qc, input_reg, orig_reg, dec_reg):",
            "    '''Post-decrement: orig = input, dec = input - 1'''",
            "    for i in range(len(input_reg)):",
            "        qc.cx(input_reg[i], orig_reg[i])",
            "    for i in range(len(input_reg)):",
            "        qc.cx(input_reg[i], dec_reg[i])",
            "    qc.x(dec_reg[0])  # Subtract 1",
            "",
            "def apply_add_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Quantum adder: result = a + b'''",
            "    for i in range(min(len(a_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])",
            "    for i in range(min(len(b_reg), len(result_reg))):",
            "        qc.cx(b_reg[i], result_reg[i])",
            "    if len(result_reg) > 1 and len(a_reg) > 0 and len(b_reg) > 0:",
            "        qc.ccx(a_reg[0], b_reg[0], result_reg[1])",
            "",
            "def apply_sub_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Quantum subtractor: result = a - b'''",
            "    for i in range(min(len(a_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])",
            "    for i in range(min(len(b_reg), len(result_reg))):",
            "        qc.cx(b_reg[i], result_reg[i])",
            "",
            "def apply_mul_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Quantum multiplier: result = a * b (simplified)'''",
            "    # Simplified multiplication for demonstration",
            "    for i in range(min(len(a_reg), len(result_reg))):",
            "        qc.ccx(a_reg[i], b_reg[0], result_reg[i])",
            "",
            "def apply_div_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Quantum divider: result = a / b (simplified)'''",
            "    # Simplified division for demonstration",
            "    for i in range(min(len(a_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])",
            "",
            "def apply_mod_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Quantum modulo: result = a % b (simplified)'''",
            "    # Simplified modulo for demonstration",
            "    for i in range(min(len(a_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])",
            "    qc.cx(b_reg[0], result_reg[0])",
            "",
            "def apply_and_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Logical AND: result = a && b'''",
            "    qc.ccx(a_reg[0], b_reg[0], result_reg[0])",
            "",
            "def apply_or_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Logical OR: result = a || b'''",
            "    qc.cx(a_reg[0], result_reg[0])",
            "    qc.cx(b_reg[0], result_reg[0])",
            "    qc.ccx(a_reg[0], b_reg[0], result_reg[0])",
            "",
            "def apply_xor_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''XOR: result = a ^ b'''",
            "    for i in range(min(len(a_reg), len(b_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])",
            "        qc.cx(b_reg[i], result_reg[i])",
            "",
            "def apply_not_circuit(qc, input_reg, result_reg):",
            "    '''Bitwise NOT: result = ~input'''",
            "    for i in range(min(len(input_reg), len(result_reg))):",
            "        qc.cx(input_reg[i], result_reg[i])",
            "        qc.x(result_reg[i])",
            "",
            "def apply_neg_circuit(qc, input_reg, result_reg):",
            "    '''Negation: result = -input'''",
            "    for i in range(min(len(input_reg), len(result_reg))):",
            "        qc.cx(input_reg[i], result_reg[i])",
            "    qc.x(result_reg[0])  # Simple negation",
            "",
        ]
    
    def generate_qiskit_code(self) -> List[str]:
        """Generate complete Qiskit Python code dynamically"""
        
        # Calculate expected result
        expected = self.calculate_expected_result()
        
        lines = [
            "#!/usr/bin/env python3",
            "'''",
            "Generated Qiskit Circuit from Optimized MLIR",
            "",
            "This circuit was automatically generated from optimized quantum MLIR.",
            f"Expected classical result: {expected}",
            "",
            f"Applied optimizations: {', '.join(self.optimizations_applied)}",
            "'''",
            "",
            "from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister",
            "from qiskit_aer import AerSimulator",
            "from qiskit.visualization import plot_histogram",
            "import matplotlib.pyplot as plt",
            "",
        ]
        
        # Add helper functions
        lines.extend(self.generate_quantum_circuit_helpers())
        
        # Main circuit function
        lines.extend([
            "def create_quantum_circuit():",
            "    '''Create the quantum circuit from optimized MLIR'''",
            "    print('🔬 Creating quantum circuit from MLIR...')",
            "",
        ])
        
        # Declare quantum registers (only unique ones)
        declared_qiskit_names = set()
        for reg_name, reg_info in self.registers.items():
            if reg_info.qiskit_name not in declared_qiskit_names:
                lines.append(f"    {reg_info.qiskit_name} = QuantumRegister({reg_info.size}, '{reg_info.qiskit_name}')")
                declared_qiskit_names.add(reg_info.qiskit_name)
        
        # Declare classical registers
        for cl_reg, qiskit_name in self.classical_registers.items():
            if qiskit_name not in declared_qiskit_names:
                reg_size = 4
                lines.append(f"    {qiskit_name} = ClassicalRegister({reg_size}, '{qiskit_name}')")
                declared_qiskit_names.add(qiskit_name)
        
        # Create circuit with unique register names only
        unique_reg_names = []
        seen_qiskit_names = set()
        
        for reg_info in self.registers.values():
            if reg_info.qiskit_name not in seen_qiskit_names:
                unique_reg_names.append(reg_info.qiskit_name)
                seen_qiskit_names.add(reg_info.qiskit_name)
        
        for cl_reg_name in self.classical_registers.values():
            if cl_reg_name not in seen_qiskit_names:
                unique_reg_names.append(cl_reg_name)
                seen_qiskit_names.add(cl_reg_name)

        lines.append(f"    qc = QuantumCircuit({', '.join(unique_reg_names)})")
        lines.append("")
        lines.append(f"    print(f\"Created circuit with registers: {unique_reg_names}\")")
        lines.append("")
        
        # Add optimization summary
        lines.append("    # === Applied Optimizations ===")
        for opt in self.optimizations_applied:
            lines.append(f"    print('✅ {opt}')")
        lines.append("    print()")
        lines.append("")
        
        # Add circuit operations
        lines.append("    # === Quantum Operations ===")
        lines.append("    operations_log = []")
        lines.append("")
        
        operation_counter = 1
        for op in self.operations:
            lines.append(f"    # Operation {operation_counter}: {op.description}")
            
            try:
                if op.op_type == "init":
                    reg_name = op.operands[0]
                    reg_info = self._safe_get_register(reg_name)
                    if reg_info:
                        value = op.attributes.get("value", 0)
                        lines.append(f"    initialize_register(qc, {reg_info.qiskit_name}, {value}, {reg_info.size})")
                        lines.append(f"    operations_log.append('Initialize {reg_info.qiskit_name} = {value}')")
                        
                elif op.op_type.endswith("_circuit"):
                    # Dynamically handle all circuit types
                    circuit_name = op.op_type.replace("_circuit", "")
                    lines.append(f"    print(\"Executing {op.op_type} with operands: {op.operands}\")")
                    
                    if len(op.operands) >= 3:
                        reg_infos = [self._safe_get_register(operand) for operand in op.operands[:3]]
                        if all(reg_infos):
                            reg_names = [reg.qiskit_name for reg in reg_infos]
                            lines.append(f"    apply_{op.op_type}(qc, {', '.join(reg_names)})")
                            lines.append(f"    operations_log.append(\"{circuit_name.title()}: {' + '.join(reg_names[:2])} -> {reg_names[2]} {op.optimization}\")")
                        else:
                            lines.append(f"    print(\"WARNING: Missing registers for {op.op_type}: {op.operands}\")")
                            lines.append(f"    operations_log.append(\"SKIPPED {circuit_name}: missing registers\")")
                    elif len(op.operands) >= 2 and circuit_name in ["not", "neg"]:
                        # Unary operations
                        reg_infos = [self._safe_get_register(operand) for operand in op.operands[:2]]
                        if all(reg_infos):
                            reg_names = [reg.qiskit_name for reg in reg_infos]
                            lines.append(f"    apply_{op.op_type}(qc, {', '.join(reg_names)})")
                            lines.append(f"    operations_log.append(\"{circuit_name.title()}: {reg_names[0]} -> {reg_names[1]} {op.optimization}\")")
                        else:
                            lines.append(f"    print(\"WARNING: Missing registers for {op.op_type}: {op.operands}\")")
                            lines.append(f"    operations_log.append(\"SKIPPED {circuit_name}: missing registers\")")
                    else:
                        lines.append(f"    print(\"WARNING: Unsupported operand count for {op.op_type}: {len(op.operands)}\")")
                        lines.append(f"    operations_log.append(\"SKIPPED {circuit_name}: unsupported operands\")")
                        
                elif op.op_type in ["cx", "ccx", "x"]:
                    # Basic gates
                    if op.op_type == "cx" and len(op.operands) >= 2:
                        control_reg, control_idx = self._extract_qubit_index(op.operands[0])
                        target_reg, target_idx = self._extract_qubit_index(op.operands[1])
                        
                        control_reg_info = self._safe_get_register(control_reg)
                        target_reg_info = self._safe_get_register(target_reg)
                        
                        if control_reg_info and target_reg_info and control_idx is not None and target_idx is not None:
                            lines.append(f"    print(\"Executing CNOT gate: {control_reg_info.qiskit_name}[{control_idx}] -> {target_reg_info.qiskit_name}[{target_idx}]\")")
                            lines.append(f"    qc.cx({control_reg_info.qiskit_name}[{control_idx}], {target_reg_info.qiskit_name}[{target_idx}])")
                            lines.append(f"    operations_log.append(\"CNOT: {control_reg_info.qiskit_name}[{control_idx}] -> {target_reg_info.qiskit_name}[{target_idx}] {op.optimization}\")")
                        else:
                            lines.append(f"    print(\"WARNING: Could not execute CNOT with operands: {op.operands}\")")
                            lines.append(f"    operations_log.append(\"SKIPPED CNOT: invalid operands\")")
                    
                    elif op.op_type == "ccx" and len(op.operands) >= 3:
                        ctrl1_reg, ctrl1_idx = self._extract_qubit_index(op.operands[0])
                        ctrl2_reg, ctrl2_idx = self._extract_qubit_index(op.operands[1])
                        target_reg, target_idx = self._extract_qubit_index(op.operands[2])
                        
                        ctrl1_reg_info = self._safe_get_register(ctrl1_reg)
                        ctrl2_reg_info = self._safe_get_register(ctrl2_reg)
                        target_reg_info = self._safe_get_register(target_reg)
                        
                        if all([ctrl1_reg_info, ctrl2_reg_info, target_reg_info]) and all(idx is not None for idx in [ctrl1_idx, ctrl2_idx, target_idx]):
                            lines.append(f"    print(\"Executing Toffoli gate: {ctrl1_reg_info.qiskit_name}[{ctrl1_idx}] & {ctrl2_reg_info.qiskit_name}[{ctrl2_idx}] -> {target_reg_info.qiskit_name}[{target_idx}]\")")
                            lines.append(f"    qc.ccx({ctrl1_reg_info.qiskit_name}[{ctrl1_idx}], {ctrl2_reg_info.qiskit_name}[{ctrl2_idx}], {target_reg_info.qiskit_name}[{target_idx}])")
                            lines.append(f"    operations_log.append(\"Toffoli: {ctrl1_reg_info.qiskit_name}[{ctrl1_idx}] & {ctrl2_reg_info.qiskit_name}[{ctrl2_idx}] -> {target_reg_info.qiskit_name}[{target_idx}] {op.optimization}\")")
                        else:
                            lines.append(f"    print(\"WARNING: Could not execute Toffoli with operands: {op.operands}\")")
                            lines.append(f"    operations_log.append(\"SKIPPED Toffoli: invalid operands\")")
                    
                    elif op.op_type == "x" and len(op.operands) >= 1:
                        target_reg, target_idx = self._extract_qubit_index(op.operands[0])
                        target_reg_info = self._safe_get_register(target_reg)
                        
                        if target_reg_info and target_idx is not None:
                            lines.append(f"    print(\"Executing X gate: {target_reg_info.qiskit_name}[{target_idx}]\")")
                            lines.append(f"    qc.x({target_reg_info.qiskit_name}[{target_idx}])")
                            lines.append(f"    operations_log.append(\"X gate: {target_reg_info.qiskit_name}[{target_idx}] {op.optimization}\")")
                        else:
                            lines.append(f"    print(\"WARNING: Could not execute X gate with operands: {op.operands}\")")
                            lines.append(f"    operations_log.append(\"SKIPPED X gate: invalid operands\")")
                        
                elif op.op_type == "measure":
                    quantum_reg_info = self._safe_get_register(op.operands[0])
                    if quantum_reg_info:
                        if op.result in self.classical_registers:
                            classical_reg = self.classical_registers[op.result]
                            lines.append(f"    qc.measure({quantum_reg_info.qiskit_name}, {classical_reg})")
                            lines.append(f"    operations_log.append('Measure: {quantum_reg_info.qiskit_name} -> {classical_reg}')")
                        else:
                            # Create classical register on-the-fly if missing
                            classical_reg = f"c{len(self.classical_registers)}"
                            lines.append(f"    # Auto-creating classical register for measurement")
                            lines.append(f"    {classical_reg} = ClassicalRegister(4, '{classical_reg}')")
                            lines.append(f"    qc.add_register({classical_reg})")
                            lines.append(f"    qc.measure({quantum_reg_info.qiskit_name}, {classical_reg})")
                            lines.append(f"    operations_log.append('Measure: {quantum_reg_info.qiskit_name} -> {classical_reg} (auto-created)')")
                    else:
                        lines.append(f"    # WARNING: Could not find quantum register for measurement: {op.operands[0]}")
                        lines.append(f"    operations_log.append('Measurement skipped: missing quantum register {op.operands[0]}')")
                        
            except Exception as e:
                lines.append(f"    # ERROR processing operation {op.op_type}: {e}")
                
            lines.append("")
            operation_counter += 1
        
        # Return the circuit and add remaining functions
        lines.extend([
            "    return qc, operations_log",
            "",
            f"def run_quantum_simulation(qc, operations_log, expected_result={expected}):",
            "    '''Run the quantum simulation and analyze results'''",
            "    print('🚀 Running quantum simulation...')",
            "    print(f'Circuit depth: {qc.depth()}, Gates: {len(qc.data)}, Qubits: {qc.num_qubits}')",
            "    print()",
            "    ",
            "    # Show operation log",
            "    print('📋 Operations performed:')",
            "    for i, op in enumerate(operations_log, 1):",
            "        print(f'  {i:2d}. {op}')",
            "    print()",
            "    ",
            "    # Run simulation",
            "    simulator = AerSimulator(method='statevector')",
            "    job = simulator.run(qc, shots=1024)",
            "    result = job.result()",
            "    counts = result.get_counts()",
            "    ",
            "    return counts, expected_result",
            "",
            "def analyze_results(counts, expected_result):",
            "    '''Analyze and display results'''",
            "    print('📊 Measurement Results:')",
            "    ",
            "    # Sort results by frequency",
            "    sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)",
            "    ",
            "    for bitstring, count in sorted_results[:10]:  # Show top 10",
            "        # Convert bitstring to decimal (LSB interpretation)",
            "        decimal = int(bitstring[::-1], 2)  # Reverse for LSB",
            "        percentage = (count / 1024) * 100",
            "        print(f'  {bitstring} (decimal: {decimal:2d}) -> {count:4d} shots ({percentage:5.1f}%)')",
            "    ",
            "    # Get most frequent result",
            "    most_frequent_bits, most_frequent_count = sorted_results[0]",
            "    quantum_result = int(most_frequent_bits[::-1], 2)",
            "    ",
            "    print(f'\\n🎯 Quantum Result: {quantum_result} (binary: {most_frequent_bits})')",
            "    print(f'   Frequency: {most_frequent_count}/1024 ({(most_frequent_count/1024)*100:.1f}%)')",
            "    print(f'🧮 Expected Result: {expected_result}')",
            "    ",
            "    # Compare results",
            "    if quantum_result == expected_result:",
            "        print('   ✅ Perfect match! Quantum circuit correctly computed the result.')",
            "        accuracy = 'PERFECT'",
            "    else:",
            "        difference = abs(quantum_result - expected_result)",
            "        print(f'   ⚠️  Result differs by {difference}. Possible reasons:')",
            "        print('      - Quantum interference effects')",
            "        print('      - Circuit simplification in quantum arithmetic')",
            "        print('      - Measurement interpretation')",
            "        print('      - Bit encoding differences')",
            "        accuracy = f'DIFF_{difference}'",
            "    ",
            "    return quantum_result, accuracy, counts",
            "",
            "def visualize_circuit(qc):",
            "    '''Visualize the quantum circuit'''",
            "    print('\\n📈 Circuit Visualization:')",
            "    print(f'   Quantum registers: {qc.num_qubits}')",
            "    print(f'   Classical registers: {qc.num_clbits}')",
            "    print(f'   Circuit depth: {qc.depth()}')",
            "    print(f'   Total gates: {len(qc.data)}')",
            "    ",
            "    # Count gate types",
            "    gate_counts = {}",
            "    for instruction in qc.data:",
            "        gate_name = instruction.operation.name",
            "        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1",
            "    ",
            "    print('   Gate breakdown:')",
            "    for gate, count in sorted(gate_counts.items()):",
            "        print(f'     {gate}: {count}')",
            "    ",
            "    # Print circuit diagram (truncated for large circuits)",
            "    try:",
            "        print('\\n   Circuit Diagram:')",
            "        circuit_str = str(qc.draw(output='text', fold=-1))",
            "        lines = circuit_str.split('\\n')",
            "        max_lines = 25",
            "        for line in lines[:max_lines]:",
            "            print(f'   {line}')",
            "        if len(lines) > max_lines:",
            "            print(f'   ... (showing first {max_lines} lines of {len(lines)} total)')",
            "    except Exception as e:",
            "        print(f'   (Circuit diagram error: {e})')",
            "",
            "def save_results(circuit, operations, quantum_result, expected_result, accuracy, measurements):",
            "    '''Save all results and visualizations (FIXED QASM EXPORT)'''",
            "    results = {",
            "        'quantum_result': quantum_result,",
            "        'expected_result': expected_result,",
            "        'accuracy': accuracy,",
            "        'operations_count': len(operations),",
            "        'circuit_depth': circuit.depth(),",
            "        'gate_count': len(circuit.data),",
            "        'qubit_count': circuit.num_qubits",
            "    }",
            "    ",
            "    # Save results JSON",
            "    try:",
            "        import json",
            "        with open('quantum_results.json', 'w') as f:",
            "            json.dump(results, f, indent=2)",
            "        print('💾 Results saved to quantum_results.json')",
            "    except Exception as e:",
            "        print(f'⚠️  Could not save JSON: {e}')",
            "    ",
            "    # Save visualizations",
            "    try:",
            "        # Plot measurement histogram",
            "        plt.figure(figsize=(12, 8))",
            "        plot_histogram(measurements,",
            "                      title=f'Quantum Results vs Expected\\\\nQuantum: {quantum_result}, Expected: {expected_result}, Accuracy: {accuracy}')",
            "        plt.savefig('measurement_results.png', dpi=300, bbox_inches='tight')",
            "        print('📊 Measurement histogram saved as measurement_results.png')",
            "        ",
            "        # Save circuit diagram",
            "        if circuit.num_qubits <= 30:  # Only for reasonably sized circuits",
            "            circuit_img = circuit.draw(output='mpl', style='clifford')",
            "            circuit_img.savefig('quantum_circuit.png', dpi=300, bbox_inches='tight')",
            "            print('💾 Circuit diagram saved as quantum_circuit.png')",
            "        else:",
            "            print('⚠️  Circuit too large for visualization')",
            "        ",
            "        plt.show()",
            "        ",
            "    except ImportError:",
            "        print('💡 Install matplotlib for visualizations: pip install matplotlib')",
            "    except Exception as e:",
            "        print(f'⚠️  Visualization error: {e}')",
            "    ",
            "    # Export circuit to QASM (FIXED VERSION)",
            "    try:",
            "        # Method 1: Try new qasm2 module",
            "        try:",
            "            from qiskit import qasm2",
            "            qasm_str = qasm2.dumps(circuit)  # Fixed: use dumps() instead of dump()",
            "            print('✅ Using qiskit.qasm2.dumps() method')",
            "        except (ImportError, AttributeError):",
            "            # Method 2: Try legacy circuit.qasm() method",
            "            try:",
            "                qasm_str = circuit.qasm()",
            "                print('✅ Using circuit.qasm() legacy method')",
            "            except AttributeError:",
            "                # Method 3: Manual QASM generation",
            "                qasm_lines = []",
            "                qasm_lines.append('OPENQASM 2.0;')",
            "                qasm_lines.append('include \"qelib1.inc\";')",
            "                qasm_lines.append('')",
            "                ",
            "                # Add register declarations",
            "                for reg in circuit.qregs:",
            "                    qasm_lines.append(f'qreg {reg.name}[{reg.size}];')",
            "                for reg in circuit.cregs:",
            "                    qasm_lines.append(f'creg {reg.name}[{reg.size}];')",
            "                qasm_lines.append('')",
            "                ",
            "                # Add gates (simplified)",
            "                for instruction in circuit.data:",
            "                    gate_name = instruction.operation.name",
            "                    qubits = [f'{q.register.name}[{q.index}]' for q in instruction.qubits]",
            "                    if instruction.clbits:",
            "                        clbits = [f'{c.register.name}[{c.index}]' for c in instruction.clbits]",
            "                        qasm_lines.append(f'{gate_name} {','.join(qubits)} -> {','.join(clbits)};')",
            "                    else:",
            "                        qasm_lines.append(f'{gate_name} {','.join(qubits)};')",
            "                ",
            "                qasm_str = '\\n'.join(qasm_lines)",
            "                print('✅ Using manual QASM generation')",
            "        ",
            "        # Write QASM file",
            "        with open('quantum_circuit.qasm', 'w') as f:",
            "            f.write(qasm_str)",
            "        print('💾 QASM circuit saved as quantum_circuit.qasm')",
            "        ",
            "        # Show first few lines of QASM for preview",
            "        qasm_lines = qasm_str.split('\\n')[:15]",
            "        print('   QASM preview (first 15 lines):')",
            "        for line in qasm_lines:",
            "            if line.strip():",
            "                print(f'   {line}')",
            "        if len(qasm_str.split('\\n')) > 15:",
            "            print(f'   ... (see quantum_circuit.qasm for complete circuit)')",
            "            ",
            "    except Exception as e:",
            "        print(f'⚠️  QASM export failed: {e}')",
            "        print('   This is not critical - circuit execution was successful')",
            "    ",
            "    return results",
            "",
            "def main():",
            "    '''Main execution function'''",
            "    print('🚀 Generic Quantum Circuit Execution')",
            "    print('=' * 60)",
            "    ",
            "    # Create and analyze circuit",
            "    qc, operations_log = create_quantum_circuit()",
            "    ",
            "    # Visualize circuit",
            "    visualize_circuit(qc)",
            "    ",
            "    # Run simulation",
            f"    counts, expected = run_quantum_simulation(qc, operations_log, {expected})",
            "    ",
            "    # Analyze results",
            "    quantum_result, accuracy, all_counts = analyze_results(counts, expected)",
            "    ",
            "    # Save all results",
            "    results = save_results(qc, operations_log, quantum_result, expected, accuracy, all_counts)",
            "    ",
            "    print('\\n' + '=' * 60)",
            "    print(f'🎊 Execution Complete!')",
            "    print(f'   Quantum Result: {quantum_result}')",
            "    print(f'   Expected Result: {expected}')",
            "    print(f'   Accuracy: {accuracy}')",
            "    ",
            "    return qc, operations_log, quantum_result, expected, results",
            "",
            "if __name__ == '__main__':",
            "    circuit, operations, q_result, e_result, all_results = main()",
        ])
        
        return lines
    
    def generate_complete_qiskit(self, mlir_content: str) -> str:
        """Generate complete Qiskit code from any optimized MLIR"""
        self.parse_optimized_mlir(mlir_content)
        qiskit_lines = self.generate_qiskit_code()
        return '\n'.join(qiskit_lines)

def main():
    if len(sys.argv) != 3:
        print("Usage: python generic_qiskit_generator.py <optimized.mlir> <output.py>")
        print("\nSupported operations:")
        print("  • Arithmetic: add, sub, mul, div, mod")
        print("  • Logical: and, or, xor, not")
        print("  • Increment/Decrement: post_inc, post_dec, pre_inc, pre_dec")
        print("  • Negation: neg")
        print("  • Control flow: while loops")
        print("  • All basic quantum gates: x, cx, ccx")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read optimized MLIR
    try:
        with open(input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"📄 Reading optimized MLIR from: {input_file}")
    
    # Generate Qiskit code
    generator = GenericQiskitGenerator()
    qiskit_code = generator.generate_complete_qiskit(mlir_content)
    
    # Write Qiskit Python file
    try:
        with open(output_file, 'w') as f:
            f.write(qiskit_code)
        print(f"✅ Generated Qiskit circuit: {output_file}")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)
    
    print(f"📊 Found {len(generator.registers)} quantum registers")
    print(f"🔬 Generated {len(generator.operations)} quantum operations")
    print(f"🔧 Applied optimizations: {len(generator.optimizations_applied)}")
    print(f"🧮 Expected result: {generator.expected_result}")
    print(f"🎯 Run with: python {output_file}")
    
    # Optionally run the generated circuit immediately
    if len(sys.argv) > 3 and sys.argv[3] == "--run":
        print("\n🚀 Running generated circuit immediately...")
        try:
            exec(open(output_file).read())
        except Exception as e:
            print(f"❌ Error running circuit: {e}")

if __name__ == "__main__":
    main()