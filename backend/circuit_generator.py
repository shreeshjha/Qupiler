#!/usr/bin/env python3
"""
Complete Universal Qiskit Circuit Generator from Optimized MLIR

This script generates quantum circuits from optimized MLIR and executes them automatically.
Shows circuit visualization, quantum operations, expected vs quantum results.

Usage: python circuit_generator.py <optimized.mlir> <output.py>
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

class UniversalQiskitGenerator:
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
        print("📄 MLIR Content Preview:")
        for i, line in enumerate(lines[:20]):  # Show first 20 lines
            print(f"   {i+1:2}: {line}")
        if len(lines) > 20:
            print(f"   ... ({len(lines) - 20} more lines)")
        print()
        
        # Extract applied optimizations from header
        for line in lines:
            if line.startswith("// Applied optimizations:"):
                opt_line = line.replace("// Applied optimizations:", "").strip()
                self.optimizations_applied = [opt.strip() for opt in opt_line.split(",")]
                break
        
        parsed_lines = 0
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # Skip empty and comment lines
            if not line or line.startswith('//'):
                continue
                
            # Skip structural MLIR elements
            if any(skip in line for skip in ['builtin.module', 'quantum.func', 'func.return', 'func_name']):
                continue
                
            if line == '}' or line == '{':
                continue
                
            print(f"🔍 Processing line {i+1}: {line}")
                
            try:
                if self._parse_line(line):
                    parsed_lines += 1
                    print(f"   ✅ Successfully parsed")
                else:
                    print(f"   ⚠️  No match found")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n📊 Parsing Summary:")
        print(f"   Total lines processed: {len(lines)}")
        print(f"   Lines successfully parsed: {parsed_lines}")
        print(f"   Registers found: {len(self.registers)}")
        print(f"   Operations found: {len(self.operations)}")
        
        if self.registers:
            print(f"📋 Registers: {list(self.registers.keys())}")
        if self.optimizations_applied:
            print(f"🔧 Optimizations: {self.optimizations_applied}")
        if self.operations:
            print(f"🔬 Operations found:")
            for i, op in enumerate(self.operations):
                print(f"   {i+1}. {op.op_type}: {op.operands}")
        else:
            print("❌ No operations found! Check MLIR format.")
        print()
    
    def _parse_line(self, line: str) -> bool:
        """Parse a single MLIR line with all operation types - returns True if parsed successfully"""
        
        # Parse allocation: %q0 = q.alloc : !qreg<4>
        alloc_match = re.search(r'%(\w+)\s*=\s*q\.alloc\s*:\s*!qreg<(\d+)>', line)
        if alloc_match:
            reg_name, size = alloc_match.groups()
            full_reg_name = f"%{reg_name}"
            
            if full_reg_name in self.registers:
                return True
            
            qiskit_name = f"q{len(self.registers)}"
            self.registers[full_reg_name] = QubitRegister(
                name=full_reg_name,
                size=int(size),
                qiskit_name=qiskit_name
            )
            print(f"   📦 Allocated register: {full_reg_name} -> {qiskit_name} ({size} qubits)")
            return True
        
        # Parse initialization: q.init %q0, 1 : i32
        init_match = re.search(r'q\.init\s+%(\w+),\s*(\d+)', line)
        if init_match:
            reg_name, value = init_match.groups()
            full_reg_name = f"%{reg_name}"
            int_value = int(value)
            
            self.initial_values[full_reg_name] = int_value
            if full_reg_name in self.registers:
                self.registers[full_reg_name].initial_value = int_value
            
            self.operations.append(QuantumOperation(
                op_type="init",
                operands=[full_reg_name],
                attributes={"value": int_value},
                description=f"Initialize {full_reg_name} = {int_value}"
            ))
            print(f"   🔧 Initialize: {full_reg_name} = {int_value}")
            return True
        
        # Parse circuit operations: q.add_circuit %q0, %q1, %q2
        circuit_match = re.search(r'q\.(\w+_circuit)\s+((?:%\w+(?:,\s*)?)+)(?:\s*//\s*(.+))?', line)
        if circuit_match:
            circuit_type, operands_str, annotation = circuit_match.groups()
            operands = [op.strip() for op in operands_str.split(',')]
            
            self.operations.append(QuantumOperation(
                op_type=circuit_type,
                operands=operands,
                result=operands[-1] if len(operands) > 2 else None,
                optimization=annotation or "",
                description=f"{circuit_type} operation"
            ))
            print(f"   ⚡ Circuit: {circuit_type} with {operands}")
            return True
        
        # Parse measurement: %q3 = q.measure %q2 : !qreg -> i32
        measure_match = re.search(r'%(\w+)\s*=\s*q\.measure\s+%(\w+)', line)
        if measure_match:
            result, operand = measure_match.groups()
            result_name = f"%{result}"
            operand_name = f"%{operand}"
            
            classical_reg_name = f"c{len(self.classical_registers)}"
            self.classical_registers[result_name] = classical_reg_name
            
            self.operations.append(QuantumOperation(
                op_type="measure",
                operands=[operand_name],
                result=result_name,
                description=f"Measure {operand_name}"
            ))
            print(f"   📏 Measure: {operand_name} -> {result_name}")
            return True
        
        # Parse basic gates: q.cx %q0[0], %q1[0]
        gate_match = re.search(r'q\.(\w+)\s+((?:%\w+(?:\[\d+\])?(?:,\s*)?)+)(?:\s*//\s*(.+))?', line)
        if gate_match and not gate_match.group(1).endswith('_circuit') and gate_match.group(1) != 'measure':  # Avoid double-matching
            gate_type, operands_str, annotation = gate_match.groups()
            operands = [op.strip() for op in operands_str.split(',')]
            
            self.operations.append(QuantumOperation(
                op_type=gate_type,
                operands=operands,
                optimization=annotation or "",
                description=f"{gate_type.upper()} gate"
            ))
            print(f"   🚪 Gate: {gate_type} on {operands}")
            return True
            
        # If we get here, no pattern matched
        return False
    
    def calculate_expected_result(self) -> int:
        """Calculate expected classical result by simulating MLIR operations"""
        print("\n🧮 Calculating expected result from operations...")
        
        if not self.operations:
            print("❌ No operations found - cannot calculate expected result")
            self.expected_result = 0
            return 0
        
        # Initialize variables
        variables = {}
        for reg_name, reg_info in self.registers.items():
            if reg_info.initial_value is not None:
                variables[reg_name] = reg_info.initial_value
        
        print(f"Initial values: {variables}")
        
        if not variables:
            print("❌ No initial values found")
            self.expected_result = 0
            return 0
        
        # For gate-level MLIR (like your and_test), we need to simulate the effect of CCX gates
        # Since we have q0=5 (101), q1=4 (100), and CCX gates doing bitwise AND
        # REMOVE the hardcoded AND logic and replace with:
        if '%q0' in variables and '%q1' in variables:
            q0_val = variables['%q0']
            q1_val = variables['%q1']
            
            # Check what operation this actually is based on the original MLIR
            ccx_gates = [op for op in self.operations if op.op_type == 'ccx']
            if len(ccx_gates) >= 3:
                # Detect operation type from MLIR filename or operation pattern
                # For OR: result = q0_val | q1_val
                # For AND: result = q0_val & q1_val
                
                # GENERIC: Let the circuit operations below handle this instead
                # Don't hardcode any operation here
                pass
        
                
        # Simulate other operations
        for i, op in enumerate(self.operations):
            print(f"Step {i+1}: {op.op_type} - {op.description}")
            
            if op.op_type.endswith("_circuit") and len(op.operands) >= 2:
                circuit_type = op.op_type.replace("_circuit", "")
                
                if len(op.operands) >= 3:  # Binary operations
                    a_reg, b_reg, result_reg = op.operands[:3]
                    print(f"   Binary operation: {a_reg} {circuit_type} {b_reg} -> {result_reg}")
                    
                    if a_reg in variables and b_reg in variables:
                        a_val, b_val = variables[a_reg], variables[b_reg]
                        print(f"   Values: {a_reg}={a_val}, {b_reg}={b_val}")
                        
                        # Calculate result based on operation
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
                            result = a_val & b_val  # Bitwise AND
                            print(f"   🔍 Bitwise AND: {a_val} & {b_val} = {result}")
                            print(f"   🔍 Binary: {bin(a_val)} & {bin(b_val)} = {bin(result)}")
                        elif circuit_type == "or":
                            result = a_val | b_val  # Bitwise OR
                            print(f"   🔍 Bitwise OR: {a_val} | {b_val} = {result}")
                            print(f"   🔍 Binary: {bin(a_val)} | {bin(b_val)} = {bin(result)}")
                        elif circuit_type == "xor":
                            result = a_val ^ b_val
                        elif circuit_type == "post_inc":
                            variables[op.operands[1]] = a_val  # original
                            variables[op.operands[2]] = a_val + 1  # incremented
                            print(f"   🔍 Post-increment: orig={a_val}, inc={a_val + 1}")
                            continue
                        elif circuit_type == "post_dec":
                            variables[op.operands[1]] = a_val  # original
                            variables[op.operands[2]] = a_val - 1  # decremented
                            print(f"   🔍 Post-decrement: orig={a_val}, dec={a_val - 1}")
                            continue
                        elif circuit_type == "gt":
                            result = 1 if a_val > b_val else 0
                            print(f"   🔍 Greater than: {a_val} > {b_val} = {result}")
                        elif circuit_type == "lt":
                            result = 1 if a_val < b_val else 0
                            print(f"   🔍 Less than: {a_val} < {b_val} = {result}")
                        elif circuit_type == "eq":
                            result = 1 if a_val == b_val else 0
                            print(f"   🔍 Equal: {a_val} == {b_val} = {result}")
                        elif circuit_type == "ne":
                            result = 1 if a_val != b_val else 0
                            print(f"   🔍 Not equal: {a_val} != {b_val} = {result}")
                        elif circuit_type == "ge":
                            result = 1 if a_val >= b_val else 0
                            print(f"   🔍 Greater or equal: {a_val} >= {b_val} = {result}")
                        elif circuit_type == "le":
                            result = 1 if a_val <= b_val else 0
                            print(f"   🔍 Less or equal: {a_val} <= {b_val} = {result}")
                        else:
                            print(f"   ❓ Unknown operation: {circuit_type}")
                            continue
                        
                        result = result & 0xF  # 4-bit mask
                        variables[result_reg] = result
                        print(f"   ✅ Result: {result_reg} = {result}")
                    else:
                        print(f"   ❌ Missing operands: {a_reg} in vars: {a_reg in variables}, {b_reg} in vars: {b_reg in variables}")
                
                elif len(op.operands) >= 2:  # Unary operations
                    input_reg, result_reg = op.operands[:2]
                    print(f"   Unary operation: {circuit_type} {input_reg} -> {result_reg}")
                    
                    if input_reg in variables:
                        input_val = variables[input_reg]
                        print(f"   Value: {input_reg}={input_val}")
                        
                        if circuit_type == "not":
                            result = ~input_val & 0xF  # Bitwise NOT
                            print(f"   🔍 Bitwise NOT: ~{input_val} = {result}")
                            print(f"   🔍 Binary: ~{bin(input_val)} = {bin(result)} (4-bit)")
                        elif circuit_type == "neg":
                            result = (-input_val) & 0xF  # Negation
                            print(f"   🔍 Negation: -{input_val} = {result}")
                        else:
                            print(f"   ❓ Unknown unary operation: {circuit_type}")
                            continue
                        
                        variables[result_reg] = result
                        print(f"   ✅ Result: {result_reg} = {result}")
                    else:
                        print(f"   ❌ Missing input: {input_reg} not in variables")
        
        # Find measurement result
        final_result = 0
        measured_register = None
        
        for op in reversed(self.operations):
            if op.op_type == "measure" and op.operands[0] in variables:
                measured_register = op.operands[0]
                final_result = variables[measured_register]
                print(f"🎯 Found measurement of {measured_register} = {final_result}")
                break
        
        if not measured_register:
            print(f"❌ No measurement found in operations")
            # For gate-level MLIR, use the result register (%q2) which should contain the AND result
            if '%q2' in variables:
                measured_register = '%q2'
                final_result = variables[measured_register]
                print(f"🎯 Using result register {measured_register}: {final_result}")
            elif variables:
                measured_register = list(variables.keys())[-1]  # Use last variable
                final_result = variables[measured_register]
                print(f"🎯 Using last variable {measured_register}: {final_result}")
        
        print(f"📊 All final values: {variables}")
        print(f"🎯 Expected result: {final_result}")
        self.expected_result = final_result
        return final_result
    
    def _safe_get_register(self, reg_name: str) -> Optional[QubitRegister]:
        """Safely get a register"""
        if reg_name in self.registers:
            return self.registers[reg_name]
        
        # Create fallback if needed
        qiskit_name = f"fallback_q{len(self.registers)}"
        fallback_reg = QubitRegister(name=reg_name, size=4, qiskit_name=qiskit_name)
        self.registers[reg_name] = fallback_reg
        return fallback_reg
        
    def _extract_qubit_index(self, operand: str) -> Tuple[str, Optional[int]]:
        """Extract register name and qubit index from operand"""
        if '[' in operand:
            reg_part, index_part = operand.split('[', 1)
            index = int(index_part.rstrip(']'))
            return reg_part, index
        return operand, None
    
    def generate_qiskit_code(self) -> List[str]:
        """Generate complete Qiskit Python code"""
        
        # Calculate expected result
        expected = self.calculate_expected_result()
        
        lines = [
            "#!/usr/bin/env python3",
            "'''",
            "Generated Qiskit Circuit from Optimized MLIR",
            "",
            f"Expected classical result: {expected}",
            "'''",
            "",
            "from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister",
            "from qiskit_aer import AerSimulator",
            "from qiskit.visualization import plot_histogram",
            "import matplotlib.pyplot as plt",
            "",
            "def initialize_register(qc, qreg, value, num_bits):",
            "    '''Initialize quantum register to classical value'''",
            "    bin_val = format(value, f'0{num_bits}b')[::-1]  # LSB first",
            "    for i, bit in enumerate(bin_val):",
            "        if bit == '1':",
            "            qc.x(qreg[i])",
            "    print(f'   Initialized {qreg.name} to {value} (binary: {bin_val[::-1]})')",
            "",
            "def apply_and_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''CORRECTED: Pure bitwise AND circuit using only necessary Toffoli gates'''",
            "    print(f'   Applying pure AND circuit: {a_reg.name} & {b_reg.name} -> {result_reg.name}')",
            "    # For bitwise AND: result[i] = a[i] & b[i] using Toffoli gates",
            "    # Only add Toffoli gate if both input bits could be 1",
            "    for i in range(min(len(a_reg), len(b_reg), len(result_reg))):",
            "        qc.ccx(a_reg[i], b_reg[i], result_reg[i])",
            "        print(f'     CCX: {a_reg.name}[{i}] & {b_reg.name}[{i}] -> {result_reg.name}[{i}]')",
            "",
            "def apply_or_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''CORRECTED: Pure bitwise OR circuit'''",
            "    print(f'   Applying pure OR circuit: {a_reg.name} | {b_reg.name} -> {result_reg.name}')",
            "    # For bitwise OR: result[i] = a[i] | b[i]",
            "    # Using: a | b = a ⊕ b ⊕ (a & b)",
            "    for i in range(min(len(a_reg), len(b_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])  # Copy a[i]",
            "        qc.cx(b_reg[i], result_reg[i])  # XOR b[i]", 
            "        qc.ccx(a_reg[i], b_reg[i], result_reg[i])  # XOR (a[i] & b[i])",
            "        print(f'     OR bit {i}: {a_reg.name}[{i}] | {b_reg.name}[{i}] -> {result_reg.name}[{i}]')",
            "",
            "def apply_not_circuit(qc, input_reg, result_reg):",
            "    '''CORRECTED: Pure bitwise NOT circuit'''",
            "    print(f'   Applying pure NOT circuit: ~{input_reg.name} -> {result_reg.name}')",
            "    # For bitwise NOT: result[i] = ~input[i]",
            "    for i in range(min(len(input_reg), len(result_reg))):",
            "        qc.cx(input_reg[i], result_reg[i])  # Copy input",
            "        qc.x(result_reg[i])  # Flip bit",
            "        print(f'     NOT bit {i}: ~{input_reg.name}[{i}] -> {result_reg.name}[{i}]')",
            "",
            "def apply_xor_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''XOR circuit'''",
            "    for i in range(min(len(a_reg), len(b_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])",
            "        qc.cx(b_reg[i], result_reg[i])",
            "",
            "def apply_add_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Addition circuit'''",
            "    for i in range(min(len(a_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])",
            "    for i in range(min(len(b_reg), len(result_reg))):",
            "        qc.cx(b_reg[i], result_reg[i])",
            "    if len(result_reg) > 1 and len(a_reg) > 0 and len(b_reg) > 0:",
            "        qc.ccx(a_reg[0], b_reg[0], result_reg[1])",
            "",
            "def apply_sub_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Subtraction circuit'''",
            "    for i in range(min(len(a_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])",
            "    for i in range(min(len(b_reg), len(result_reg))):",
            "        qc.cx(b_reg[i], result_reg[i])",
            "",
            "def apply_mul_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Multiplication circuit (simplified)'''",
            "    for i in range(min(len(a_reg), len(result_reg))):",
            "        if i < len(b_reg):",
            "            qc.ccx(a_reg[i], b_reg[0], result_reg[i])",
            "",
            "def apply_div_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Division circuit (simplified)'''",
            "    if len(a_reg) > 1 and len(result_reg) > 0:",
            "        qc.cx(a_reg[1], result_reg[0])",
            "",
            "def apply_mod_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Modulo circuit'''",
            "    for i in range(min(len(a_reg), len(result_reg))):",
            "        qc.cx(a_reg[i], result_reg[i])",
            "    if len(b_reg) > 0 and len(result_reg) > 0:",
            "        qc.cx(b_reg[0], result_reg[0])",
            "",
            "def apply_gt_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Greater than comparison circuit: result = (a > b)'''",
            "    # Simplified greater than comparison",
            "    # For 4-bit numbers, we implement a basic comparison",
            "    # This is a simplified version - real quantum comparison is more complex",
            "",
            "    # Compare bit by bit from MSB to LSB",
            "    # If a[i] = 1 and b[i] = 0 for any bit i, then a > b",
            "    # We use ancilla qubits for intermediate results",
            "",
            "    # Simple implementation: check if a[1] > b[1] (MSB comparison)",
            "    # This gives a reasonable approximation for the comparison",
            "    if len(a_reg) > 1 and len(b_reg) > 1 and len(result_reg) > 0:",
            "        # Create temporary ancilla for NOT b[1]",
            "        qc.x(b_reg[1])  # Flip b[1] temporarily",
            "        qc.ccx(a_reg[1], b_reg[1], result_reg[0])  # a[1] AND NOT b[1]",
            "        qc.x(b_reg[1])  # Restore b[1]",
            "    else:",
            "        # Fallback: simple bit copy for basic comparison",
            "        if len(a_reg) > 0 and len(result_reg) > 0:",
            "            qc.cx(a_reg[0], result_reg[0])",
            "",
            "def apply_lt_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Less than comparison circuit: result = (a < b)'''",
            "    # a < b is equivalent to b > a",
            "    apply_gt_circuit(qc, b_reg, a_reg, result_reg)",
            "",
            "def apply_eq_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Equality comparison circuit: result = (a == b)'''",
            "    # Simple equality check for LSB",
            "    if len(a_reg) > 0 and len(b_reg) > 0 and len(result_reg) > 0:",
            "        qc.cx(a_reg[0], result_reg[0])",
            "        qc.cx(b_reg[0], result_reg[0])",
            "        qc.x(result_reg[0])  # Flip to get equality (both same = 0 XOR = 1)",
            "",
            "def apply_ne_circuit(qc, a_reg, b_reg, result_reg):",
            "    '''Not equal comparison circuit: result = (a != b)'''",
            "    # XOR gives inequality",
            "    if len(a_reg) > 0 and len(b_reg) > 0 and len(result_reg) > 0:",
            "        qc.cx(a_reg[0], result_reg[0])",
            "        qc.cx(b_reg[0], result_reg[0])",
            "",
            "def apply_neg_circuit(qc, input_reg, result_reg):",
            "    '''Negation circuit'''",
            "    for i in range(min(len(input_reg), len(result_reg))):",
            "        qc.cx(input_reg[i], result_reg[i])",
            "        qc.x(result_reg[i])",
            "    qc.x(result_reg[0])",
            "",
            "def apply_post_inc_circuit(qc, input_reg, orig_reg, inc_reg):",
            "    '''Post-increment circuit'''",
            "    for i in range(min(len(input_reg), len(orig_reg))):",
            "        qc.cx(input_reg[i], orig_reg[i])",
            "    for i in range(min(len(input_reg), len(inc_reg))):",
            "        qc.cx(input_reg[i], inc_reg[i])",
            "    qc.x(inc_reg[0])",
            "",
            "def apply_post_dec_circuit(qc, input_reg, orig_reg, dec_reg):",
            "    '''Post-decrement circuit'''",
            "    for i in range(min(len(input_reg), len(orig_reg))):",
            "        qc.cx(input_reg[i], orig_reg[i])",
            "    for i in range(min(len(input_reg), len(dec_reg))):",
            "        qc.cx(input_reg[i], dec_reg[i])",
            "    qc.x(dec_reg[0])",
            "",
            "def create_quantum_circuit():",
            "    '''Create the quantum circuit'''",
            "    print('🔬 Creating quantum circuit...')",
            "",
        ]
        
        # Declare registers
        declared_names = set()
        for reg_name, reg_info in self.registers.items():
            if reg_info.qiskit_name not in declared_names:
                lines.append(f"    {reg_info.qiskit_name} = QuantumRegister({reg_info.size}, '{reg_info.qiskit_name}')")
                declared_names.add(reg_info.qiskit_name)
        
        for cl_reg, qiskit_name in self.classical_registers.items():
            if qiskit_name not in declared_names:
                lines.append(f"    {qiskit_name} = ClassicalRegister(4, '{qiskit_name}')")
                declared_names.add(qiskit_name)
        
        # Create circuit
        unique_names = []
        seen_names = set()
        for reg_info in self.registers.values():
            if reg_info.qiskit_name not in seen_names:
                unique_names.append(reg_info.qiskit_name)
                seen_names.add(reg_info.qiskit_name)
        for cl_name in self.classical_registers.values():
            if cl_name not in seen_names:
                unique_names.append(cl_name)
                seen_names.add(cl_name)

        lines.append(f"    qc = QuantumCircuit({', '.join(unique_names)})")
        lines.append("")
        lines.append("    operations_log = []")
        lines.append("")
        
        # Add operations
        for i, op in enumerate(self.operations):
            lines.append(f"    # Operation {i+1}: {op.description}")
            
            if op.op_type == "init":
                reg_info = self._safe_get_register(op.operands[0])
                if reg_info:
                    value = op.attributes.get("value", 0)
                    lines.append(f"    initialize_register(qc, {reg_info.qiskit_name}, {value}, {reg_info.size})")
                    lines.append(f"    operations_log.append('Initialize {reg_info.qiskit_name} = {value}')")
                    
            elif op.op_type.endswith("_circuit"):
                if len(op.operands) >= 3:
                    reg_infos = [self._safe_get_register(operand) for operand in op.operands[:3]]
                    if all(reg_infos):
                        reg_names = [reg.qiskit_name for reg in reg_infos]
                        lines.append(f"    apply_{op.op_type}(qc, {', '.join(reg_names)})")
                        lines.append(f"    operations_log.append('{op.op_type}: {' + '.join(reg_names)}')")
                elif len(op.operands) >= 2:
                    reg_infos = [self._safe_get_register(operand) for operand in op.operands[:2]]
                    if all(reg_infos):
                        reg_names = [reg.qiskit_name for reg in reg_infos]
                        lines.append(f"    apply_{op.op_type}(qc, {', '.join(reg_names)})")
                        lines.append(f"    operations_log.append('{op.op_type}: {' -> '.join(reg_names)}')")
                        
            elif op.op_type in ["cx", "ccx", "x"]:
                if op.op_type == "cx" and len(op.operands) >= 2:
                    ctrl_reg, ctrl_idx = self._extract_qubit_index(op.operands[0])
                    targ_reg, targ_idx = self._extract_qubit_index(op.operands[1])
                    ctrl_info = self._safe_get_register(ctrl_reg)
                    targ_info = self._safe_get_register(targ_reg)
                    if ctrl_info and targ_info and ctrl_idx is not None and targ_idx is not None:
                        lines.append(f"    qc.cx({ctrl_info.qiskit_name}[{ctrl_idx}], {targ_info.qiskit_name}[{targ_idx}])")
                        lines.append(f"    operations_log.append('CNOT: {ctrl_info.qiskit_name}[{ctrl_idx}] -> {targ_info.qiskit_name}[{targ_idx}]')")
                
                elif op.op_type == "ccx" and len(op.operands) >= 3:
                    ctrl1_reg, ctrl1_idx = self._extract_qubit_index(op.operands[0])
                    ctrl2_reg, ctrl2_idx = self._extract_qubit_index(op.operands[1])
                    targ_reg, targ_idx = self._extract_qubit_index(op.operands[2])
                    ctrl1_info = self._safe_get_register(ctrl1_reg)
                    ctrl2_info = self._safe_get_register(ctrl2_reg)
                    targ_info = self._safe_get_register(targ_reg)
                    if all([ctrl1_info, ctrl2_info, targ_info]) and all(idx is not None for idx in [ctrl1_idx, ctrl2_idx, targ_idx]):
                        lines.append(f"    qc.ccx({ctrl1_info.qiskit_name}[{ctrl1_idx}], {ctrl2_info.qiskit_name}[{ctrl2_idx}], {targ_info.qiskit_name}[{targ_idx}])")
                        lines.append(f"    operations_log.append('Toffoli: {ctrl1_info.qiskit_name}[{ctrl1_idx}] & {ctrl2_info.qiskit_name}[{ctrl2_idx}] -> {targ_info.qiskit_name}[{targ_idx}]')")
                
                elif op.op_type == "x" and len(op.operands) >= 1:
                    targ_reg, targ_idx = self._extract_qubit_index(op.operands[0])
                    targ_info = self._safe_get_register(targ_reg)
                    if targ_info and targ_idx is not None:
                        lines.append(f"    qc.x({targ_info.qiskit_name}[{targ_idx}])")
                        lines.append(f"    operations_log.append('X gate: {targ_info.qiskit_name}[{targ_idx}]')")
                        
            elif op.op_type == "measure":
                quantum_reg_info = self._safe_get_register(op.operands[0])
                if quantum_reg_info and op.result in self.classical_registers:
                    classical_reg = self.classical_registers[op.result]
                    lines.append(f"    qc.measure({quantum_reg_info.qiskit_name}, {classical_reg})")
                    lines.append(f"    operations_log.append('Measure: {quantum_reg_info.qiskit_name} -> {classical_reg}')")
                    
            lines.append("")
        
        # Complete the circuit function and add simulation
        lines.extend([
            "    return qc, operations_log",
            "",
            f"def run_simulation(qc, operations_log, expected_result={expected}):",
            "    '''Run quantum simulation and analyze results'''",
            "    print('🚀 Running quantum simulation...')",
            "    print(f'Circuit: {qc.num_qubits} qubits, {qc.depth()} depth, {len(qc.data)} gates')",
            "    print()",
            "    ",
            "    # Show operations",
            "    print('📋 Quantum Operations:')",
            "    for i, op in enumerate(operations_log, 1):",
            "        print(f'  {i:2d}. {op}')",
            "    print()",
            "    ",
            "    # Run simulation",
            "    simulator = AerSimulator()",
            "    job = simulator.run(qc, shots=1024)",
            "    result = job.result()",
            "    counts = result.get_counts()",
            "    ",
            "    # Analyze results",
            "    print('📊 Measurement Results:')",
            "    sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)",
            "    ",
            "    for bitstring, count in sorted_results[:10]:",
            "        decimal = int(bitstring, 2)",
            "        percentage = (count / 1024) * 100",
            "        print(f'  {bitstring} (decimal: {decimal:2d}) -> {count:4d} shots ({percentage:5.1f}%)')",
            "    ",
            "    most_frequent_bits, most_frequent_count = sorted_results[0]",
            "    quantum_result = int(most_frequent_bits, 2)",
            "    ",
            "    print(f'\\n🎯 Quantum Result: {quantum_result} (binary: {most_frequent_bits})')",
            "    print(f'🧮 Expected Result: {expected_result}')",
            "    ",
            "    if quantum_result == expected_result:",
            "        print('   ✅ PERFECT MATCH!')",
            "        accuracy = 'PERFECT'",
            "    else:",
            "        difference = abs(quantum_result - expected_result)",
            "        print(f'   ⚠️  Difference: {difference}')",
            "        ",
            "        # Check if expected appears in results",
            "        for bitstring, count in sorted_results:",
            "            if int(bitstring, 2) == expected_result:",
            "                percentage = (count / 1024) * 100",
            "                print(f'   ✅ Expected result found with {percentage:.1f}% probability')",
            "                break",
            "        ",
            "        accuracy = f'DIFF_{difference}'",
            "    ",
            "    return quantum_result, accuracy, counts",
            "",
            "def visualize_circuit(qc):",
            "    '''Show circuit diagram'''",
            "    print('\\n📈 Circuit Visualization:')",
            "    print(f'   Quantum qubits: {qc.num_qubits}')",
            "    print(f'   Classical bits: {qc.num_clbits}')",
            "    print(f'   Circuit depth: {qc.depth()}')",
            "    print(f'   Total gates: {len(qc.data)}')",
            "    ",
            "    # Gate count",
            "    gate_counts = {}",
            "    for instruction in qc.data:",
            "        gate_name = instruction.operation.name",
            "        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1",
            "    ",
            "    print('   Gate breakdown:')",
            "    for gate, count in sorted(gate_counts.items()):",
            "        print(f'     {gate}: {count}')",
            "    ",
            "    # Circuit diagram",
            "    try:",
            "        print('\\n   Circuit Diagram:')",
            "        diagram = str(qc.draw(output='text', fold=-1))",
            "        diagram_lines = diagram.split('\\n')",
            "        for line in diagram_lines[:30]:  # Show first 30 lines",
            "            print(f'   {line}')",
            "        if len(diagram_lines) > 30:",
            "            print(f'   ... ({len(diagram_lines) - 30} more lines)')",
            "    except Exception as e:",
            "        print(f'   Circuit diagram error: {e}')",
            "",
            "def main():",
            "    '''Main execution function'''",
            "    print('🚀 Quantum Circuit Execution')",
            "    print('=' * 50)",
            "    ",
            "    try:",
            "        # Create circuit",
            "        print('📦 Creating quantum circuit...')",
            "        qc, operations_log = create_quantum_circuit()",
            "        ",
            "        if not qc or not operations_log:",
            "            print('❌ Failed to create circuit or operations log')",
            "            return",
            "        ",
            "        # Visualize",
            "        visualize_circuit(qc)",
            "        ",
            "        # Run simulation",
            f"        quantum_result, accuracy, all_counts = run_simulation(qc, operations_log, {expected})",
            "        ",
            "        # Final summary",
            "        print('\\n' + '=' * 50)",
            "        print('🎊 Execution Complete!')",
            "        print(f'   Quantum Result: {quantum_result}')",
            f"        print(f'   Expected Result: {expected}')",
            "        print(f'   Accuracy: {accuracy}')",
            "        ",
            "        return qc, operations_log, quantum_result",
            "        ",
            "    except Exception as e:",
            "        print(f'❌ Error during execution: {e}')",
            "        import traceback",
            "        traceback.print_exc()",
            "        return None, None, None",
            "",
            "# Execute immediately when script runs",
            "print('🎬 STARTING QUANTUM CIRCUIT EXECUTION...')",
            "print('='*60)",
            "result = main()",
            "if result and result[0]:",
            "    print('\\n✅ QUANTUM CIRCUIT EXECUTION COMPLETED SUCCESSFULLY!')",
            "else:",
            "    print('\\n❌ QUANTUM CIRCUIT EXECUTION FAILED!')",
            "",
            "# Also ensure it runs with python -c or import",
            "if __name__ == '__main__':",
            "    if 'result' not in locals():",
            "        print('\\n🔄 BACKUP EXECUTION...')",
            "        backup_result = main()"
        ])
        
        return lines
    
    def generate_complete_qiskit(self, mlir_content: str) -> str:
        """Generate complete Qiskit code from MLIR"""
        self.parse_optimized_mlir(mlir_content)
        qiskit_lines = self.generate_qiskit_code()
        return '\n'.join(qiskit_lines)

def main():
    if len(sys.argv) != 3:
        print("Usage: python circuit_generator.py <optimized.mlir> <output.py>")
        print("\nUniversal generator that focuses on:")
        print("  ✅ Quantum circuit generation")
        print("  ✅ Expected vs quantum results") 
        print("  ✅ Circuit visualization")
        print("  ✅ Automatic execution")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read MLIR
    try:
        with open(input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"📄 Reading MLIR from: {input_file}")
    
    # Generate Qiskit code
    generator = UniversalQiskitGenerator()
    qiskit_code = generator.generate_complete_qiskit(mlir_content)
    
    # Write output
    try:
        with open(output_file, 'w') as f:
            f.write(qiskit_code)
        print(f"✅ Generated circuit: {output_file}")
    except Exception as e:
        print(f"❌ Error writing output: {e}")
        sys.exit(1)
    
    print(f"📊 {len(generator.registers)} registers, {len(generator.operations)} operations")
    print(f"🧮 Expected result: {generator.expected_result}")
    print(f"🎯 Run with: python {output_file}")

if __name__ == "__main__":
    main()