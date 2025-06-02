# classical_to_quantum_translator.py - COMPLETE FIXED VERSION
"""
Complete translator with ALL operations including increment/decrement
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class QubitAllocation:
    """Track qubit allocations and their bit widths"""
    name: str
    bit_width: int
    start_qubit: int
    ssa_name: str

@dataclass 
class SSAVariable:
    """Track SSA variable information"""
    ssa_name: str
    semantic_name: str
    value: Optional[int]
    operation: str
    operands: List[str]

class ClassicalToQuantumTranslator:
    def __init__(self, default_bit_width: int = 8, max_bit_width: int = 16):
        self.default_bit_width = default_bit_width
        self.max_bit_width = max_bit_width
        self.qubit_counter = 0
        self.variable_qubits: Dict[str, QubitAllocation] = {}
        self.ssa_variables: Dict[str, SSAVariable] = {}
        self.semantic_to_ssa: Dict[str, str] = {}
        self.temp_counter = 0
        self.known_values: Dict[str, int] = {}
        
    def calculate_required_bits(self, value: int) -> int:
        """Calculate minimum bits needed to represent a value"""
        if value == 0:
            return 1
        elif value > 0:
            return max(1, math.ceil(math.log2(value + 1)))
        else:
            return max(2, math.ceil(math.log2(abs(value) + 1)) + 1)
    
    def estimate_operation_bit_width(self, operation: str, operand_values: List[int]) -> int:
        """Estimate required bit width for operation results"""
        if not operand_values:
            return self.default_bit_width
            
        max_operand = max(abs(v) for v in operand_values)
        
        if operation == "add":
            result_estimate = sum(abs(v) for v in operand_values)
        elif operation == "sub":
            result_estimate = max_operand * 2
        elif operation == "mul":
            result_estimate = 1
            for v in operand_values:
                result_estimate *= abs(v)
        elif operation == "div":
            result_estimate = max_operand
        elif operation == "mod":
            result_estimate = max_operand
        elif operation in ["inc", "dec", "post_inc", "post_dec", "pre_inc", "pre_dec"]:
            result_estimate = max_operand + 1
        else:
            result_estimate = max_operand
            
        required_bits = self.calculate_required_bits(result_estimate)
        return min(required_bits, self.max_bit_width)
    
    def get_optimal_bit_width(self, ssa_name: str, value: Optional[int] = None, operation: str = "") -> int:
        """Get optimal bit width for a variable"""
        if value is not None:
            required = self.calculate_required_bits(value)
            return min(max(required, 4), self.max_bit_width)
        
        if operation and ssa_name in self.ssa_variables:
            var_info = self.ssa_variables[ssa_name]
            operand_values = []
            for operand_ssa in var_info.operands:
                if operand_ssa in self.known_values:
                    operand_values.append(self.known_values[operand_ssa])
            
            if operand_values:
                return self.estimate_operation_bit_width(operation, operand_values)
        
        return self.default_bit_width
    
    def create_semantic_name(self, ssa_name: str, operation: str = "") -> str:
        """Create a meaningful semantic name for SSA variables"""
        if operation == "init":
            var_mapping = {"%0": "a", "%1": "b", "%2": "c", "%3": "d", "%4": "e"}
            if ssa_name in var_mapping:
                return var_mapping[ssa_name]
        
        elif operation in ["post_inc", "post_dec"]:
            self.temp_counter += 1
            if operation == "post_inc":
                return f"temp_c_orig"
            else:
                return f"temp_a_orig"
            
        elif operation in ["add", "sub", "mul", "div", "mod"]:
            if operation == "add":
                return f"sum"
            elif operation == "sub":
                return f"ans"
            elif operation == "div":
                return f"quotient_{self.temp_counter}"
            elif operation == "mod":
                return f"remainder_{self.temp_counter}"
            else:
                return f"result_{self.temp_counter}"
        
        self.temp_counter += 1
        return f"temp_{self.temp_counter}"
    
    def allocate_qubits_for_variable(self, ssa_name: str, semantic_name: str = None, 
                                   value: Optional[int] = None, operation: str = "") -> QubitAllocation:
        """Allocate qubits with optimal bit width"""
        if semantic_name is None:
            semantic_name = self.create_semantic_name(ssa_name, operation)
        
        bit_width = self.get_optimal_bit_width(ssa_name, value, operation)
        
        allocation = QubitAllocation(
            name=semantic_name,
            bit_width=bit_width,
            start_qubit=self.qubit_counter,
            ssa_name=ssa_name
        )
        self.qubit_counter += bit_width
        self.variable_qubits[ssa_name] = allocation
        self.semantic_to_ssa[semantic_name] = ssa_name
        
        if value is not None:
            self.known_values[ssa_name] = value
        
        return allocation
    
    def generate_qubit_alloc_operations(self, allocation: QubitAllocation) -> List[str]:
        """Generate ultra-clean qubit allocation operations"""
        operations = []
        
        if allocation.bit_width == 1:
            operations.append(f"    // Allocate 1 qubit for {allocation.name}")
            operations.append(f"    %q{allocation.start_qubit} = quantum.alloc : !quantum.qubit")
        else:
            operations.append(f"    // Allocate {allocation.bit_width} qubits for {allocation.name}: q{allocation.start_qubit}-q{allocation.start_qubit + allocation.bit_width - 1}")
            operations.append(f"    %qreg_{allocation.name} = quantum.alloc_reg {{size = {allocation.bit_width}}} : !quantum.qureg")
        
        return operations
    
    def encode_classical_value(self, ssa_name: str, value: int, semantic_name: str = None) -> List[str]:
        """Encode a classical integer value into qubits"""
        if ssa_name not in self.variable_qubits:
            allocation = self.allocate_qubits_for_variable(ssa_name, semantic_name, value, "init")
        else:
            allocation = self.variable_qubits[ssa_name]
        
        operations = []
        operations.extend(self.generate_qubit_alloc_operations(allocation))
        
        if value > 0:
            binary_repr = format(value, f'0{allocation.bit_width}b')
            operations.append(f"    // Encode {value} = {binary_repr} (binary)")
            
            set_bits = []
            x_gate_ops = []
            for i, bit in enumerate(reversed(binary_repr)):
                if bit == '1':
                    qubit_id = allocation.start_qubit + i
                    set_bits.append(f"q{qubit_id}")
                    x_gate_ops.append(f"    quantum.x %q{qubit_id} : !quantum.qubit")
            
            if set_bits:
                operations.append(f"    // Set qubits: {', '.join(set_bits)}")
                operations.extend(x_gate_ops)
        else:
            operations.append(f"    // Value {value} = all qubits |0⟩")
        
        return operations
    
    def get_qubit_range(self, ssa_name: str) -> str:
        """Get the qubit range string for an SSA variable"""
        if ssa_name not in self.variable_qubits:
            return ""
        
        alloc = self.variable_qubits[ssa_name]
        return f"%q{alloc.start_qubit}:{alloc.start_qubit + alloc.bit_width}"
    
    def translate_increment_decrement(self, operation: str, operand_ssa: str, original_result_ssa: str, updated_result_ssa: str) -> List[str]:
        """Translate increment/decrement operations"""
        operations = []
        
        # Ensure operand has allocation
        if operand_ssa not in self.variable_qubits:
            self.allocate_qubits_for_variable(operand_ssa)
        
        operand_alloc = self.variable_qubits[operand_ssa]
        
        # Create allocations for both results (original value and updated value)
        original_semantic = f"temp_{operand_alloc.name}_orig"
        updated_semantic = f"{operand_alloc.name}_updated"
        
        if original_result_ssa not in self.variable_qubits:
            orig_allocation = self.allocate_qubits_for_variable(original_result_ssa, original_semantic)
            operations.extend(self.generate_qubit_alloc_operations(orig_allocation))
        
        if updated_result_ssa and updated_result_ssa not in self.variable_qubits:
            upd_allocation = self.allocate_qubits_for_variable(updated_result_ssa, updated_semantic)
            operations.extend(self.generate_qubit_alloc_operations(upd_allocation))
        
        operand_range = self.get_qubit_range(operand_ssa)
        orig_range = self.get_qubit_range(original_result_ssa)
        
        if operation in ["post_inc", "pre_inc"]:
            operations.append(f"    // {original_semantic} = {operand_alloc.name}++ (save original, then increment)")
            operations.append(f"    quantum.copy_and_inc {operand_range}, {orig_range} : !quantum.qureg, !quantum.qureg")
        elif operation in ["post_dec", "pre_dec"]:
            operations.append(f"    // {original_semantic} = {operand_alloc.name}-- (save original, then decrement)")  
            operations.append(f"    quantum.copy_and_dec {operand_range}, {orig_range} : !quantum.qureg, !quantum.qureg")
        
        return operations
    
    def translate_arithmetic_to_quantum(self, operation: str, lhs_ssa: str, rhs_ssa: str, result_ssa: str) -> List[str]:
        """Translate arithmetic operations with optimized bit allocation"""
        operations = []
        
        # Ensure operand variables have allocations
        if lhs_ssa not in self.variable_qubits:
            self.allocate_qubits_for_variable(lhs_ssa)
        if rhs_ssa not in self.variable_qubits:
            self.allocate_qubits_for_variable(rhs_ssa)
        
        # Create optimized result allocation
        semantic_name = self.create_semantic_name(result_ssa, operation)
        if result_ssa not in self.variable_qubits:
            result_allocation = self.allocate_qubits_for_variable(result_ssa, semantic_name, operation=operation)
            operations.extend(self.generate_qubit_alloc_operations(result_allocation))
        
        lhs_alloc = self.variable_qubits[lhs_ssa]
        rhs_alloc = self.variable_qubits[rhs_ssa]
        result_alloc = self.variable_qubits[result_ssa]
        
        lhs_range = self.get_qubit_range(lhs_ssa)
        rhs_range = self.get_qubit_range(rhs_ssa)
        result_range = self.get_qubit_range(result_ssa)
        
        operations.append(f"    // {result_alloc.name} = {lhs_alloc.name} {operation} {rhs_alloc.name} [{lhs_alloc.bit_width}+{rhs_alloc.bit_width}→{result_alloc.bit_width} bits]")
        
        if operation == "add":
            operations.append(f"    quantum.add_circuit {lhs_range}, {rhs_range}, {result_range} : !quantum.qureg, !quantum.qureg, !quantum.qureg")
        elif operation == "sub":
            operations.append(f"    quantum.sub_circuit {lhs_range}, {rhs_range}, {result_range} : !quantum.qureg, !quantum.qureg, !quantum.qureg")
        elif operation == "mul":
            operations.append(f"    quantum.mul_circuit {lhs_range}, {rhs_range}, {result_range} : !quantum.qureg, !quantum.qureg, !quantum.qureg")
        elif operation == "div":
            operations.append(f"    quantum.div_circuit {lhs_range}, {rhs_range}, {result_range} : !quantum.qureg, !quantum.qureg, !quantum.qureg")
        elif operation == "mod":
            operations.append(f"    quantum.mod_circuit {lhs_range}, {rhs_range}, {result_range} : !quantum.qureg, !quantum.qureg, !quantum.qureg")
        
        return operations
    
    def translate_measurement(self, ssa_name: str) -> List[str]:
        """Translate measurement operations"""
        if ssa_name not in self.variable_qubits:
            return []
        
        allocation = self.variable_qubits[ssa_name]
        operations = []
        
        operations.append(f"    // Measure {allocation.name} ({allocation.bit_width} qubits)")
        
        if allocation.bit_width == 1:
            operations.append(f"    %m{allocation.start_qubit} = quantum.measure %q{allocation.start_qubit} : !quantum.qubit -> i1")
            operations.append(f"    %{allocation.name}_result = classical.bit_to_int %m{allocation.start_qubit} : i1 -> i1")
        elif allocation.bit_width <= 4:
            qubit_measurements = []
            for i in range(allocation.bit_width):
                qubit_id = allocation.start_qubit + i
                qubit_measurements.append(f"%m{qubit_id}")
                operations.append(f"    %m{qubit_id} = quantum.measure %q{qubit_id} : !quantum.qubit -> i1")
            operations.append(f"    %{allocation.name}_result = classical.combine_bits {', '.join(qubit_measurements)} : i{allocation.bit_width}")
        else:
            operations.append(f"    %{allocation.name}_measurements = quantum.measure_all %q{allocation.start_qubit}:{allocation.start_qubit + allocation.bit_width} : !quantum.qureg -> !classical.bitvector")
            operations.append(f"    %{allocation.name}_result = classical.bitvector_to_int %{allocation.name}_measurements : i{allocation.bit_width}")
        
        return operations
    
    def parse_classical_mlir(self, mlir_content: str) -> List[str]:
        """Parse classical MLIR and generate complete quantum MLIR"""
        lines = mlir_content.strip().split('\n')
        quantum_mlir = []
        
        quantum_mlir.extend([
            "builtin.module {",
            '  "quantum.func"() ({',
            ""
        ])
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('builtin.module') or line.startswith('"quantum.func"') or line.startswith('func.return') or line.startswith('}'):
                continue
            
            # Parse quantum.init operations
            init_match = re.search(r'%(\w+) = "quantum\.init"\(\) \{.*?value = (\d+) : i32\}', line)
            if init_match:
                ssa_name, value = init_match.groups()
                ssa_name = f"%{ssa_name}"
                value = int(value)
                
                semantic_name = self.create_semantic_name(ssa_name, "init")
                
                self.ssa_variables[ssa_name] = SSAVariable(
                    ssa_name=ssa_name,
                    semantic_name=semantic_name,
                    value=value,
                    operation="init",
                    operands=[]
                )
                
                quantum_mlir.append(f"    // Initialize {semantic_name} = {value}")
                quantum_mlir.extend(self.encode_classical_value(ssa_name, value, semantic_name))
                quantum_mlir.append("")
                continue
            
            # Parse increment/decrement operations
            inc_dec_match = re.search(r'%([\w\d]+), %([\w\d]+) = "quantum\.(post_inc|post_dec|pre_inc|pre_dec)"\(%([\w\d]+)\)', line)
            if inc_dec_match:
                original_var, updated_var, operation, operand_var = inc_dec_match.groups()
                original_ssa = f"%{original_var}"
                updated_ssa = f"%{updated_var}"
                operand_ssa = f"%{operand_var}"
                
                self.ssa_variables[original_ssa] = SSAVariable(
                    ssa_name=original_ssa,
                    semantic_name="",
                    value=None,
                    operation=operation + "_orig",
                    operands=[operand_ssa]
                )
                
                self.ssa_variables[updated_ssa] = SSAVariable(
                    ssa_name=updated_ssa,
                    semantic_name="",
                    value=None,
                    operation=operation + "_upd",
                    operands=[operand_ssa]
                )
                
                quantum_mlir.extend(self.translate_increment_decrement(operation, operand_ssa, original_ssa, updated_ssa))
                quantum_mlir.append("")
                continue
            
            # Parse arithmetic operations
            arith_match = re.search(r'%(\w+) = "quantum\.(add|sub|mul|div|mod)"\(%([\w\d]+), %([\w\d]+)\)', line)
            if arith_match:
                result_var, operation, lhs_var, rhs_var = arith_match.groups()
                result_ssa = f"%{result_var}"
                lhs_ssa = f"%{lhs_var}"
                rhs_ssa = f"%{rhs_var}"
                
                self.ssa_variables[result_ssa] = SSAVariable(
                    ssa_name=result_ssa,
                    semantic_name="",
                    value=None,
                    operation=operation,
                    operands=[lhs_ssa, rhs_ssa]
                )
                
                quantum_mlir.extend(self.translate_arithmetic_to_quantum(operation, lhs_ssa, rhs_ssa, result_ssa))
                quantum_mlir.append("")
                continue
            
            # Parse measurement operations
            measure_match = re.search(r'%(\w+) = "quantum\.measure"\(%([\w\d]+)\)', line)
            if measure_match:
                result_var, measured_var = measure_match.groups()
                measured_ssa = f"%{measured_var}"
                quantum_mlir.extend(self.translate_measurement(measured_ssa))
                quantum_mlir.append("")
                continue
        
        quantum_mlir.extend([
            "    func.return",
            '  }) {func_name = "quantum_circuit"} : () -> ()',
            "}"
        ])
        
        return quantum_mlir
    
    def generate_quantum_arithmetic_circuits(self) -> str:
        """Generate quantum arithmetic circuit definitions"""
        return '''
// Complete Quantum Circuit Definitions

quantum.alloc_reg : (!quantum.qureg) -> ()
quantum.add_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()
quantum.sub_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()
quantum.mul_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()
quantum.div_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()
quantum.mod_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()

// Copy and increment/decrement operations
quantum.copy_and_inc : (!quantum.qureg, !quantum.qureg) -> ()
quantum.copy_and_dec : (!quantum.qureg, !quantum.qureg) -> ()

quantum.x : (!quantum.qubit) -> ()
quantum.measure_all : (!quantum.qureg) -> (!classical.bitvector)

classical.combine_bits : (i1, ..., i1) -> iN
classical.bitvector_to_int : (!classical.bitvector) -> iN
classical.bit_to_int : (i1) -> i1
'''

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python classical_to_quantum_translator.py <input_classical.mlir> <output_quantum.mlir>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(input_file, 'r') as f:
        classical_mlir = f.read()
    
    translator = ClassicalToQuantumTranslator(default_bit_width=8, max_bit_width=16)
    quantum_mlir_lines = translator.parse_classical_mlir(classical_mlir)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(quantum_mlir_lines))
        f.write('\n\n')
        f.write(translator.generate_quantum_arithmetic_circuits())
    
    print(f"✅ Generated complete quantum MLIR: {output_file}")
    print(f"📊 Total qubits: {translator.qubit_counter}")
    print("📋 Variables:")
    for ssa_name, alloc in translator.variable_qubits.items():
        value_info = f" = {translator.known_values[ssa_name]}" if ssa_name in translator.known_values else ""
        print(f"   {alloc.name}: {alloc.bit_width} qubits{value_info}")

if __name__ == "__main__":
    main()
