# classical_to_quantum_translator.py
"""
Translates classical MLIR operations to quantum circuit operations.
This is the second stage of compilation: Classical MLIR → Quantum MLIR
FIXED VERSION - Proper SSA variable tracking and semantic mapping
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class QubitAllocation:
    """Track qubit allocations and their bit widths"""
    name: str
    bit_width: int
    start_qubit: int
    ssa_name: str  # Track the SSA variable name (%0, %1, etc.)

@dataclass 
class SSAVariable:
    """Track SSA variable information"""
    ssa_name: str        # %0, %1, %2, etc.
    semantic_name: str   # a, b, c, temp_c, etc.
    value: Optional[int] # Known constant value if any
    operation: str       # What operation created this variable
    operands: List[str]  # SSA names of operands

class ClassicalToQuantumTranslator:
    def __init__(self, bit_width: int = 32):
        self.bit_width = bit_width
        self.qubit_counter = 0
        self.variable_qubits: Dict[str, QubitAllocation] = {}  # Maps SSA names to allocations
        self.ssa_variables: Dict[str, SSAVariable] = {}       # Maps SSA names to variable info
        self.semantic_to_ssa: Dict[str, str] = {}             # Maps semantic names to SSA names
        self.quantum_operations = []
        
        # Counter for creating semantic names for unnamed variables
        self.temp_counter = 0
        
    def create_semantic_name(self, ssa_name: str, operation: str = "") -> str:
        """Create a meaningful semantic name for SSA variables"""
        if operation == "init":
            # Map %0, %1, %2 to a, b, c for the first few variables
            var_mapping = {"%0": "a", "%1": "b", "%2": "c", "%3": "d", "%4": "e"}
            if ssa_name in var_mapping:
                return var_mapping[ssa_name]
        
        elif operation in ["post_inc", "post_dec"]:
            # Create temp names for increment/decrement results
            self.temp_counter += 1
            prefix = "temp_" if operation.startswith("post") else "pre_"
            return f"{prefix}{self.temp_counter}"
            
        elif operation in ["add", "sub", "mul", "div"]:
            # Create meaningful names for arithmetic results
            if operation == "add":
                return f"sum_{self.temp_counter}"
            elif operation == "sub":
                return f"diff_{self.temp_counter}"
            else:
                return f"result_{self.temp_counter}"
        
        # Default: create a generic temp name
        self.temp_counter += 1
        return f"temp_{self.temp_counter}"
    
    def allocate_qubits_for_variable(self, ssa_name: str, semantic_name: str = None) -> QubitAllocation:
        """Allocate qubits to represent a classical variable"""
        if semantic_name is None:
            semantic_name = self.create_semantic_name(ssa_name)
            
        allocation = QubitAllocation(
            name=semantic_name,
            bit_width=self.bit_width,
            start_qubit=self.qubit_counter,
            ssa_name=ssa_name
        )
        self.qubit_counter += self.bit_width
        self.variable_qubits[ssa_name] = allocation
        self.semantic_to_ssa[semantic_name] = ssa_name
        
        return allocation
    
    def generate_qubit_alloc_operations(self, allocation: QubitAllocation) -> List[str]:
        """Generate clean qubit allocation operations"""
        operations = []
        
        # Use quantum register allocation for cleaner output
        if self.bit_width <= 8:
            # For small bit widths, show individual allocations
            for i in range(self.bit_width):
                operations.append(f"    %q{allocation.start_qubit + i} = quantum.alloc : !quantum.qubit")
        else:
            # For larger bit widths, use quantum register allocation
            operations.append(f"    // Allocate {self.bit_width} qubits for {allocation.name}")
            operations.append(f"    %qreg_{allocation.name} = quantum.alloc_reg {{size = {self.bit_width}}} : !quantum.qureg")
            
            # Still generate individual qubit references for compatibility
            operations.append(f"    // Individual qubit references: %q{allocation.start_qubit} to %q{allocation.start_qubit + self.bit_width - 1}")
        
        return operations
    
    def encode_classical_value(self, ssa_name: str, value: int, semantic_name: str = None) -> List[str]:
        """Encode a classical integer value into qubits using binary representation"""
        if ssa_name not in self.variable_qubits:
            allocation = self.allocate_qubits_for_variable(ssa_name, semantic_name)
        else:
            allocation = self.variable_qubits[ssa_name]
        
        operations = []
        
        # Add qubit allocation operations (cleaner version)
        operations.extend(self.generate_qubit_alloc_operations(allocation))
        
        # Convert value to binary and encode using X gates
        if value > 0:
            operations.append(f"    // Encode value {value} in binary: {format(value, f'0{min(8, self.bit_width)}b')}...")
            binary_repr = format(value, f'0{self.bit_width}b')
            
            # Only show the X gates for set bits
            set_bits = []
            for i, bit in enumerate(reversed(binary_repr)):  # LSB first
                if bit == '1':
                    qubit_id = allocation.start_qubit + i
                    set_bits.append(f"q{qubit_id}")
                    operations.append(f"    quantum.x %q{qubit_id} : !quantum.qubit")
            
            if set_bits:
                operations.insert(-len(set_bits), f"    // Set bits: {', '.join(set_bits)}")
        
        return operations
    
    def get_qubit_range(self, ssa_name: str) -> str:
        """Get the qubit range string for an SSA variable"""
        if ssa_name not in self.variable_qubits:
            return ""
        
        alloc = self.variable_qubits[ssa_name]
        return f"%q{alloc.start_qubit}:{alloc.start_qubit + alloc.bit_width}"
    
    def translate_measurement(self, ssa_name: str) -> List[str]:
        """Translate measurement to quantum measurement operations"""
        if ssa_name not in self.variable_qubits:
            return []
        
        allocation = self.variable_qubits[ssa_name]
        operations = []
        
        operations.append(f"    // Measure {allocation.name} ({ssa_name}) - {allocation.bit_width} qubits")
        
        if allocation.bit_width <= 4:
            # For small variables, show individual measurements
            for i in range(allocation.bit_width):
                qubit_id = allocation.start_qubit + i
                operations.append(f"    %m{qubit_id} = quantum.measure %q{qubit_id} : !quantum.qubit -> i1")
        else:
            # For larger variables, use bulk measurement
            operations.append(f"    %{allocation.name}_measurements = quantum.measure_all %q{allocation.start_qubit}:{allocation.start_qubit + allocation.bit_width} : !quantum.qureg -> !classical.bitvector")
        
        # Combine measurements into classical result
        operations.append(f"    %{allocation.name}_classical = classical.combine_bits %m{allocation.start_qubit}:{allocation.start_qubit + allocation.bit_width} : i{allocation.bit_width}")
        
        return operations
    def translate_arithmetic_to_quantum(self, operation: str, lhs_ssa: str, rhs_ssa: str, result_ssa: str) -> List[str]:
        """Translate arithmetic operations to quantum arithmetic circuits"""
        operations = []
        
        # Ensure operand variables have qubit allocations
        if lhs_ssa not in self.variable_qubits:
            self.allocate_qubits_for_variable(lhs_ssa)
        if rhs_ssa not in self.variable_qubits:
            self.allocate_qubits_for_variable(rhs_ssa)
        
        # Create result variable allocation
        semantic_name = self.create_semantic_name(result_ssa, operation)
        if result_ssa not in self.variable_qubits:
            result_allocation = self.allocate_qubits_for_variable(result_ssa, semantic_name)
            # Add qubit allocations for result (cleaner version)
            operations.extend(self.generate_qubit_alloc_operations(result_allocation))
        
        lhs_alloc = self.variable_qubits[lhs_ssa]
        rhs_alloc = self.variable_qubits[rhs_ssa]
        result_alloc = self.variable_qubits[result_ssa]
        
        # Get qubit ranges
        lhs_range = self.get_qubit_range(lhs_ssa)
        rhs_range = self.get_qubit_range(rhs_ssa)
        result_range = self.get_qubit_range(result_ssa)
        
        # Create operation comment with semantic names
        operations.append(f"    // Quantum {operation}: {result_alloc.name} = {lhs_alloc.name} {operation} {rhs_alloc.name}")
        
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
    
    def translate_unary_to_quantum(self, operation: str, operand_ssa: str, result_ssa: str) -> List[str]:
        """Translate unary operations to quantum circuits"""
        operations = []
        
        # Ensure operand has qubit allocation
        if operand_ssa not in self.variable_qubits:
            self.allocate_qubits_for_variable(operand_ssa)
        
        operand_alloc = self.variable_qubits[operand_ssa]
        
        # Create result allocation
        semantic_name = self.create_semantic_name(result_ssa, operation)
        if result_ssa not in self.variable_qubits:
            result_allocation = self.allocate_qubits_for_variable(result_ssa, semantic_name)
            # Add qubit allocations for result (cleaner version)
            operations.extend(self.generate_qubit_alloc_operations(result_allocation))
        
        result_alloc = self.variable_qubits[result_ssa]
        operand_range = self.get_qubit_range(operand_ssa)
        result_range = self.get_qubit_range(result_ssa)
        
        if operation == "neg":
            operations.append(f"    // Quantum negation: {result_alloc.name} = -{operand_alloc.name}")
            operations.append(f"    quantum.neg_circuit {operand_range}, {result_range} : !quantum.qureg, !quantum.qureg")
        elif operation == "not":
            operations.append(f"    // Quantum bitwise NOT: {result_alloc.name} = ~{operand_alloc.name}")
            operations.append(f"    quantum.not_circuit {operand_range}, {result_range} : !quantum.qureg, !quantum.qureg")
        
        return operations
    
    def translate_increment_decrement(self, operation: str, operand_ssa: str, original_result_ssa: str, updated_result_ssa: str) -> List[str]:
        """Translate increment/decrement with proper result handling"""
        operations = []
        
        # Ensure operand has allocation
        if operand_ssa not in self.variable_qubits:
            self.allocate_qubits_for_variable(operand_ssa)
        
        operand_alloc = self.variable_qubits[operand_ssa]
        
        # Create allocations for both results
        if operation.startswith("post"):
            # Post-increment/decrement: original_result gets the original value
            original_semantic = f"temp_{operand_alloc.name}_orig"
            updated_semantic = f"{operand_alloc.name}_updated"
        else:
            # Pre-increment/decrement: result is the same
            original_semantic = f"{operand_alloc.name}_inc" if "inc" in operation else f"{operand_alloc.name}_dec"
            updated_semantic = original_semantic
        
        # Allocate qubits for results (cleaner version)
        if original_result_ssa not in self.variable_qubits:
            orig_allocation = self.allocate_qubits_for_variable(original_result_ssa, original_semantic)
            operations.extend(self.generate_qubit_alloc_operations(orig_allocation))
        
        if updated_result_ssa and updated_result_ssa not in self.variable_qubits:
            upd_allocation = self.allocate_qubits_for_variable(updated_result_ssa, updated_semantic)
            operations.extend(self.generate_qubit_alloc_operations(upd_allocation))
        
        operand_range = self.get_qubit_range(operand_ssa)
        
        if operation in ["post_inc", "pre_inc"]:
            operations.append(f"    // Quantum increment: {operand_alloc.name}++ (post-increment semantics)")
            operations.append(f"    quantum.inc_circuit {operand_range} : !quantum.qureg")
        elif operation in ["post_dec", "pre_dec"]:
            operations.append(f"    // Quantum decrement: {operand_alloc.name}-- (post-decrement semantics)")
            operations.append(f"    quantum.dec_circuit {operand_range} : !quantum.qureg")
        
        return operations
    
    def translate_comparison_to_quantum(self, operation: str, lhs_ssa: str, rhs_ssa: str, result_ssa: str) -> List[str]:
        """Translate comparison operations to quantum circuits"""
        operations = []
        
        # Ensure input variables have qubit allocations
        if lhs_ssa not in self.variable_qubits:
            self.allocate_qubits_for_variable(lhs_ssa)
        if rhs_ssa not in self.variable_qubits:
            self.allocate_qubits_for_variable(rhs_ssa)
        
        # Result is a single bit (boolean)
        if result_ssa not in self.variable_qubits:
            semantic_name = f"cmp_result_{self.temp_counter}"
            self.temp_counter += 1
            allocation = QubitAllocation(
                name=semantic_name,
                bit_width=1,
                start_qubit=self.qubit_counter,
                ssa_name=result_ssa
            )
            self.qubit_counter += 1
            self.variable_qubits[result_ssa] = allocation
            operations.append(f"    // Allocate 1 qubit for boolean result {semantic_name}")
            operations.append(f"    %q{allocation.start_qubit} = quantum.alloc : !quantum.qubit")
        
        lhs_alloc = self.variable_qubits[lhs_ssa]
        rhs_alloc = self.variable_qubits[rhs_ssa]
        result_alloc = self.variable_qubits[result_ssa]
        
        lhs_range = self.get_qubit_range(lhs_ssa)
        rhs_range = self.get_qubit_range(rhs_ssa)
        result_qubit = f"%q{result_alloc.start_qubit}"
        
        # Map operations to circuit names
        op_map = {
            "lt": "quantum.lt_circuit",
            "gt": "quantum.gt_circuit", 
            "eq": "quantum.eq_circuit",
            "ne": "quantum.ne_circuit",
            "le": "quantum.le_circuit",
            "ge": "quantum.ge_circuit"
        }
        
        if operation in op_map:
            operations.append(f"    // Quantum {operation}: {result_alloc.name} = {lhs_alloc.name} {operation} {rhs_alloc.name}")
            operations.append(f"    {op_map[operation]} {lhs_range}, {rhs_range}, {result_qubit} : !quantum.qureg, !quantum.qureg, !quantum.qubit")
        
        return operations

    def parse_classical_mlir(self, mlir_content: str) -> List[str]:
        """Parse classical MLIR and generate quantum MLIR with proper SSA tracking"""
        lines = mlir_content.strip().split('\n')
        quantum_mlir = []
        
        # Add quantum module header
        quantum_mlir.extend([
            "builtin.module {",
            '  "quantum.func"() ({',
            ""
        ])
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('builtin.module') or line.startswith('"quantum.func"') or line.startswith('func.return') or line.startswith('}'):
                continue
            
            # Parse quantum.init operations (constant assignments)
            init_match = re.search(r'%(\w+) = "quantum\.init"\(\) \{.*?value = (\d+) : i32\}', line)
            if init_match:
                ssa_name, value = init_match.groups()
                ssa_name = f"%{ssa_name}"
                value = int(value)
                
                # Create semantic name for this variable
                semantic_name = self.create_semantic_name(ssa_name, "init")
                
                # Store SSA variable info
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
            
            # Parse arithmetic operations (add, sub, mul, div, mod)
            arith_match = re.search(r'%(\w+) = "quantum\.(add|sub|mul|div|mod)"\(%([\w\d]+), %([\w\d]+)\)', line)
            if arith_match:
                result_var, operation, lhs_var, rhs_var = arith_match.groups()
                result_ssa = f"%{result_var}"
                lhs_ssa = f"%{lhs_var}"
                rhs_ssa = f"%{rhs_var}"
                
                # Store SSA variable info
                self.ssa_variables[result_ssa] = SSAVariable(
                    ssa_name=result_ssa,
                    semantic_name="",  # Will be set in translation
                    value=None,
                    operation=operation,
                    operands=[lhs_ssa, rhs_ssa]
                )
                
                quantum_mlir.extend(self.translate_arithmetic_to_quantum(operation, lhs_ssa, rhs_ssa, result_ssa))
                quantum_mlir.append("")
                continue
            
            # Parse increment/decrement operations
            inc_dec_match = re.search(r'%([\w\d]+), %([\w\d]+) = "quantum\.(post_inc|post_dec|pre_inc|pre_dec)"\(%([\w\d]+)\)', line)
            if inc_dec_match:
                original_var, updated_var, operation, operand_var = inc_dec_match.groups()
                original_ssa = f"%{original_var}"
                updated_ssa = f"%{updated_var}"
                operand_ssa = f"%{operand_var}"
                
                # Store SSA variable info for both results
                self.ssa_variables[original_ssa] = SSAVariable(
                    ssa_name=original_ssa,
                    semantic_name="",  # Will be set in translation
                    value=None,
                    operation=operation + "_orig",
                    operands=[operand_ssa]
                )
                
                self.ssa_variables[updated_ssa] = SSAVariable(
                    ssa_name=updated_ssa,
                    semantic_name="",  # Will be set in translation
                    value=None,
                    operation=operation + "_upd",
                    operands=[operand_ssa]
                )
                
                quantum_mlir.extend(self.translate_increment_decrement(operation, operand_ssa, original_ssa, updated_ssa))
                quantum_mlir.append("")
                continue
            
            # Parse unary operations (neg, not)
            unary_match = re.search(r'%(\w+) = "quantum\.(neg|not)"\(%([\w\d]+)\)', line)
            if unary_match:
                result_var, operation, operand_var = unary_match.groups()
                result_ssa = f"%{result_var}"
                operand_ssa = f"%{operand_var}"
                
                self.ssa_variables[result_ssa] = SSAVariable(
                    ssa_name=result_ssa,
                    semantic_name="",  # Will be set in translation
                    value=None,
                    operation=operation,
                    operands=[operand_ssa]
                )
                
                quantum_mlir.extend(self.translate_unary_to_quantum(operation, operand_ssa, result_ssa))
                quantum_mlir.append("")
                continue
            
            # Parse comparison operations (lt, gt, eq, ne, le, ge)
            comp_match = re.search(r'%(\w+) = "quantum\.(lt|gt|eq|ne|le|ge)"\(%([\w\d]+), %([\w\d]+)\)', line)
            if comp_match:
                result_var, operation, lhs_var, rhs_var = comp_match.groups()
                result_ssa = f"%{result_var}"
                lhs_ssa = f"%{lhs_var}"
                rhs_ssa = f"%{rhs_var}"
                
                self.ssa_variables[result_ssa] = SSAVariable(
                    ssa_name=result_ssa,
                    semantic_name="",  # Will be set in translation
                    value=None,
                    operation=operation,
                    operands=[lhs_ssa, rhs_ssa]
                )
                
                quantum_mlir.extend(self.translate_comparison_to_quantum(operation, lhs_ssa, rhs_ssa, result_ssa))
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
        
        # Add function footer
        quantum_mlir.extend([
            "    func.return",
            '  }) {func_name = "quantum_circuit"} : () -> ()',
            "}"
        ])
        
        return quantum_mlir
    
    def generate_quantum_arithmetic_circuits(self) -> str:
        """Generate quantum arithmetic circuit definitions"""
        return '''
// Quantum Arithmetic Circuit Definitions
// These would be implemented as composite operations or lowered to basic gates

// Arithmetic operations
quantum.add_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()
quantum.sub_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()
quantum.mul_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()
quantum.div_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()
quantum.mod_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qureg) -> ()

// Increment/Decrement operations
quantum.inc_circuit : (!quantum.qureg) -> ()
quantum.dec_circuit : (!quantum.qureg) -> ()

// Unary operations
quantum.neg_circuit : (!quantum.qureg, !quantum.qureg) -> ()
quantum.not_circuit : (!quantum.qureg, !quantum.qureg) -> ()

// Comparison operations  
quantum.lt_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qubit) -> ()
quantum.gt_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qubit) -> ()
quantum.eq_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qubit) -> ()
quantum.ne_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qubit) -> ()
quantum.le_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qubit) -> ()
quantum.ge_circuit : (!quantum.qureg, !quantum.qureg, !quantum.qubit) -> ()

// Basic quantum gates
quantum.x : (!quantum.qubit) -> ()
quantum.cx : (!quantum.qubit, !quantum.qubit) -> ()
quantum.ccx : (!quantum.qubit, !quantum.qubit, !quantum.qubit) -> ()
quantum.h : (!quantum.qubit) -> ()

// Quantum register operations
quantum.alloc_reg : (!quantum.qureg) -> ()
quantum.measure_all : (!quantum.qureg) -> (!classical.bitvector)

// Classical utility operations
classical.combine_bits : (i1, ..., i1) -> i32
'''

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python classical_to_quantum_translator.py <input_classical.mlir> <output_quantum.mlir>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read classical MLIR
    with open(input_file, 'r') as f:
        classical_mlir = f.read()
    
    # Translate to quantum MLIR
    translator = ClassicalToQuantumTranslator(bit_width=32)
    quantum_mlir_lines = translator.parse_classical_mlir(classical_mlir)
    
    # Write quantum MLIR
    with open(output_file, 'w') as f:
        f.write('\n'.join(quantum_mlir_lines))
        f.write('\n\n')
        f.write(translator.generate_quantum_arithmetic_circuits())
    
    print(f"Translated classical MLIR to quantum MLIR: {input_file} → {output_file}")
    print(f"Allocated {translator.qubit_counter} qubits total")
    print("\nVariable mappings:")
    for ssa_name, alloc in translator.variable_qubits.items():
        qubit_range = f"q{alloc.start_qubit}-q{alloc.start_qubit + alloc.bit_width - 1}"
        print(f"  {ssa_name} ({alloc.name}): {qubit_range}")
    
    print("\nSSA Variable tracking:")
    for ssa_name, var_info in translator.ssa_variables.items():
        print(f"  {ssa_name} → {var_info.semantic_name} (op: {var_info.operation}, value: {var_info.value})")

if __name__ == "__main__":
    main()
