#!/usr/bin/env python3
"""
Comprehensive Gate-Level MLIR Converter

This script converts high-level quantum MLIR to gate-level MLIR while preserving
ALL operations including div, mod, and, or, xor, not, negation, etc.

Usage: python comprehensive_gate_converter.py <input.mlir> <output.mlir> [--debug]
"""

import re
import sys
import argparse
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass

@dataclass
class SSAVariable:
    """Track SSA variable information"""
    ssa_name: str
    operation: str
    value: Optional[int]
    dependencies: List[str]
    operands: List[str]
    attributes: Dict[str, str]

@dataclass
class QubitRegister:
    """Track qubit register allocations"""
    mlir_name: str
    qiskit_name: str
    size: int
    initial_value: Optional[int] = None

class ComprehensiveGateConverter:
    def __init__(self, enable_debug=False):
        self.enable_debug = enable_debug
        self.ssa_variables: Dict[str, SSAVariable] = {}
        self.variable_to_qubits: Dict[str, QubitRegister] = {}
        self.qubit_counter = 0
        self.processed_multi_ops: Set[str] = set()
        self.supported_operations = {
            # Arithmetic operations
            'add', 'sub', 'mul', 'div', 'mod',
            # Logical operations  
            'and', 'or', 'xor', 'not',
            # Unary operations
            'neg', 'pre_inc', 'pre_dec',
            # Multi-result operations
            'post_inc', 'post_dec',
            # Initialization and measurement
            'init', 'measure'
        }
        
    def debug_print(self, message: str):
        if self.enable_debug:
            print(f"[DEBUG] {message}")
    
    def parse_high_level_mlir(self, mlir_content: str):
        """Parse high-level MLIR and extract ALL SSA variables and operations"""
        self.debug_print("=== Parsing High-Level MLIR ===")
        
        lines = mlir_content.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('builtin') or line.startswith('"quantum.func"') or line.startswith('func.return') or line == '}':
                continue
                
            self.debug_print(f"Line {i:2}: {line}")
            self._parse_mlir_line(line, i)
    
    def _parse_mlir_line(self, line: str, line_num: int):
        """Parse a single MLIR line and extract operation information"""
        
        # Pattern 1: quantum.init operations
        # %0 = "quantum.init"() {type = i32, value = 1 : i32} : () -> i32
        init_match = re.search(r'%(\d+)\s*=\s*"quantum\.init"\(\).*?value\s*=\s*(\d+)', line)
        if init_match:
            var_id, value = init_match.groups()
            ssa_name = f"%{var_id}"
            self.ssa_variables[ssa_name] = SSAVariable(
                ssa_name=ssa_name, 
                operation="init", 
                value=int(value), 
                dependencies=[], 
                operands=[],
                attributes={"value": value}
            )
            self.debug_print(f"  → Found INIT: {ssa_name} = {value}")
            return
        
        # Pattern 2: Multi-result operations (post_inc, post_dec)
        # %3, %4 = "quantum.post_inc"(%2) : (i32) -> (i32, i32)
        multi_result_match = re.search(r'%(\d+),\s*%(\d+)\s*=\s*"quantum\.(\w+)"\s*\(\s*%(\d+)\s*\)', line)
        if multi_result_match:
            result1_id, result2_id, operation, operand_id = multi_result_match.groups()
            result1_ssa = f"%{result1_id}"
            result2_ssa = f"%{result2_id}"
            operand_ssa = f"%{operand_id}"
            
            # Create entries for both results
            self.ssa_variables[result1_ssa] = SSAVariable(
                ssa_name=result1_ssa, 
                operation=f"{operation}_orig", 
                value=None, 
                dependencies=[operand_ssa],
                operands=[operand_ssa],
                attributes={"multi_result": "first"}
            )
            self.ssa_variables[result2_ssa] = SSAVariable(
                ssa_name=result2_ssa, 
                operation=f"{operation}_new", 
                value=None, 
                dependencies=[operand_ssa],
                operands=[operand_ssa],
                attributes={"multi_result": "second"}
            )
            self.debug_print(f"  → Found MULTI-RESULT: {result1_ssa}, {result2_ssa} = {operation}({operand_ssa})")
            return
        
        # Pattern 3: Binary operations (add, sub, mul, div, mod, and, or, xor)
        # %7 = "quantum.add"(%5, %1) : (i32, i32) -> i32
        # %8 = "quantum.div"(%6, %2) : (i32, i32) -> i32
        binary_match = re.search(r'%(\d+)\s*=\s*"quantum\.(\w+)"\s*\(\s*%(\d+)\s*,\s*%(\d+)\s*\)', line)
        if binary_match:
            result_var, operation, operand1, operand2 = binary_match.groups()
            result_ssa = f"%{result_var}"
            operand1_ssa = f"%{operand1}"
            operand2_ssa = f"%{operand2}"
            
            self.ssa_variables[result_ssa] = SSAVariable(
                ssa_name=result_ssa, 
                operation=operation, 
                value=None, 
                dependencies=[operand1_ssa, operand2_ssa],
                operands=[operand1_ssa, operand2_ssa],
                attributes={"operation_type": "binary"}
            )
            self.debug_print(f"  → Found BINARY: {result_ssa} = {operation}({operand1_ssa}, {operand2_ssa})")
            return
        
        # Pattern 4: Unary operations (not, neg, pre_inc, pre_dec)  
        # %5 = "quantum.not"(%4) : (i32) -> i32
        # %6 = "quantum.neg"(%3) : (i32) -> i32
        unary_match = re.search(r'%(\d+)\s*=\s*"quantum\.(\w+)"\s*\(\s*%(\d+)\s*\)', line)
        if unary_match:
            result_var, operation, operand = unary_match.groups()
            result_ssa = f"%{result_var}"
            operand_ssa = f"%{operand}"
            
            # Only process if it's actually a unary operation and not a measurement
            if operation in ['not', 'neg', 'pre_inc', 'pre_dec'] or operation not in ['measure']:
                self.ssa_variables[result_ssa] = SSAVariable(
                    ssa_name=result_ssa, 
                    operation=operation, 
                    value=None, 
                    dependencies=[operand_ssa],
                    operands=[operand_ssa],
                    attributes={"operation_type": "unary"}
                )
                self.debug_print(f"  → Found UNARY: {result_ssa} = {operation}({operand_ssa})")
                return
        
        # Pattern 5: quantum.measure operations
        # %9 = "quantum.measure"(%8) : (i32) -> i1
        measure_match = re.search(r'%(\d+)\s*=\s*"quantum\.measure"\s*\(\s*%(\d+)\s*\)', line)
        if measure_match:
            result_var, measured_var = measure_match.groups()
            result_ssa = f"%{result_var}"
            measured_ssa = f"%{measured_var}"
            
            self.ssa_variables[result_ssa] = SSAVariable(
                ssa_name=result_ssa, 
                operation="measure", 
                value=None, 
                dependencies=[measured_ssa],
                operands=[measured_ssa],
                attributes={"operation_type": "measurement"}
            )
            self.debug_print(f"  → Found MEASURE: {result_ssa} = measure({measured_ssa})")
            return
                
        self.debug_print(f"  → UNMATCHED: {line}")
    
    def allocate_qubit_register(self, ssa_name: str, size: int = 4) -> QubitRegister:
        """Allocate a unique qubit register"""
        if ssa_name in self.variable_to_qubits:
            return self.variable_to_qubits[ssa_name]
        
        # Keep the % prefix for MLIR SSA format
        qiskit_name = f"%q{self.qubit_counter}"
        register = QubitRegister(
            mlir_name=ssa_name,
            qiskit_name=qiskit_name,
            size=size
        )
        
        self.variable_to_qubits[ssa_name] = register
        self.qubit_counter += 1
        self.debug_print(f"Allocated {qiskit_name} for {ssa_name} (size: {size})")
        return register
    
    def convert_to_gate_level(self) -> List[str]:
        """Convert ALL SSA variables to gate-level operations"""
        self.debug_print("\n=== Converting to Gate-Level ===")
        
        gate_ops = []
        processed = set()
        
        # Show summary of what we found
        self.debug_print(f"Total variables to process: {len(self.ssa_variables)}")
        operation_counts = {}
        for var in self.ssa_variables.values():
            base_op = var.operation.replace("_orig", "").replace("_new", "")
            operation_counts[base_op] = operation_counts.get(base_op, 0) + 1
        
        self.debug_print("Operation distribution:")
        for op, count in sorted(operation_counts.items()):
            self.debug_print(f"  {op}: {count}")
        
        def process_variable(ssa_name: str):
            if ssa_name in processed or ssa_name not in self.ssa_variables:
                return
            
            var_info = self.ssa_variables[ssa_name]
            self.debug_print(f"Processing {ssa_name} ({var_info.operation})")
            
            # Process dependencies first
            for dep in var_info.dependencies:
                process_variable(dep)
            
            if var_info.operation == "init":
                # Allocate qubits and initialize
                register = self.allocate_qubit_register(ssa_name)
                gate_ops.append(f"    {register.qiskit_name} = q.alloc : !qreg<{register.size}>")
                gate_ops.append(f"    q.init {register.qiskit_name}, {var_info.value} : i32")
                self.debug_print(f"  → Generated: alloc + init {register.qiskit_name} = {var_info.value}")
                
            elif var_info.operation == "measure":
                # Create measurement operation
                measured_var = var_info.dependencies[0]
                if measured_var in self.variable_to_qubits:
                    measured_reg = self.variable_to_qubits[measured_var]
                    result_reg = self.allocate_qubit_register(ssa_name)
                    gate_ops.append(f"    {result_reg.qiskit_name} = q.measure {measured_reg.qiskit_name} : !qreg -> i32")
                    self.debug_print(f"  → Generated: {result_reg.qiskit_name} = measure({measured_reg.qiskit_name})")
                else:
                    self.debug_print(f"  → ERROR: Cannot find qubit for measured variable {measured_var}")
                    
            # elif var_info.operation in ["add", "sub", "mul", "div", "mod", "and", "or", "xor"]:
            #     # Create arithmetic/logical circuit
            #     if len(var_info.operands) >= 2:
            #         operand1, operand2 = var_info.operands[:2]
            #         if operand1 in self.variable_to_qubits and operand2 in self.variable_to_qubits:
            #             reg1 = self.variable_to_qubits[operand1]
            #             reg2 = self.variable_to_qubits[operand2]
            #             result_reg = self.allocate_qubit_register(ssa_name)
                        
            #             gate_ops.append(f"    {result_reg.qiskit_name} = q.alloc : !qreg<{result_reg.size}>")
            #             gate_ops.append(f"    q.{var_info.operation}_circuit {reg1.qiskit_name}, {reg2.qiskit_name}, {result_reg.qiskit_name}")
            #             self.debug_print(f"  → Generated: {var_info.operation}_circuit({reg1.qiskit_name}, {reg2.qiskit_name}) -> {result_reg.qiskit_name}")
            #         else:
            #             self.debug_print(f"  → ERROR: Missing dependencies for {var_info.operation}: {var_info.operands}")
            #             # Create fallback registers
            #             for operand in var_info.operands:
            #                 if operand not in self.variable_to_qubits:
            #                     fallback_reg = self.allocate_qubit_register(operand)
            #                     gate_ops.append(f"    {fallback_reg.qiskit_name} = q.alloc : !qreg<{fallback_reg.size}>")
            #                     gate_ops.append(f"    q.init {fallback_reg.qiskit_name}, 0 : i32  // Fallback for missing operand")
                        
            #             # Now try again
            #             reg1 = self.variable_to_qubits[var_info.operands[0]]
            #             reg2 = self.variable_to_qubits[var_info.operands[1]]
            #             result_reg = self.allocate_qubit_register(ssa_name)
            #             gate_ops.append(f"    {result_reg.qiskit_name} = q.alloc : !qreg<{result_reg.size}>")
            #             gate_ops.append(f"    q.{var_info.operation}_circuit {reg1.qiskit_name}, {reg2.qiskit_name}, {result_reg.qiskit_name}")
            #             self.debug_print(f"  → Generated with fallback: {var_info.operation}_circuit({reg1.qiskit_name}, {reg2.qiskit_name}) -> {result_reg.qiskit_name}")
            elif var_info.operation in ["add", "sub", "mul", "div", "mod", "and", "or", "xor"]:
                # Create arithmetic/logical circuit
                if len(var_info.operands) >= 2:
                    operand1, operand2 = var_info.operands[:2]
                    if operand1 in self.variable_to_qubits and operand2 in self.variable_to_qubits:
                        reg1 = self.variable_to_qubits[operand1]
                        reg2 = self.variable_to_qubits[operand2]
                        result_reg = self.allocate_qubit_register(ssa_name)

                        # always allocate the result register
                        gate_ops.append(f"    {result_reg.qiskit_name} = q.alloc : !qreg<{result_reg.size}>")

                        # All binary operations use 3 operands consistently
                        gate_ops.append(
                            f"    q.{var_info.operation}_circuit "
                            f"{reg1.qiskit_name}, {reg2.qiskit_name}, {result_reg.qiskit_name}"
                        )
                        self.debug_print(
                            f"  → Generated: {var_info.operation}_circuit("
                            f"{reg1.qiskit_name}, {reg2.qiskit_name}) -> {result_reg.qiskit_name}"
                        )
                    else:
                        # fallback if dependencies missing
                        self.debug_print(f"  → ERROR: Missing dependencies for {var_info.operation}: {var_info.operands}")
                        for operand in var_info.operands:
                            if operand not in self.variable_to_qubits:
                                fallback_reg = self.allocate_qubit_register(operand)
                                gate_ops.append(f"    {fallback_reg.qiskit_name} = q.alloc : !qreg<{fallback_reg.size}>")
                                gate_ops.append(f"    q.init {fallback_reg.qiskit_name}, 0 : i32  // Fallback for missing operand")

                        # re-fetch regs now that we've created them
                        reg1 = self.variable_to_qubits[var_info.operands[0]]
                        reg2 = self.variable_to_qubits[var_info.operands[1]]
                        result_reg = self.allocate_qubit_register(ssa_name)
                        gate_ops.append(f"    {result_reg.qiskit_name} = q.alloc : !qreg<{result_reg.size}>")
                        gate_ops.append(
                            f"    q.{var_info.operation}_circuit "
                            f"{reg1.qiskit_name}, {reg2.qiskit_name}, {result_reg.qiskit_name}"
                        )
                        self.debug_print(
                            f"  → Generated with fallback: {var_info.operation}_circuit("
                            f"{reg1.qiskit_name}, {reg2.qiskit_name}) -> {result_reg.qiskit_name}"
                        )

                        
            elif var_info.operation in ["not", "neg", "pre_inc", "pre_dec"]:
                # Create unary circuit
                if len(var_info.operands) >= 1:
                    operand = var_info.operands[0]
                    if operand in self.variable_to_qubits:
                        operand_reg = self.variable_to_qubits[operand]
                        result_reg = self.allocate_qubit_register(ssa_name)
                        
                        gate_ops.append(f"    {result_reg.qiskit_name} = q.alloc : !qreg<{result_reg.size}>")
                        gate_ops.append(f"    q.{var_info.operation}_circuit {operand_reg.qiskit_name}, {result_reg.qiskit_name}")
                        self.debug_print(f"  → Generated: {var_info.operation}_circuit({operand_reg.qiskit_name}) -> {result_reg.qiskit_name}")
                    else:
                        self.debug_print(f"  → ERROR: Missing operand for {var_info.operation}: {operand}")
                        # Create fallback
                        fallback_reg = self.allocate_qubit_register(operand)
                        gate_ops.append(f"    {fallback_reg.qiskit_name} = q.alloc : !qreg<{fallback_reg.size}>")
                        gate_ops.append(f"    q.init {fallback_reg.qiskit_name}, 0 : i32  // Fallback for missing operand")
                        
                        result_reg = self.allocate_qubit_register(ssa_name)
                        gate_ops.append(f"    {result_reg.qiskit_name} = q.alloc : !qreg<{result_reg.size}>")
                        gate_ops.append(f"    q.{var_info.operation}_circuit {fallback_reg.qiskit_name}, {result_reg.qiskit_name}")
                        self.debug_print(f"  → Generated with fallback: {var_info.operation}_circuit({fallback_reg.qiskit_name}) -> {result_reg.qiskit_name}")
                        
            elif var_info.operation.endswith("_orig"):
                # This is the original value from a post_inc/post_dec operation
                base_op = var_info.operation.replace("_orig", "")
                operand_var = var_info.dependencies[0]
                
                # Create a unique key for this multi-result operation
                multi_op_key = f"{base_op}_{operand_var}"
                if multi_op_key in self.processed_multi_ops:
                    # Already processed this multi-result operation
                    return
                
                # Find the corresponding _new operation
                new_var = None
                for other_ssa, other_info in self.ssa_variables.items():
                    if (other_info.operation == f"{base_op}_new" and 
                        other_info.dependencies == var_info.dependencies):
                        new_var = other_ssa
                        break
                
                if operand_var in self.variable_to_qubits and new_var:
                    operand_reg = self.variable_to_qubits[operand_var]
                    orig_reg = self.allocate_qubit_register(ssa_name)
                    new_reg = self.allocate_qubit_register(new_var)
                    
                    gate_ops.append(f"    {orig_reg.qiskit_name} = q.alloc : !qreg<{orig_reg.size}>")
                    gate_ops.append(f"    {new_reg.qiskit_name} = q.alloc : !qreg<{new_reg.size}>")
                    gate_ops.append(f"    q.{base_op}_circuit {operand_reg.qiskit_name}, {orig_reg.qiskit_name}, {new_reg.qiskit_name}")
                    self.debug_print(f"  → Generated: {base_op}_circuit({operand_reg.qiskit_name}) -> {orig_reg.qiskit_name}, {new_reg.qiskit_name}")
                    
                    # Mark both variables as processed and this multi-op as done
                    processed.add(new_var)
                    self.processed_multi_ops.add(multi_op_key)
                else:
                    self.debug_print(f"  → ERROR: Missing dependencies for {var_info.operation}")
                    # Create fallbacks if needed
                    if operand_var not in self.variable_to_qubits:
                        fallback_reg = self.allocate_qubit_register(operand_var)
                        gate_ops.append(f"    {fallback_reg.qiskit_name} = q.alloc : !qreg<{fallback_reg.size}>")
                        gate_ops.append(f"    q.init {fallback_reg.qiskit_name}, 0 : i32  // Fallback for missing operand")
                    
                    if new_var:
                        operand_reg = self.variable_to_qubits[operand_var]
                        orig_reg = self.allocate_qubit_register(ssa_name)
                        new_reg = self.allocate_qubit_register(new_var)
                        
                        gate_ops.append(f"    {orig_reg.qiskit_name} = q.alloc : !qreg<{orig_reg.size}>")
                        gate_ops.append(f"    {new_reg.qiskit_name} = q.alloc : !qreg<{new_reg.size}>")
                        gate_ops.append(f"    q.{base_op}_circuit {operand_reg.qiskit_name}, {orig_reg.qiskit_name}, {new_reg.qiskit_name}")
                        self.debug_print(f"  → Generated with fallback: {base_op}_circuit({operand_reg.qiskit_name}) -> {orig_reg.qiskit_name}, {new_reg.qiskit_name}")
                        
                        processed.add(new_var)
                        self.processed_multi_ops.add(multi_op_key)
                    
            elif var_info.operation.endswith("_new"):
                # Skip _new operations as they're handled by _orig operations
                self.debug_print(f"  → Skipping _new operation (handled by _orig)")
                pass
            
            else:
                # Handle any other operation type generically
                self.debug_print(f"  → WARNING: Unknown operation type: {var_info.operation}")
                # Try to create a generic circuit
                if var_info.operands:
                    if len(var_info.operands) == 1:
                        # Treat as unary
                        operand = var_info.operands[0]
                        if operand not in self.variable_to_qubits:
                            fallback_reg = self.allocate_qubit_register(operand)
                            gate_ops.append(f"    {fallback_reg.qiskit_name} = q.alloc : !qreg<{fallback_reg.size}>")
                            gate_ops.append(f"    q.init {fallback_reg.qiskit_name}, 0 : i32  // Fallback")
                        
                        operand_reg = self.variable_to_qubits[operand]
                        result_reg = self.allocate_qubit_register(ssa_name)
                        gate_ops.append(f"    {result_reg.qiskit_name} = q.alloc : !qreg<{result_reg.size}>")
                        gate_ops.append(f"    q.{var_info.operation}_circuit {operand_reg.qiskit_name}, {result_reg.qiskit_name}")
                        self.debug_print(f"  → Generated generic unary: {var_info.operation}_circuit({operand_reg.qiskit_name}) -> {result_reg.qiskit_name}")
                        
                    elif len(var_info.operands) == 2:
                        # Treat as binary
                        for operand in var_info.operands:
                            if operand not in self.variable_to_qubits:
                                fallback_reg = self.allocate_qubit_register(operand)
                                gate_ops.append(f"    {fallback_reg.qiskit_name} = q.alloc : !qreg<{fallback_reg.size}>")
                                gate_ops.append(f"    q.init {fallback_reg.qiskit_name}, 0 : i32  // Fallback")
                        
                        reg1 = self.variable_to_qubits[var_info.operands[0]]
                        reg2 = self.variable_to_qubits[var_info.operands[1]]
                        result_reg = self.allocate_qubit_register(ssa_name)
                        gate_ops.append(f"    {result_reg.qiskit_name} = q.alloc : !qreg<{result_reg.size}>")
                        gate_ops.append(f"    q.{var_info.operation}_circuit {reg1.qiskit_name}, {reg2.qiskit_name}, {result_reg.qiskit_name}")
                        self.debug_print(f"  → Generated generic binary: {var_info.operation}_circuit({reg1.qiskit_name}, {reg2.qiskit_name}) -> {result_reg.qiskit_name}")
            
            processed.add(ssa_name)
        
        # Process all variables in dependency order
        for ssa_name in self.ssa_variables:
            process_variable(ssa_name)
        
        return gate_ops
    
    def generate_gate_mlir(self, mlir_content: str) -> str:
        """Main conversion function"""
        # Step 1: Parse high-level MLIR
        self.parse_high_level_mlir(mlir_content)
        
        if not self.ssa_variables:
            self.debug_print("WARNING: No SSA variables found!")
            return """// WARNING: No operations found
builtin.module {
  "quantum.func"() ({
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}"""
        
        # Step 2: Convert to gate operations
        gate_ops = self.convert_to_gate_level()
        
        # Step 3: Add statistics comment
        operation_counts = {}
        for var in self.ssa_variables.values():
            base_op = var.operation.replace("_orig", "").replace("_new", "")
            operation_counts[base_op] = operation_counts.get(base_op, 0) + 1
        
        # Step 4: Build final MLIR
        result = [
            "// Comprehensive Gate-Level Quantum MLIR",
            f"// Converted {len(self.ssa_variables)} high-level operations",
            f"// Operation types: {', '.join(sorted(operation_counts.keys()))}",
            f"// Total quantum registers: {self.qubit_counter}",
            "builtin.module {",
            '  "quantum.func"() ({'
        ]
        result.extend(gate_ops)
        result.extend([
            "    func.return",
            '  }) {func_name = "quantum_circuit"} : () -> ()',
            "}"
        ])
        
        return '\n'.join(result)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Gate-Level MLIR Converter')
    parser.add_argument('input_file', help='Input high-level MLIR file')
    parser.add_argument('output_file', help='Output gate-level MLIR file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    if args.debug:
        print("📄 Input MLIR content:")
        for i, line in enumerate(mlir_content.split('\n')):
            print(f"{i:2}: {line}")
        print()
    
    # Create converter and convert
    converter = ComprehensiveGateConverter(enable_debug=args.debug)
    gate_mlir = converter.generate_gate_mlir(mlir_content)
    
    # Write output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(gate_mlir)
        print(f"✅ Comprehensive gate-level MLIR written to: {args.output_file}")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)
    
    # Show conversion statistics
    print(f"📊 Conversion Statistics:")
    print(f"   Input variables: {len(converter.ssa_variables)}")
    print(f"   Output registers: {converter.qubit_counter}")
    
    operation_counts = {}
    for var in converter.ssa_variables.values():
        base_op = var.operation.replace("_orig", "").replace("_new", "")
        operation_counts[base_op] = operation_counts.get(base_op, 0) + 1
    
    print(f"   Operations preserved:")
    for op, count in sorted(operation_counts.items()):
        print(f"     {op}: {count}")
    
    if args.debug:
        print("\n📄 Generated gate-level MLIR:")
        print(gate_mlir)

if __name__ == "__main__":
    main()