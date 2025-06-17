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

@dataclass
class LoopStructure:
    """Represents the structure of a while loop"""
    init_vars: Set[str]          # Variables initialized before loop
    condition_vars: Set[str]      # Variables used in condition
    condition_op: str            # Type of condition operation (gt, lt, eq, etc.)
    body_vars: Set[str]          # Variables modified in loop body
    loop_modified_vars: Set[str]  # Variables that change and affect condition
    post_loop_vars: Set[str]     # Variables used after loop
    measured_var: Optional[str]   # Final variable to measure

class GeneralizedWhileLoopHandler:
    def __init__(self, ssa_variables: Dict, variable_to_qubits: Dict, debug_print):
        self.ssa_variables = ssa_variables
        self.variable_to_qubits = variable_to_qubits
        self.debug_print = debug_print
        
    def analyze_loop_structure(self, gate_ops: List[str]) -> LoopStructure:
        """Analyze gate operations to understand loop structure generically"""
        
        init_vars = set()
        condition_vars = set()
        condition_op = None
        body_vars = set()
        loop_modified_vars = set()
        post_loop_vars = set()
        measured_var = None
        
        # Pattern detection for different operation types
        comparison_patterns = {
            'gt_circuit': 'gt',
            'lt_circuit': 'lt', 
            'eq_circuit': 'eq',
            'ne_circuit': 'ne',
            'ge_circuit': 'ge',
            'le_circuit': 'le'
        }
        
        arithmetic_patterns = {
            'add_circuit', 'sub_circuit', 'mul_circuit', 'div_circuit', 'mod_circuit'
        }
        
        # Step 1: Find initialization operations (before any circuit operations)
        circuit_started = False
        for op in gate_ops:
            op_stripped = op.strip()
            
            # Mark when we see first circuit operation
            if any(pattern in op_stripped for pattern in comparison_patterns.keys()) or \
               any(pattern in op_stripped for pattern in arithmetic_patterns):
                circuit_started = True
            
            # Variables initialized before circuits start are loop initializations
            if not circuit_started and ('alloc' in op_stripped or 'init' in op_stripped):
                var_match = re.search(r'(%q\d+)', op_stripped)
                if var_match:
                    init_vars.add(var_match.group(1))
                    self.debug_print(f"Found init variable: {var_match.group(1)}")
        
        # Step 2: Find condition operations and variables
        for op in gate_ops:
            op_stripped = op.strip()
            
            # Look for comparison circuits
            for pattern, op_type in comparison_patterns.items():
                if pattern in op_stripped:
                    condition_op = op_type
                    # Extract operands: q.gt_circuit %q0, %q1, %q2
                    operand_matches = re.findall(r'(%q\d+)', op_stripped)
                    if len(operand_matches) >= 2:
                        condition_vars.update(operand_matches[:2])  # First two are condition operands
                    self.debug_print(f"Found condition: {op_type} with vars {operand_matches[:2]}")
                    break
        
        # Step 3: Find loop body modifications
        for op in gate_ops:
            op_stripped = op.strip()
            
            # Look for arithmetic operations that might modify loop variables
            for pattern in arithmetic_patterns:
                if pattern in op_stripped:
                    operand_matches = re.findall(r'(%q\d+)', op_stripped)
                    if len(operand_matches) >= 3:
                        input1, input2, output = operand_matches[:3]
                        body_vars.update([input1, input2, output])
                        
                        # If output overwrites one of the condition variables, it's a loop modification
                        if output in condition_vars or input1 in condition_vars:
                            loop_modified_vars.add(output)
                            self.debug_print(f"Found loop modification: {pattern} -> {output}")
                    break
        
        # Step 4: Find post-loop operations (measurements)
        for op in gate_ops:
            op_stripped = op.strip()
            
            if 'measure' in op_stripped:
                var_match = re.search(r'measure\s+(%q\d+)', op_stripped)
                if var_match:
                    measured_var = var_match.group(1)
                    post_loop_vars.add(measured_var)
                    self.debug_print(f"Found measurement of: {measured_var}")
        
        # Step 5: Infer the primary loop variable
        # The primary loop variable is typically:
        # 1. Used in condition AND modified in body
        primary_loop_var = None
        for var in condition_vars:
            if var in loop_modified_vars:
                primary_loop_var = var
                break
        
        if not primary_loop_var and condition_vars:
            primary_loop_var = list(condition_vars)[0]  # Fallback
        
        structure = LoopStructure(
            init_vars=init_vars,
            condition_vars=condition_vars,
            condition_op=condition_op,
            body_vars=body_vars,
            loop_modified_vars=loop_modified_vars,
            post_loop_vars=post_loop_vars,
            measured_var=measured_var or primary_loop_var
        )
        
        self.debug_print(f"Loop structure analysis:")
        self.debug_print(f"  Init vars: {init_vars}")
        self.debug_print(f"  Condition vars: {condition_vars}")
        self.debug_print(f"  Condition op: {condition_op}")
        self.debug_print(f"  Body vars: {body_vars}")
        self.debug_print(f"  Loop modified: {loop_modified_vars}")
        self.debug_print(f"  Post-loop vars: {post_loop_vars}")
        self.debug_print(f"  Measured var: {measured_var}")
        
        return structure
    
    def categorize_operations_generically(self, gate_ops: List[str], structure: LoopStructure) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Categorize operations based on analyzed loop structure"""
        
        init_ops = []
        condition_ops = []
        body_ops = []
        post_loop_ops = []
        
        for op in gate_ops:
            op_stripped = op.strip()
            
            # Categorize based on variables involved
            op_vars = set(re.findall(r'(%q\d+)', op_stripped))
            
            # 1. Initialization operations
            if op_vars.intersection(structure.init_vars) and \
               ('alloc' in op_stripped or 'init' in op_stripped) and \
               not any(circuit in op_stripped for circuit in ['_circuit']):
                init_ops.append(op_stripped)
                continue
            
            # 2. Condition operations
            if structure.condition_op and f'{structure.condition_op}_circuit' in op_stripped:
                condition_ops.append(op_stripped)
                continue
            
            # 3. Body operations - operations that modify loop variables
            if op_vars.intersection(structure.loop_modified_vars) and '_circuit' in op_stripped:
                # Check if this operation updates a loop variable in-place
                operand_matches = re.findall(r'(%q\d+)', op_stripped)
                if len(operand_matches) >= 3:
                    input1, input2, output = operand_matches[:3]
                    # If output should go back to a condition variable, modify the operation
                    if input1 in structure.condition_vars and output != input1:
                        op_modified = op_stripped.replace(output, input1)
                        body_ops.append(op_modified)
                        self.debug_print(f"Modified body op: {op_stripped} -> {op_modified}")
                    else:
                        body_ops.append(op_stripped)
                else:
                    body_ops.append(op_stripped)
                continue
            
            # 4. Support operations for body (like constant initialization)
            if op_vars.intersection(structure.body_vars) and \
               ('alloc' in op_stripped or 'init' in op_stripped):
                body_ops.append(op_stripped)
                continue
            
            # 5. Post-loop operations (measurements, etc.)
            if 'measure' in op_stripped:
                # Always measure the primary loop variable (the one that gets modified)
                primary_loop_var = None
                for var in structure.condition_vars:
                    if var in structure.loop_modified_vars:
                        primary_loop_var = var
                        break
                
                if not primary_loop_var and structure.condition_vars:
                    primary_loop_var = list(structure.condition_vars)[0]
                
                if primary_loop_var:
                    # Replace any measurement with measurement of the primary loop variable
                    result_var = re.search(r'(%q\d+)\s*=\s*q\.measure', op_stripped)
                    if result_var:
                        op_modified = f"{result_var.group(1)} = q.measure {primary_loop_var} : !qreg -> i32"
                        post_loop_ops.append(op_modified)
                    else:
                        post_loop_ops.append(f"q.measure {primary_loop_var} : !qreg -> i32")
                continue
            else:
                # Skip measurements of non-loop variables
                continue
            
            # 6. Other post-loop operations
            if op_vars.intersection(structure.post_loop_vars):
                post_loop_ops.append(op_stripped)
                continue
            
            # Default: if we can't categorize, try to infer
            self.debug_print(f"Uncategorized operation: {op_stripped}")
            
        return init_ops, condition_ops, body_ops, post_loop_ops
    
    def generate_while_loop_mlir(self, gate_ops: List[str]) -> List[str]:
        """Generate complete while loop MLIR using generic analysis"""
        
        # Step 1: Analyze the loop structure
        structure = self.analyze_loop_structure(gate_ops)
        
        # Step 2: Categorize operations generically
        init_ops, condition_ops, body_ops, post_loop_ops = \
            self.categorize_operations_generically(gate_ops, structure)
        
        # Step 3: Build MLIR structure
        result = []
        
        # Add initialization operations (before while loop)
        for op in init_ops:
            result.append(f"    {op}")
        
        # Start while loop structure
        result.append('    "quantum.while"() ({')
        
        # Add condition block operations
        for op in condition_ops:
            result.append(f"      {op}")
        
        # Add condition terminator - find the condition result variable
        condition_result_var = None
        for op in condition_ops:
            if f'{structure.condition_op}_circuit' in op:
                matches = re.findall(r'(%q\d+)', op)
                if len(matches) >= 3:
                    condition_result_var = matches[2]  # Third operand is usually result
                    break
        
        if condition_result_var:
            result.append(f'      "quantum.condition"({condition_result_var}) : (i1) -> ()')
        else:
            result.append('      "quantum.condition"(%q2) : (i1) -> ()')  # Fallback
        
        # Transition to body block
        result.append('    }, {')
        
        # Add body block operations
        for op in body_ops:
            result.append(f"      {op}")
        
        # Close while loop
        result.append('    }) : () -> ()')
        
        # Add post-loop operations
        for op in post_loop_ops:
            result.append(f"    {op}")
        
        return result

def integrate_generalized_while_handler(converter_instance):
    """Integration function to add generalized while loop handling to existing converter"""
    
    def new_generate_gate_mlir(self, mlir_content: str) -> str:
        """Enhanced generate_gate_mlir with generalized while loop support"""
        has_while_loop = '"quantum.while"()' in mlir_content
        
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
            "// Comprehensive Gate-Level Quantum MLIR (Generalized While Loop)",
            f"// Converted {len(self.ssa_variables)} high-level operations",
            f"// Operation types: {', '.join(sorted(operation_counts.keys()))}",
            f"// Total quantum registers: {self.qubit_counter}",
            "builtin.module {",
            '  "quantum.func"() ({'
        ]
        
        if has_while_loop:
            self.debug_print("Processing WHILE LOOP with generalized handler")
            
            # Use generalized while loop handler
            handler = GeneralizedWhileLoopHandler(
                self.ssa_variables, 
                self.variable_to_qubits,
                self.debug_print
            )
            
            loop_mlir = handler.generate_while_loop_mlir(gate_ops)
            result.extend(loop_mlir)
            
        else:
            # No while loops - use original method
            for op in gate_ops:
                result.append(op)
        
        # Add function closing
        result.extend([
            "    func.return",
            '  }) {func_name = "quantum_circuit"} : () -> ()',
            "}"
        ])
        
        return '\n'.join(result)
    
    # Replace the method
    converter_instance.generate_gate_mlir = new_generate_gate_mlir.__get__(converter_instance)
    return converter_instance

class ComprehensiveGateConverter:
    def __init__(self, enable_debug=False):
        self.enable_debug = enable_debug
        self.ssa_variables: Dict[str, SSAVariable] = {}
        self.variable_to_qubits: Dict[str, QubitRegister] = {}
        self.qubit_counter = 0
        self.processed_multi_ops: Set[str] = set()
        self.original_mlir_content = ""
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
        self.original_mlir_content = mlir_content
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
        if '"quantum.while"()' in line:
            
            self.debug_print(f"  → PRESERVING WHILE LOOP structure")
            
            return
            
        if '"quantum.condition"(' in line:
            self.debug_print(f"  → PRESERVING CONDITION operation")
            return
        
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
        has_while_loop = '"quantum.while"()' in mlir_content
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
        if has_while_loop:
            self.debug_print("Processing WHILE LOOP with generalized handler")
            
            # Use generalized while loop handler
            handler = GeneralizedWhileLoopHandler(
                self.ssa_variables, 
                self.variable_to_qubits,
                self.debug_print
            )
            
            loop_mlir = handler.generate_while_loop_mlir(gate_ops)
            result.extend(loop_mlir)
                
        else:
                # No while loops - use original method
            for op in gate_ops:
                result.append(op)
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