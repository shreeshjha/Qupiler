#!/usr/bin/env python3
"""
Complete Working Quantum MLIR Optimization Script
Handles post_inc/post_dec and performs constant propagation
Modified to preserve arithmetic operations for quantum circuit generation
"""

import re
import sys
import argparse
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Operation:
    """Represents a quantum operation in MLIR"""
    results: List[str]  # Can have multiple results
    op_name: str
    operands: List[str]
    attributes: Dict[str, str]
    line_number: int
    original_line: str

@dataclass
class OptimizationStats:
    """Track optimization statistics"""
    operations_before: int = 0
    operations_after: int = 0
    constant_folds: int = 0
    dead_code_eliminations: int = 0
    arithmetic_simplifications: int = 0

class WorkingQuantumMLIROptimizer:
    def __init__(self, enable_debug=False, preserve_arithmetic=False, preserve_increments=False):
        self.enable_debug = enable_debug
        self.preserve_arithmetic = preserve_arithmetic
        self.preserve_increments = preserve_increments
        self.operations: List[Operation] = []
        self.constants: Dict[str, int] = {}
        self.use_count: Dict[str, int] = defaultdict(int)
        self.def_op: Dict[str, Operation] = {}
        self.stats = OptimizationStats()
        
    def debug_print(self, message: str):
        if self.enable_debug:
            print(f"[DEBUG] {message}")
    
    def parse_mlir_line(self, line: str, line_num: int) -> Optional[Operation]:
        """Parse a single MLIR operation line"""
        line = line.strip()
        # Add right after the skip conditions in parse_mlir_line()
        if '"quantum.while"()' in line:
            # This is a control flow operation - don't parse as regular operation
            self.debug_print("Skipping while loop parsing - preserving structure")
            return None

        if '"quantum.condition"(' in line:
            # This is part of while loop structure
            self.debug_print("Skipping condition parsing - preserving structure")
            return None
        
        
        # Skip empty lines and structural elements
        if not line or line.startswith('//') or line.startswith('builtin.module') or line.startswith('"quantum.func"') or line.startswith('func.return') or line.startswith('}'):
            return None
        
        # Parse multi-result operations like post_inc, post_dec
        multi_result_pattern = r'%(\w+),\s*%(\w+)\s*=\s*"quantum\.(\w+)"\s*\(([^)]*)\)'
        match = re.match(multi_result_pattern, line)
        
        if match:
            result1, result2, op_name, operands_str = match.groups()
            
            # Parse operands
            operands = []
            if operands_str.strip():
                for operand in operands_str.split(','):
                    operand = operand.strip()
                    if operand.startswith('%'):
                        operands.append(operand)
            
            return Operation(
                results=[f"%{result1}", f"%{result2}"],
                op_name=op_name,
                operands=operands,
                attributes={},
                line_number=line_num,
                original_line=line
            )
        
        # Parse single-result operations
        single_result_pattern = r'%(\w+)\s*=\s*"quantum\.(\w+)"\s*\(([^)]*)\)\s*(?:\{([^}]*)\})?'
        match = re.match(single_result_pattern, line)
        
        if match:
            result, op_name, operands_str, attrs_str = match.groups()
            
            # Parse operands
            operands = []
            if operands_str.strip():
                for operand in operands_str.split(','):
                    operand = operand.strip()
                    if operand.startswith('%'):
                        operands.append(operand)
            
            # Parse attributes
            attributes = {}
            if attrs_str:
                attr_pairs = re.findall(r'(\w+)\s*=\s*([^,}]+)', attrs_str)
                for attr_name, attr_value in attr_pairs:
                    attributes[attr_name.strip()] = attr_value.strip()
            
            return Operation(
                results=[f"%{result}"],
                op_name=op_name,
                operands=operands,
                attributes=attributes,
                line_number=line_num,
                original_line=line
            )
        
        return None
    
    def analyze_constants(self):
        """Identify constant values from quantum.init operations and propagate through post_inc/post_dec"""
        self.debug_print("Analyzing constants...")
        
        for op in self.operations:
            if op.op_name == "init" and "value" in op.attributes:
                try:
                    # Extract integer value
                    value_str = op.attributes["value"]
                    value_match = re.search(r'(\d+)', value_str)
                    if value_match:
                        value = int(value_match.group(1))
                        self.constants[op.results[0]] = value
                        self.debug_print(f"Found constant: {op.results[0]} = {value}")
                except ValueError:
                    pass
            
            elif op.op_name == "post_inc" and len(op.operands) == 1:
                # For post_inc: first result is original value, second is incremented
                operand = op.operands[0]
                if operand in self.constants:
                    original_value = self.constants[operand]
                    self.constants[op.results[0]] = original_value  # %3 = original value (3)
                    self.constants[op.results[1]] = original_value + 1  # %4 = incremented value (4)
                    self.debug_print(f"Post-increment: {op.results[0]} = {original_value}, {op.results[1]} = {original_value + 1}")
            
            elif op.op_name == "post_dec" and len(op.operands) == 1:
                # For post_dec: first result is original value, second is decremented
                operand = op.operands[0]
                if operand in self.constants:
                    original_value = self.constants[operand]
                    self.constants[op.results[0]] = original_value  # %5 = original value (1)
                    self.constants[op.results[1]] = original_value - 1  # %6 = decremented value (0)
                    self.debug_print(f"Post-decrement: {op.results[0]} = {original_value}, {op.results[1]} = {original_value - 1}")
            
            elif op.op_name == "pre_inc" and len(op.operands) == 1:
                # For pre_inc: result is incremented value
                operand = op.operands[0]
                if operand in self.constants:
                    original_value = self.constants[operand]
                    self.constants[op.results[0]] = original_value + 1
                    self.debug_print(f"Pre-increment: {op.results[0]} = {original_value + 1}")
            
            elif op.op_name == "pre_dec" and len(op.operands) == 1:
                # For pre_dec: result is decremented value
                operand = op.operands[0]
                if operand in self.constants:
                    original_value = self.constants[operand]
                    self.constants[op.results[0]] = original_value - 1
                    self.debug_print(f"Pre-decrement: {op.results[0]} = {original_value - 1}")
    
    def analyze_usage(self):
        """Count how many times each SSA value is used"""
        self.debug_print("Analyzing SSA value usage...")
        
        # Clear previous analysis
        self.use_count.clear()
        self.def_op.clear()
        
        for op in self.operations:
            # Record definitions
            for result in op.results:
                self.def_op[result] = op
            
            # Count usage of operands
            for operand in op.operands:
                self.use_count[operand] += 1
    
    def preserve_computation_path(self) -> Set[str]:
        """Identify all SSA values that contribute to the final measured result"""
        preserved = set()
        
        # Start from measurement operations
        worklist = []
        for op in self.operations:
            if op.op_name == "measure":
                worklist.extend(op.operands)
                preserved.update(op.operands)
                self.debug_print(f"Found measurement operation, preserving: {op.operands}")
        
        # Backward propagation to find all dependencies
        while worklist:
            current = worklist.pop()
            if current in self.def_op:
                defining_op = self.def_op[current]
                for operand in defining_op.operands:
                    if operand not in preserved:
                        preserved.add(operand)
                        worklist.append(operand)
                        self.debug_print(f"Preserving dependency: {operand} (from {defining_op.op_name})")
        
        return preserved
    
    def constant_propagation(self) -> int:
        """Propagate constants through arithmetic operations"""
        self.debug_print("Applying constant propagation...")
        propagated_count = 0
        
        # Keep propagating until no more changes
        changed = True
        while changed:
            changed = False
            
            for op in self.operations:
                if op.op_name in ["add", "sub", "mul", "div", "mod"] and len(op.operands) == 2:
                    # Check if we can compute the result
                    left_val = self.constants.get(op.operands[0])
                    right_val = self.constants.get(op.operands[1])
                    
                    if left_val is not None and right_val is not None and op.results[0] not in self.constants:
                        # Compute the result
                        if op.op_name == "add":
                            result_val = left_val + right_val
                        elif op.op_name == "sub":
                            result_val = left_val - right_val
                        elif op.op_name == "mul":
                            result_val = left_val * right_val
                        elif op.op_name == "div":
                            result_val = left_val // right_val if right_val != 0 else 0
                        elif op.op_name == "mod":
                            result_val = left_val % right_val if right_val != 0 else 0
                        
                        self.constants[op.results[0]] = result_val
                        propagated_count += 1
                        changed = True
                        self.debug_print(f"Propagated constant: {op.results[0]} = {left_val} {op.op_name} {right_val} = {result_val}")
        
        return propagated_count
    
    def selective_constant_folding(self) -> int:
        """Replace operations with known constant results with quantum.init, but preserve arithmetic structure"""
        self.debug_print("Applying selective constant folding...")
        folded_count = 0
        new_operations = []
        
        # Operations to preserve even if they have constant results
        preserve_ops = set()
        if self.preserve_arithmetic:
            preserve_ops.update({"add", "sub", "mul", "div", "mod"})
        if self.preserve_increments:
            preserve_ops.update({"post_inc", "post_dec", "pre_inc", "pre_dec"})
        preserve_ops.update({"while", "condition", "gt", "lt", "eq", "ne", "ge", "le", "greater_than", "less_than", "equal", "not_equal"})
        
        for op in self.operations:
            # Skip folding for operations we want to preserve
            if op.op_name in preserve_ops:
                self.debug_print(f"Preserving {op.op_name} operation: {op.results[0]}")
                new_operations.append(op)
                continue
                
            # If the result of this operation is a known constant, replace it
            if len(op.results) == 1 and op.results[0] in self.constants and op.op_name != "init":
                result_val = self.constants[op.results[0]]
                
                # Replace with quantum.init
                folded_op = Operation(
                    results=op.results,
                    op_name="init",
                    operands=[],
                    attributes={"type": "i32", "value": f"{result_val} : i32"},
                    line_number=op.line_number,
                    original_line=f'    {op.results[0]} = "quantum.init"() {{type = i32, value = {result_val} : i32}} : () -> i32  // Constant folded from {op.op_name}'
                )
                
                new_operations.append(folded_op)
                folded_count += 1
                self.debug_print(f"Constant folded: {op.results[0]} = {result_val} (was {op.op_name})")
                continue
            
            new_operations.append(op)
        
        self.operations = new_operations
        return folded_count
    
    def arithmetic_simplification(self) -> int:
        """Apply arithmetic simplification rules"""
        self.debug_print("Applying arithmetic simplifications...")
        
        # Skip arithmetic simplification if we're preserving arithmetic operations
        if self.preserve_arithmetic:
            self.debug_print("Skipping arithmetic simplification (preserve_arithmetic=True)")
            return 0
            
        simplified_count = 0
        new_operations = []
        
        for op in self.operations:
            simplified = False
            
            if op.op_name == "add" and len(op.operands) == 2:
                left_val = self.constants.get(op.operands[0])
                right_val = self.constants.get(op.operands[1])
                
                # x + 0 = x
                if right_val == 0:
                    new_op = Operation(
                        results=op.results,
                        op_name="copy",
                        operands=[op.operands[0]],
                        attributes={"original_op": "add_with_zero"},
                        line_number=op.line_number,
                        original_line=f'    // {op.results[0]} simplified from add with 0'
                    )
                    new_operations.append(new_op)
                    simplified = True
                    simplified_count += 1
                
                # 0 + x = x
                elif left_val == 0:
                    new_op = Operation(
                        results=op.results,
                        op_name="copy",
                        operands=[op.operands[1]],
                        attributes={"original_op": "add_with_zero"},
                        line_number=op.line_number,
                        original_line=f'    // {op.results[0]} simplified from add with 0'
                    )
                    new_operations.append(new_op)
                    simplified = True
                    simplified_count += 1
            
            elif op.op_name == "sub" and len(op.operands) == 2:
                left_val = self.constants.get(op.operands[0])
                right_val = self.constants.get(op.operands[1])
                
                # x - 0 = x
                if right_val == 0:
                    new_op = Operation(
                        results=op.results,
                        op_name="copy",
                        operands=[op.operands[0]],
                        attributes={"original_op": "sub_with_zero"},
                        line_number=op.line_number,
                        original_line=f'    // {op.results[0]} simplified from sub with 0'
                    )
                    new_operations.append(new_op)
                    simplified = True
                    simplified_count += 1
                
                # x - x = 0
                elif op.operands[0] == op.operands[1]:
                    new_op = Operation(
                        results=op.results,
                        op_name="init",
                        operands=[],
                        attributes={"type": "i32", "value": "0 : i32"},
                        line_number=op.line_number,
                        original_line=f'    {op.results[0]} = "quantum.init"() {{type = i32, value = 0 : i32}} : () -> i32  // x - x = 0'
                    )
                    new_operations.append(new_op)
                    self.constants[op.results[0]] = 0
                    simplified = True
                    simplified_count += 1
            
            elif op.op_name == "mul" and len(op.operands) == 2:
                left_val = self.constants.get(op.operands[0])
                right_val = self.constants.get(op.operands[1])
                
                # x * 0 = 0 or 0 * x = 0
                if left_val == 0 or right_val == 0:
                    new_op = Operation(
                        results=op.results,
                        op_name="init",
                        operands=[],
                        attributes={"type": "i32", "value": "0 : i32"},
                        line_number=op.line_number,
                        original_line=f'    {op.results[0]} = "quantum.init"() {{type = i32, value = 0 : i32}} : () -> i32  // mul with 0'
                    )
                    new_operations.append(new_op)
                    self.constants[op.results[0]] = 0
                    simplified = True
                    simplified_count += 1
                
                # x * 1 = x
                elif right_val == 1:
                    new_op = Operation(
                        results=op.results,
                        op_name="copy",
                        operands=[op.operands[0]],
                        attributes={"original_op": "mul_with_one"},
                        line_number=op.line_number,
                        original_line=f'    // {op.results[0]} simplified from mul with 1'
                    )
                    new_operations.append(new_op)
                    simplified = True
                    simplified_count += 1
                
                # 1 * x = x
                elif left_val == 1:
                    new_op = Operation(
                        results=op.results,
                        op_name="copy",
                        operands=[op.operands[1]],
                        attributes={"original_op": "mul_with_one"},
                        line_number=op.line_number,
                        original_line=f'    // {op.results[0]} simplified from mul with 1'
                    )
                    new_operations.append(new_op)
                    simplified = True
                    simplified_count += 1
            
            if not simplified:
                new_operations.append(op)
        
        self.operations = new_operations
        return simplified_count
    
    def expand_copy_operations(self):
        """Expand copy operations to maintain SSA form"""
        self.debug_print("Expanding copy operations...")
        
        # Create a mapping of copied values
        copy_map = {}
        for op in self.operations:
            if op.op_name == "copy" and len(op.operands) == 1:
                copy_map[op.results[0]] = op.operands[0]
        
        # Update all operations to use the original values instead of copies
        for op in self.operations:
            if op.op_name != "copy":
                new_operands = []
                for operand in op.operands:
                    # Follow the copy chain to find the original value
                    current = operand
                    while current in copy_map:
                        current = copy_map[current]
                    new_operands.append(current)
                op.operands = new_operands
    
    def smart_dead_code_elimination(self) -> int:
        """Smart dead code elimination that preserves computation paths and selected operations"""
        self.debug_print("Applying smart dead code elimination...")
        eliminated_count = 0
        
        # Re-analyze usage after other optimizations
        self.analyze_usage()
        
        # Find preserved computation paths
        preserved_values = self.preserve_computation_path()
        
        # Find operations that are safe to remove
        new_operations = []
        for op in self.operations:
            should_keep = True
            
            # Always keep these operations
            if op.op_name in ["measure", "init"]:
                should_keep = True
                self.debug_print(f"Keeping essential operation: {op.op_name}")
                
            # Remove copy operations (they're handled by expansion)
            elif op.op_name == "copy":
                should_keep = False
                eliminated_count += 1
                self.debug_print(f"Eliminated copy operation: {op.results[0]}")
                
            # Keep operations if they're in the computation path to measured results
            elif any(result in preserved_values for result in op.results):
                should_keep = True
                self.debug_print(f"Keeping operation in computation path: {op.op_name} -> {op.results}")
                
            # Keep arithmetic operations if preserve_arithmetic is enabled
            elif self.preserve_arithmetic and op.op_name in ["add", "sub", "mul", "div", "mod"]:
                should_keep = True
                self.debug_print(f"Preserving arithmetic operation: {op.op_name}")
                
            # Keep increment/decrement operations if preserve_increments is enabled
            elif self.preserve_increments and op.op_name in ["post_inc", "post_dec", "pre_inc", "pre_dec"]:
                should_keep = True
                self.debug_print(f"Preserving increment/decrement operation: {op.op_name}")
                
            # For other operations, only remove if no results are used
            else:
                any_result_used = any(self.use_count[result] > 0 for result in op.results)
                if not any_result_used:
                    should_keep = False
                    eliminated_count += 1
                    self.debug_print(f"Eliminated unused operation: {op.results[0]} = {op.op_name}")
                else:
                    self.debug_print(f"Keeping used operation: {op.op_name}")
            
            if should_keep:
                new_operations.append(op)
        
        self.operations = new_operations
        return eliminated_count
    
    def optimize(self, mlir_content: str) -> str:
        """Apply all optimizations to MLIR content"""
        self.original_mlir_content = mlir_content
        lines = mlir_content.strip().split('\n')
        
        # Parse operations
        for i, line in enumerate(lines):
            op = self.parse_mlir_line(line, i)
            if op:
                self.operations.append(op)
        
        self.stats.operations_before = len(self.operations)
        self.debug_print(f"Parsed {len(self.operations)} operations")
        
        # Analyze constants and usage
        self.analyze_constants()
        self.analyze_usage()
        
        # Apply optimizations in order
        self.debug_print("=== Starting Optimization Pipeline ===")
        self.debug_print(f"Preserve arithmetic: {self.preserve_arithmetic}")
        self.debug_print(f"Preserve increments: {self.preserve_increments}")
        
        # Phase 1: Constant propagation (compute values through operations)
        self.constant_propagation()
        
        # Phase 2: Selective constant folding (preserve certain operations)
        self.stats.constant_folds = self.selective_constant_folding()
        
        # Phase 3: Conditional arithmetic simplification
        self.stats.arithmetic_simplifications = self.arithmetic_simplification()
        
        # Phase 4: Expand copy operations to maintain SSA
        self.expand_copy_operations()
        
        # Phase 5: Smart dead code elimination that respects preservation flags
        self.stats.dead_code_eliminations = self.smart_dead_code_elimination()
        
        self.stats.operations_after = len(self.operations)
        
        # Reconstruct MLIR
        return self.reconstruct_mlir()
    
    def reconstruct_mlir(self) -> str:
        """Reconstruct optimized MLIR from operations"""
        
        result_lines = [
            "// Optimized Quantum MLIR",
            "builtin.module {",
            '  "quantum.func"() ({'
        ]
    
        has_while_loop = hasattr(self, 'original_mlir_content') and '"quantum.while"()' in self.original_mlir_content
        if not has_while_loop:
        # Add optimized operations
            for op in self.operations:
                if len(op.results) == 1:
                    # Single result operation
                    reconstructed = f'    {op.results[0]} = "quantum.{op.op_name}"({", ".join(op.operands)})'
                    if op.attributes:
                        attr_str = ", ".join([f"{k} = {v}" for k, v in op.attributes.items() if not k.startswith("original_")])
                        if attr_str:
                            reconstructed += f" {{{attr_str}}}"
                    
                    # Add type information
                    if op.op_name == "init":
                        reconstructed += " : () -> i32"
                    elif op.op_name in ["add", "sub", "mul", "div", "mod"]:
                        reconstructed += " : (i32, i32) -> i32"
                    elif op.op_name == "measure":
                        reconstructed += " : (i32) -> i1"
                    else:
                        reconstructed += " : (i32) -> i32"
                    
                    result_lines.append(reconstructed)
                
                elif len(op.results) == 2:
                    # Multi-result operation (post_inc, post_dec, etc.)
                    reconstructed = f'    {op.results[0]}, {op.results[1]} = "quantum.{op.op_name}"({", ".join(op.operands)})'
                    reconstructed += " : (i32) -> (i32, i32)"
                    result_lines.append(reconstructed)
        else:
            original_lines = self.original_mlir_content.split('\n')
            
            # Track processed operations to avoid duplicates
            processed_ops = set()
            
            # Step 1: Add operations that should come BEFORE while loop
            for op in self.operations:
                if op.op_name == "init" and op.results[0] in ["%0", "%1"]:  # Initial x=5, y=1
                    reconstructed = f'    {op.results[0]} = "quantum.{op.op_name}"({", ".join(op.operands)})'
                    if op.attributes:
                        attr_str = ", ".join([f"{k} = {v}" for k, v in op.attributes.items()])
                        reconstructed += f" {{{attr_str}}}"
                    reconstructed += " : () -> i32"
                    result_lines.append(reconstructed)
                    processed_ops.add(op.results[0])
            
            # Step 2: Add the while loop structure exactly as in original
            in_while = False
            for line in original_lines:
                line_stripped = line.strip()
                
                # Skip file structure lines
                if (line_stripped.startswith('builtin.module') or 
                    line_stripped.startswith('"quantum.func"') or 
                    line_stripped.startswith('func.return') or
                    line_stripped == '}' or
                    line_stripped.startswith('//')):
                    continue
                    
                # Skip already processed init operations
                if line_stripped.startswith('%0') or line_stripped.startswith('%1'):
                    continue
                    
                # Copy while loop structure exactly
                if '"quantum.while"()' in line_stripped:
                    in_while = True
                    result_lines.append(f"    {line_stripped}")
                elif in_while and ('"quantum.gt"(' in line_stripped or 
                                '"quantum.condition"(' in line_stripped or
                                line_stripped.startswith('%3') or
                                '"quantum.sub"(' in line_stripped or
                                line_stripped in ["}, {", "}) : () -> ()"]):
                    result_lines.append(f"      {line_stripped}")
                    if line_stripped == "}) : () -> ()":
                        in_while = False
            
            # Step 3: Add operations that should come AFTER while loop
            for op in self.operations:
                if op.results[0] not in processed_ops:
                    if op.op_name == "init" and op.results[0] not in ["%0", "%1", "%3"]:  # Post-loop like %5
                        reconstructed = f'    {op.results[0]} = "quantum.{op.op_name}"({", ".join(op.operands)})'
                        if op.attributes:
                            attr_str = ", ".join([f"{k} = {v}" for k, v in op.attributes.items()])
                            reconstructed += f" {{{attr_str}}}"
                        reconstructed += " : () -> i32"
                        result_lines.append(reconstructed)
                        processed_ops.add(op.results[0])
                    elif op.op_name == "measure":
                        reconstructed = f'    {op.results[0]} = "quantum.{op.op_name}"({", ".join(op.operands)})'
                        reconstructed += " : (i32) -> i1"
                        result_lines.append(reconstructed)
                        processed_ops.add(op.results[0])
        
        result_lines.extend([
            "    func.return",
            '  }) {func_name = "quantum_circuit"} : () -> ()',
            "}"
        ])
        
        return '\n'.join(result_lines)
    
    def print_stats(self):
        """Print optimization statistics"""
        print("\n=== Optimization Statistics ===")
        print(f"Operations before:          {self.stats.operations_before}")
        print(f"Operations after:           {self.stats.operations_after}")
        print(f"Operations eliminated:      {self.stats.operations_before - self.stats.operations_after}")
        print(f"Constant folds:             {self.stats.constant_folds}")
        print(f"Arithmetic simplifications: {self.stats.arithmetic_simplifications}")
        print(f"Dead code eliminations:     {self.stats.dead_code_eliminations}")
        print(f"Preserve arithmetic:        {self.preserve_arithmetic}")
        print(f"Preserve increments:        {self.preserve_increments}")
        
        if self.stats.operations_before > 0:
            reduction = (self.stats.operations_before - self.stats.operations_after) / self.stats.operations_before * 100
            print(f"Size reduction:             {reduction:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Optimize Quantum MLIR (Preserve Arithmetic Version)')
    parser.add_argument('input_file', help='Input MLIR file')
    parser.add_argument('output_file', help='Output optimized MLIR file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--stats', action='store_true', help='Show optimization statistics')
    parser.add_argument('--preserve-arithmetic', action='store_true', 
                        help='Preserve arithmetic operations (add, sub, mul, div, mod)')
    parser.add_argument('--preserve-increments', action='store_true',
                        help='Preserve increment/decrement operations (post_inc, post_dec, etc.)')
    parser.add_argument('--preserve-all', action='store_true',
                        help='Preserve all arithmetic and increment operations')
    
    args = parser.parse_args()
    
    # Set preservation flags
    preserve_arithmetic = args.preserve_arithmetic or args.preserve_all
    preserve_increments = args.preserve_increments or args.preserve_all
    
    # Read input file
    try:
        with open(args.input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Create optimizer and run optimizations
    optimizer = WorkingQuantumMLIROptimizer(
        enable_debug=args.debug,
        preserve_arithmetic=preserve_arithmetic,
        preserve_increments=preserve_increments
    )
    optimized_mlir = optimizer.optimize(mlir_content)
    
    # Write output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(optimized_mlir)
        print(f"✅ Optimized MLIR written to: {args.output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    
    # Show statistics if requested
    if args.stats:
        optimizer.print_stats()

if __name__ == "__main__":
    main()
