#!/usr/bin/env python3
"""
Fixed Gate-Level MLIR Optimizer - Simplified and Robust Version
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

class GateLevelOptimizer:
    def __init__(self, enable_debug=False):
        self.enable_debug = enable_debug
        self.ssa_variables: Dict[str, SSAVariable] = {}
        self.variable_to_qubits: Dict[str, str] = {}
        self.qubit_counter = 0
        self.processed_multi_ops: Set[str] = set()
        
    def debug_print(self, message: str):
        if self.enable_debug:
            print(f"[DEBUG] {message}")
    
    def parse_high_level_mlir(self, mlir_content: str):
        """Parse high-level MLIR and extract all SSA variables"""
        self.debug_print("=== Parsing High-Level MLIR ===")
        
        lines = mlir_content.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('builtin') or line.startswith('"quantum.func"') or line.startswith('func.return') or line == '}':
                continue
                
            self.debug_print(f"Line {i:2}: {line}")
            
            # Pattern 1: quantum.init operations
            # %0 = "quantum.init"() {type = i32, value = 1 : i32} : () -> i32
            init_match = re.search(r'%(\d+)\s*=\s*"quantum\.init"\(\).*?value\s*=\s*(\d+)', line)
            if init_match:
                var_id, value = init_match.groups()
                ssa_name = f"%{var_id}"
                self.ssa_variables[ssa_name] = SSAVariable(ssa_name, "init", int(value), [])
                self.debug_print(f"  → Found INIT: {ssa_name} = {value}")
                continue
            
            # Pattern 2: Multi-result operations (post_inc, post_dec, pre_inc, pre_dec)
            # %3, %4 = "quantum.post_inc"(%2) : (i32) -> (i32, i32)
            multi_result_match = re.search(r'%(\d+),\s*%(\d+)\s*=\s*"quantum\.(\w+)"\s*\(\s*%(\d+)\s*\)', line)
            if multi_result_match:
                result1_id, result2_id, operation, operand_id = multi_result_match.groups()
                result1_ssa = f"%{result1_id}"
                result2_ssa = f"%{result2_id}"
                operand_ssa = f"%{operand_id}"
                
                # Create entries for both results
                self.ssa_variables[result1_ssa] = SSAVariable(result1_ssa, f"{operation}_orig", None, [operand_ssa])
                self.ssa_variables[result2_ssa] = SSAVariable(result2_ssa, f"{operation}_new", None, [operand_ssa])
                self.debug_print(f"  → Found MULTI-RESULT: {result1_ssa}, {result2_ssa} = {operation}({operand_ssa})")
                continue
            
            # Pattern 3: quantum.measure operations
            # %9 = "quantum.measure"(%8) : (i32) -> i1
            measure_match = re.search(r'%(\d+)\s*=\s*"quantum\.measure"\s*\(\s*%(\d+)\s*\)', line)
            if measure_match:
                result_var, measured_var = measure_match.groups()
                result_ssa = f"%{result_var}"
                measured_ssa = f"%{measured_var}"
                self.ssa_variables[result_ssa] = SSAVariable(result_ssa, "measure", None, [measured_ssa])
                self.debug_print(f"  → Found MEASURE: {result_ssa} = measure({measured_ssa})")
                continue
            
            # Pattern 4: arithmetic operations
            # %7 = "quantum.add"(%5, %1) : (i32, i32) -> i32
            arith_match = re.search(r'%(\d+)\s*=\s*"quantum\.(\w+)"\s*\(\s*%(\d+)\s*,\s*%(\d+)\s*\)', line)
            if arith_match:
                result_var, operation, operand1, operand2 = arith_match.groups()
                result_ssa = f"%{result_var}"
                self.ssa_variables[result_ssa] = SSAVariable(result_ssa, operation, None, [f"%{operand1}", f"%{operand2}"])
                self.debug_print(f"  → Found ARITH: {result_ssa} = {operation}(%{operand1}, %{operand2})")
                continue
                
            self.debug_print(f"  → UNMATCHED: {line}")
        
        self.debug_print(f"\nTotal SSA variables found: {len(self.ssa_variables)}")
        for ssa_name, var_info in self.ssa_variables.items():
            deps_str = ", ".join(var_info.dependencies) if var_info.dependencies else "none"
            self.debug_print(f"  {ssa_name}: {var_info.operation} (value={var_info.value}, deps=[{deps_str}])")
    
    def allocate_qubit_register(self, ssa_name: str) -> str:
        """Allocate a unique qubit register name"""
        if ssa_name in self.variable_to_qubits:
            return self.variable_to_qubits[ssa_name]
        
        qubit_name = f"%q{self.qubit_counter}"
        self.variable_to_qubits[ssa_name] = qubit_name
        self.qubit_counter += 1
        self.debug_print(f"Allocated {qubit_name} for {ssa_name}")
        return qubit_name
    
    def convert_to_gate_level(self) -> List[str]:
        """Convert SSA variables to gate-level operations"""
        self.debug_print("\n=== Converting to Gate-Level ===")
        
        gate_ops = []
        processed = set()
        
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
                qubit_reg = self.allocate_qubit_register(ssa_name)
                gate_ops.append(f"    {qubit_reg} = q.alloc : !qreg<4>")
                gate_ops.append(f"    q.init {qubit_reg}, {var_info.value} : i32")
                self.debug_print(f"  → Generated: alloc + init {qubit_reg} = {var_info.value}")
                
            elif var_info.operation == "measure":
                # Create measurement operation
                measured_var = var_info.dependencies[0]
                if measured_var in self.variable_to_qubits:
                    measured_reg = self.variable_to_qubits[measured_var]
                    result_reg = self.allocate_qubit_register(ssa_name)
                    gate_ops.append(f"    {result_reg} = q.measure {measured_reg} : !qreg -> i32")
                    self.debug_print(f"  → Generated: {result_reg} = measure({measured_reg})")
                else:
                    self.debug_print(f"  → ERROR: Cannot find qubit for measured variable {measured_var}")
                    
            elif var_info.operation in ["add", "sub", "mul", "div", "mod"]:
                # Create arithmetic circuit
                operand1, operand2 = var_info.dependencies
                if operand1 in self.variable_to_qubits and operand2 in self.variable_to_qubits:
                    reg1 = self.variable_to_qubits[operand1]
                    reg2 = self.variable_to_qubits[operand2]
                    result_reg = self.allocate_qubit_register(ssa_name)
                    
                    gate_ops.append(f"    {result_reg} = q.alloc : !qreg<4>")
                    gate_ops.append(f"    q.{var_info.operation}_circuit {reg1}, {reg2}, {result_reg}")
                    self.debug_print(f"  → Generated: {var_info.operation}_circuit({reg1}, {reg2}) -> {result_reg}")
                else:
                    self.debug_print(f"  → ERROR: Missing dependencies for {var_info.operation}")
                    
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
                    
                    gate_ops.append(f"    {orig_reg} = q.alloc : !qreg<4>")
                    gate_ops.append(f"    {new_reg} = q.alloc : !qreg<4>")
                    gate_ops.append(f"    q.{base_op}_circuit {operand_reg}, {orig_reg}, {new_reg}")
                    self.debug_print(f"  → Generated: {base_op}_circuit({operand_reg}) -> {orig_reg}, {new_reg}")
                    
                    # Mark both variables as processed and this multi-op as done
                    processed.add(new_var)
                    self.processed_multi_ops.add(multi_op_key)
                else:
                    self.debug_print(f"  → ERROR: Missing dependencies for {var_info.operation}")
                    
            elif var_info.operation.endswith("_new"):
                # Skip _new operations as they're handled by _orig operations
                pass
            
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
        
        # Step 3: Build final MLIR
        result = [
            "// Gate-Level Quantum MLIR",
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
    parser = argparse.ArgumentParser(description='Gate-Level MLIR Optimizer')
    parser.add_argument('input_file', help='Input high-level MLIR file')
    parser.add_argument('output_file', help='Output gate-level MLIR file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    if args.debug:
        print("Input MLIR content:")
        for i, line in enumerate(mlir_content.split('\n')):
            print(f"{i:2}: {line}")
        print()
    
    # Create optimizer and convert
    optimizer = GateLevelOptimizer(enable_debug=args.debug)
    gate_mlir = optimizer.generate_gate_mlir(mlir_content)
    
    # Write output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(gate_mlir)
        print(f"✅ Gate-level MLIR written to: {args.output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    
    if args.debug:
        print("\nGenerated gate-level MLIR:")
        print(gate_mlir)

if __name__ == "__main__":
    main()
