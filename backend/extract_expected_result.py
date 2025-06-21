#!/usr/bin/env python3
"""
Extract Expected Result from High-Level MLIR

This script extracts operands and operations from high-level MLIR,
calculates the correct expected result, and saves it to expected_res.txt
for use in the circuit generation pipeline.

Usage: python extract_expected_result.py <high_level.mlir> <expected_res.txt>
"""

import re
import sys
from typing import Dict, List, Tuple, Optional

class MLIRExpectedResultExtractor:
    def __init__(self, enable_debug=False):
        self.enable_debug = enable_debug
        self.variables = {}
        self.operations = []
        self.expected_result = None
        
    def debug_print(self, message: str):
        if self.enable_debug:
            print(f"[DEBUG] {message}")
    
    def extract_initial_values(self, mlir_content: str) -> Dict[str, int]:
        """Extract initial values from quantum.init operations"""
        variables = {}
        
        # Pattern to match quantum.init operations
        # %0 = "quantum.init"() {type = i32, value = 3 : i32} : () -> i32
        init_pattern = r'%(\d+)\s*=\s*"quantum\.init"\(\)\s*\{[^}]*value\s*=\s*(\d+)[^}]*\}'
        
        matches = re.findall(init_pattern, mlir_content)
        for var_id, value in matches:
            var_name = f"%{var_id}"
            variables[var_name] = int(value)
            self.debug_print(f"Found initial value: {var_name} = {value}")
        
        return variables
    
    def extract_operations(self, mlir_content: str) -> List[Dict]:
        """Extract quantum operations from MLIR"""
        operations = []
        
        # Patterns for different operation types
        patterns = {
            # Binary operations: %2 = "quantum.mul"(%0, %1) : (i32, i32) -> i32
            'binary': r'%(\d+)\s*=\s*"quantum\.(\w+)"\s*\(\s*%(\d+)\s*,\s*%(\d+)\s*\)',
            
            # Unary operations: %2 = "quantum.neg"(%0) : (i32) -> i32
            'unary': r'%(\d+)\s*=\s*"quantum\.(\w+)"\s*\(\s*%(\d+)\s*\)',
            
            # Multi-result operations: %3, %4 = "quantum.post_inc"(%2) : (i32) -> (i32, i32)
            'multi_result': r'%(\d+),\s*%(\d+)\s*=\s*"quantum\.(\w+)"\s*\(\s*%(\d+)\s*\)',
            
            # Measurement: %3 = "quantum.measure"(%2) : (i32) -> i1
            'measure': r'%(\d+)\s*=\s*"quantum\.measure"\s*\(\s*%(\d+)\s*\)'
        }
        
        lines = mlir_content.split('\n')
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # Check binary operations first (most common)
            match = re.search(patterns['binary'], line)
            if match:
                result_var, op_name, operand1, operand2 = match.groups()
                operations.append({
                    'type': 'binary',
                    'operation': op_name,
                    'result': f"%{result_var}",
                    'operands': [f"%{operand1}", f"%{operand2}"],
                    'line': line_num + 1
                })
                self.debug_print(f"Found binary op: {op_name}({operand1}, {operand2}) -> {result_var}")
                continue
            
            # Check multi-result operations
            match = re.search(patterns['multi_result'], line)
            if match:
                result1, result2, op_name, operand = match.groups()
                operations.append({
                    'type': 'multi_result',
                    'operation': op_name,
                    'results': [f"%{result1}", f"%{result2}"],
                    'operands': [f"%{operand}"],
                    'line': line_num + 1
                })
                self.debug_print(f"Found multi-result op: {op_name}({operand}) -> ({result1}, {result2})")
                continue
            
            # Check unary operations (excluding measure and init)
            match = re.search(patterns['unary'], line)
            if match:
                result_var, op_name, operand = match.groups()
                if op_name not in ['init', 'measure']:  # Skip init and measure
                    operations.append({
                        'type': 'unary',
                        'operation': op_name,
                        'result': f"%{result_var}",
                        'operands': [f"%{operand}"],
                        'line': line_num + 1
                    })
                    self.debug_print(f"Found unary op: {op_name}({operand}) -> {result_var}")
                continue
            
            # Check measurement operations
            match = re.search(patterns['measure'], line)
            if match:
                result_var, operand = match.groups()
                operations.append({
                    'type': 'measure',
                    'operation': 'measure',
                    'result': f"%{result_var}",
                    'operands': [f"%{operand}"],
                    'line': line_num + 1
                })
                self.debug_print(f"Found measurement: measure({operand}) -> {result_var}")
                continue
        
        return operations
    
    def calculate_expected_result(self, variables: Dict[str, int], operations: List[Dict]) -> int:
        """Calculate the expected result by simulating the operations"""
        self.debug_print("\n=== Calculating Expected Result ===")
        self.debug_print(f"Initial variables: {variables}")
        self.debug_print(f"Operations to simulate: {len(operations)}")
        
        # Create a working copy of variables
        working_vars = variables.copy()
        
        # Process each operation in order
        for i, op in enumerate(operations):
            self.debug_print(f"\nStep {i+1}: {op['operation']}")
            
            if op['type'] == 'binary':
                # Binary operations: result = operand1 OP operand2
                op1_var, op2_var = op['operands']
                result_var = op['result']
                
                if op1_var in working_vars and op2_var in working_vars:
                    val1, val2 = working_vars[op1_var], working_vars[op2_var]
                    
                    if op['operation'] == 'add':
                        result = (val1 + val2) & 0xF  # 4-bit result
                    elif op['operation'] == 'sub':
                        result = (val1 - val2) & 0xF
                    elif op['operation'] == 'mul':
                        result = (val1 * val2) & 0xF
                    elif op['operation'] == 'div':
                        result = (val1 // val2) & 0xF if val2 != 0 else 0
                    elif op['operation'] == 'mod':
                        result = (val1 % val2) & 0xF if val2 != 0 else 0
                    elif op['operation'] == 'and':
                        result = (val1 & val2) & 0xF  # Bitwise AND
                    elif op['operation'] == 'or':
                        result = (val1 | val2) & 0xF  # Bitwise OR
                    elif op['operation'] == 'xor':
                        result = (val1 ^ val2) & 0xF
                    elif op['operation'] == 'gt':
                        result = 1 if val1 > val2 else 0
                    elif op['operation'] == 'lt':
                        result = 1 if val1 < val2 else 0
                    elif op['operation'] == 'eq':
                        result = 1 if val1 == val2 else 0
                    elif op['operation'] == 'ne':
                        result = 1 if val1 != val2 else 0
                    elif op['operation'] == 'ge':
                        result = 1 if val1 >= val2 else 0
                    elif op['operation'] == 'le':
                        result = 1 if val1 <= val2 else 0
                    else:
                        self.debug_print(f"Unknown binary operation: {op['operation']}")
                        result = 0
                    
                    working_vars[result_var] = result
                    self.debug_print(f"  {op['operation']}: {val1} {op['operation']} {val2} = {result}")
                    self.debug_print(f"  Stored {result_var} = {result}")
                else:
                    self.debug_print(f"  Missing operands: {op1_var}={op1_var in working_vars}, {op2_var}={op2_var in working_vars}")
            
            elif op['type'] == 'unary':
                # Unary operations: result = OP(operand)
                operand_var = op['operands'][0]
                result_var = op['result']
                
                if operand_var in working_vars:
                    val = working_vars[operand_var]
                    
                    if op['operation'] == 'neg':
                        result = (-val) & 0xF  # 4-bit negation
                    elif op['operation'] == 'not':
                        result = (~val) & 0xF  # Bitwise NOT
                    elif op['operation'] == 'pre_inc':
                        result = (val + 1) & 0xF
                    elif op['operation'] == 'pre_dec':
                        result = (val - 1) & 0xF
                    else:
                        self.debug_print(f"Unknown unary operation: {op['operation']}")
                        result = val
                    
                    working_vars[result_var] = result
                    self.debug_print(f"  {op['operation']}: {op['operation']}({val}) = {result}")
                    self.debug_print(f"  Stored {result_var} = {result}")
                else:
                    self.debug_print(f"  Missing operand: {operand_var}")
            
            elif op['type'] == 'multi_result':
                # Multi-result operations (post_inc, post_dec)
                operand_var = op['operands'][0]
                result1_var, result2_var = op['results']
                
                if operand_var in working_vars:
                    val = working_vars[operand_var]
                    
                    if op['operation'] == 'post_inc':
                        # result1 = original value, result2 = incremented value
                        working_vars[result1_var] = val
                        working_vars[result2_var] = (val + 1) & 0xF
                        self.debug_print(f"  post_inc: {val} -> orig={val}, inc={val+1}")
                    elif op['operation'] == 'post_dec':
                        # result1 = original value, result2 = decremented value
                        working_vars[result1_var] = val
                        working_vars[result2_var] = (val - 1) & 0xF
                        self.debug_print(f"  post_dec: {val} -> orig={val}, dec={val-1}")
                    
                    self.debug_print(f"  Stored {result1_var} = {working_vars[result1_var]}, {result2_var} = {working_vars[result2_var]}")
                else:
                    self.debug_print(f"  Missing operand: {operand_var}")
            
            elif op['type'] == 'measure':
                # Measurement - this tells us what the final result should be
                operand_var = op['operands'][0]
                if operand_var in working_vars:
                    final_result = working_vars[operand_var]
                    self.debug_print(f"  Measurement of {operand_var} = {final_result}")
                    return final_result
        
        # If no measurement found, return the last computed value
        if working_vars:
            # Find the highest numbered variable (likely the final result)
            last_var = max(working_vars.keys(), key=lambda x: int(x[1:]))
            final_result = working_vars[last_var]
            self.debug_print(f"No measurement found, using last variable {last_var} = {final_result}")
            return final_result
        
        self.debug_print("No result could be determined")
        return 0
    
    def extract_and_calculate(self, mlir_content: str) -> int:
        """Main function to extract and calculate expected result"""
        print("🔍 Extracting expected result from high-level MLIR...")
        
        # Extract initial values
        variables = self.extract_initial_values(mlir_content)
        print(f"📊 Found {len(variables)} initial variables: {variables}")
        
        # Extract operations
        operations = self.extract_operations(mlir_content)
        print(f"⚙️  Found {len(operations)} operations")
        
        if self.enable_debug:
            for i, op in enumerate(operations):
                print(f"  {i+1}. {op['operation']} (line {op['line']})")
        
        # Calculate expected result
        if variables and operations:
            expected = self.calculate_expected_result(variables, operations)
            print(f"🎯 Expected result: {expected}")
            return expected
        elif variables and not operations:
            # Simple case: just initial values, might be a single variable
            if len(variables) == 1:
                result = list(variables.values())[0]
                print(f"🎯 Single variable result: {result}")
                return result
        
        print("❌ Could not determine expected result")
        return 0
    
    def save_expected_result(self, expected_result: int, output_file: str) -> None:
        """Save the expected result to a file"""
        try:
            with open(output_file, 'w') as f:
                f.write(f"{expected_result}\n")
            print(f"💾 Saved expected result to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving expected result: {e}")
            sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_expected_result.py <high_level.mlir> <expected_res.txt>")
        print("\nExtracts the correct expected result from high-level MLIR")
        print("and saves it for use in the circuit generation pipeline.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read input MLIR file
    try:
        with open(input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"📄 Reading MLIR from: {input_file}")
    
    # Extract and calculate expected result
    extractor = MLIRExpectedResultExtractor(enable_debug=False)
    expected_result = extractor.extract_and_calculate(mlir_content)
    
    # Save to output file
    extractor.save_expected_result(expected_result, output_file)
    
    print(f"✅ Expected result extraction complete!")
    print(f"   Input:  {input_file}")
    print(f"   Output: {output_file}")
    print(f"   Result: {expected_result}")

if __name__ == "__main__":
    main()
