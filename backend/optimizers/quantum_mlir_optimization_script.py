


#!/usr/bin/env python3
"""
Quantum MLIR While-Loop Unroller

This script is specifically designed to optimize 'quantum.while' loops whose
conditions can be evaluated at compile time.

It performs a single optimization:
- It finds all constant values defined with 'quantum.init'.
- It evaluates the loop condition (e.g., 'quantum.gt').
- If the condition is a constant 'true', it replaces the loop with its body.
- If the condition is a constant 'false', it deletes the loop entirely.
- If the condition is not constant, it leaves the loop unchanged.
"""

import re
import sys
from typing import Dict, List, Optional

class WhileLoopUnroller:
    def __init__(self):
        self.pre_loop_lines: List[str] = []
        self.condition_lines: List[str] = []
        self.body_lines: List[str] = []
        self.post_loop_lines: List[str] = []
        self.constants: Dict[str, int] = {}
        self.header_lines: List[str] = []

    def _find_constants(self, lines: List[str]) -> None:
        """Parses quantum.init operations to find all known constants."""
        init_pattern = r'(%\w+)\s*=\s*"quantum\.init".*value\s*=\s*(\d+)\s*:'
        for line in lines:
            match = re.search(init_pattern, line)
            if match:
                register, value = match.groups()
                self.constants[register] = int(value)
        print(f"Found constants: {self.constants}")

    def parse_and_partition_mlir(self, content: str) -> bool:
        """
        Parses the MLIR file and partitions it into sections.
        Returns True if a while loop was found, False otherwise.
        """
        lines = content.split('\n')
        state = "PRE_LOOP"  # States: PRE_LOOP, CONDITION, BODY, POST_LOOP

        for line in lines:
            stripped_line = line.strip()

            if stripped_line.startswith(('builtin.module', '"quantum.func"')):
                self.header_lines.append(line)
                continue

            if state == "PRE_LOOP":
                if '"quantum.while"()' in stripped_line:
                    state = "CONDITION"
                else:
                    self.pre_loop_lines.append(line)
            elif state == "CONDITION":
                if '}, {' in stripped_line:
                    state = "BODY"
                else:
                    # Ignore the opening brace of the condition block
                    if stripped_line != '{':
                        self.condition_lines.append(line)
            elif state == "BODY":
                if '}) : () -> ()' in stripped_line:
                    state = "POST_LOOP"
                else:
                     # Ignore the opening brace of the body block
                    if stripped_line != '{':
                        self.body_lines.append(line)
            elif state == "POST_LOOP":
                self.post_loop_lines.append(line)
        
        return state != "PRE_LOOP"

    def _evaluate_condition(self) -> Optional[bool]:
        """
        Evaluates the loop condition if its inputs are constant.
        Returns True, False, or None (if not constant).
        """
        gt_pattern = r'"quantum\.gt"\((%\w+),\s*(%\w+)\)'
        
        for line in self.condition_lines:
            match = re.search(gt_pattern, line)
            if match:
                left_reg, right_reg = match.groups()
                left_val = self.constants.get(left_reg)
                right_val = self.constants.get(right_reg)
                
                if left_val is not None and right_val is not None:
                    print(f"Evaluating constant condition: {left_val} > {right_val}")
                    return left_val > right_val
        
        print("Condition could not be evaluated to a constant.")
        return None

    def optimize(self, mlir_content: str) -> str:
        """Runs the optimization to unroll a while loop."""
        print("🚀 Starting While-Loop Unroller...")
        
        if not self.parse_and_partition_mlir(mlir_content):
            print("No 'quantum.while' loop found. Returning original content.")
            return mlir_content

        self._find_constants(self.pre_loop_lines)
        condition_result = self._evaluate_condition()

        if condition_result is True:
            print("✅ Condition is constant TRUE. Unrolling loop body.")
            final_lines = self.header_lines + self.pre_loop_lines + self.body_lines + self.post_loop_lines
            return "\n".join(final_lines)
        elif condition_result is False:
            print("✅ Condition is constant FALSE. Deleting loop entirely.")
            final_lines = self.header_lines + self.pre_loop_lines + self.post_loop_lines
            return "\n".join(final_lines)
        else:
            print("⚠️ Condition is not constant. Loop cannot be unrolled.")
            return mlir_content

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_while.mlir> <output_optimized.mlir>")
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]

    try:
        with open(input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found.")
        sys.exit(1)

    optimizer = WhileLoopUnroller()
    optimized_mlir = optimizer.optimize(mlir_content)

    try:
        with open(output_file, 'w') as f:
            f.write(optimized_mlir)
        print(f"💾 Optimized MLIR written to: {output_file}")
    except IOError as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
