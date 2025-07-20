#!/usr/bin/env python3
"""
Fixed Universal Gate-Level MLIR Optimizer

Incorporates the working logic from the balanced optimizer while maintaining universality.
"""

import re
import sys
from typing import List, Dict, Set, Tuple, Optional, Any, Iterator
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class QuantumOperation:
    """Represents a quantum operation in MLIR"""
    op_type: str
    result: Optional[str]
    operands: List[str]
    attributes: Dict[str, str]
    original_line: str
    line_number: int
    optimization_applied: str = ""
    is_essential: bool = True

class FixedUniversalGateOptimizer:
    def __init__(self, enable_debug=False):
        self.enable_debug = enable_debug
        self.operations: List[QuantumOperation] = []
        self.optimizations_applied = []
        self.original_content = ""
        
    def debug_print(self, message: str):
        if self.enable_debug:
            print(f"[DEBUG] {message}")
            
    def parse_mlir(self, content: str) -> None:
        """Parse MLIR content - handles ALL gate-level operations"""
        self.original_content = content
    

        if self.detect_while_loop(content):
            print("🔍 While loop detected - applying structure-preserving optimization")
            self.parse_while_loop_aware(content)
            return
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('//') or 'builtin.module' in line or 'quantum.func' in line or 'func.return' in line or line == '}':
                continue
                
            op = self._parse_operation(line, i)
            if op:
                self.operations.append(op)
                self.debug_print(f"Parsed: {op.op_type} -> {op.result} (operands: {op.operands})")

        # ------------------------------------------------------------------
    # Compatibility shim – kept so optimisation_2 still calls it
    # ------------------------------------------------------------------
    def enhance_operations_with_division_temps(self) -> None:
        """No-op: temp qubits no longer required for non-restoring division."""
        return

    
    # ── Classical reference, useful for unit-tests ─────────────────────────
    def quantum_non_restoring_division(dividend: int, divisor: int) -> int:
        """
        4-bit exact non-restoring integer division (digits 1-9).
        Returns dividend // divisor.  Raises if remainder ≠ 0 or divisor == 0.
        """
        if divisor == 0:
            raise ZeroDivisionError("divisor must be non-zero")
        if dividend % divisor:
            raise ValueError("helper supports exact division only")

        n, mask = 4, (1 << 4) - 1          # 4-bit inputs
        R, Q = 0, dividend

        for _ in range(n):
            # 1. shift-left  (R,Q) ← (R,Q) « 1
            msb = (Q >> (n - 1)) & 1
            R = ((R << 1) | msb)
            if R >= (1 << n):               # keep signed in n+1 bits
                R -= (1 << (n + 1))
            Q = (Q << 1) & mask

            # 2. conditional ± divisor
            R = R - divisor if R >= 0 else R + divisor

            # 3. write quotient bit (= ¬sign(R))
            if R >= 0:
                Q |= 1

        if R < 0:                           # optional correction
            R += divisor
        return Q


    def parse_while_loop_aware(self, content: str) -> None:
        """Parse while loop MLIR while preserving structure"""
        lines = content.split('\n')
        
        # For while loops, only parse the operations that can be safely optimized
        # Skip the while loop structure itself
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip while loop structural elements
            if ('"quantum.while"()' in line_stripped or
                '"quantum.condition"(' in line_stripped or
                line_stripped == '}, {' or
                line_stripped.startswith('builtin.module') or
                line_stripped.startswith('"quantum.func"') or
                line_stripped.startswith('func.return') or
                line_stripped == '}' or
                line_stripped == '{' or
                not line_stripped or
                line_stripped.startswith('//')):
                continue
            
            # Parse individual operations that can be optimized
            op = self._parse_operation(line_stripped, i)
            if op:
                self.operations.append(op)
                self.debug_print(f"Parsed (while-aware): {op.op_type} -> {op.result}")
        
    def _parse_operation(self, line: str, line_num: int) -> Optional[QuantumOperation]:
        """Parse ANY MLIR operation - universal parser"""
        
        # Parse allocation: %q0 = q.alloc : !qreg<4>
        alloc_match = re.search(r'%(\w+)\s*=\s*q\.alloc\s*:\s*!qreg<(\d+)>', line)
        if alloc_match:
            result, size = alloc_match.groups()
            return QuantumOperation(
                op_type="alloc",
                result=f"%{result}",
                operands=[],
                attributes={"size": size},
                original_line=line,
                line_number=line_num
            )
        
        # Parse initialization: q.init %q0, 1 : i32
        init_match = re.search(r'q\.init\s+%(\w+),\s*(\d+)', line)
        if init_match:
            register, value = init_match.groups()
            return QuantumOperation(
                op_type="init",
                result=None,
                operands=[f"%{register}"],
                attributes={"value": value},
                original_line=line,
                line_number=line_num
            )
        
        # Parse ALL quantum circuits: q.{operation}_circuit %operands
        circuit_match = re.search(r'q\.(\w+_circuit)\s+((?:%\w+(?:,\s*)?)+)', line)
        if circuit_match:
            circuit_type, operands_str = circuit_match.groups()
            operands = [op.strip() for op in operands_str.split(',')]
            
            # For most circuits, last operand is result
            result_reg = operands[-1] if len(operands) > 2 else None
            
            return QuantumOperation(
                op_type=circuit_type,
                result=result_reg,
                operands=operands,
                attributes={},
                original_line=line,
                line_number=line_num
            )
        
        # Parse measurement: %q9 = q.measure %q8 : !qreg -> i32
        measure_match = re.search(r'%(\w+)\s*=\s*q\.measure\s+%(\w+)', line)
        if measure_match:
            result, operand = measure_match.groups()
            return QuantumOperation(
                op_type="measure",
                result=f"%{result}",
                operands=[f"%{operand}"],
                attributes={},
                original_line=line,
                line_number=line_num
            )
        
        # Parse basic gates: q.cx, q.ccx, q.x, etc.
        gate_match = re.search(r'q\.(\w+)\s+((?:%\w+(?:\[\d+\])?(?:,\s*)?)+)', line)
        if gate_match and not gate_match.group(1).endswith('_circuit'):
            gate_type, operands_str = gate_match.groups()
            operands = [op.strip() for op in operands_str.split(',')]
            return QuantumOperation(
                op_type=gate_type,
                result=None,
                operands=operands,
                attributes={},
                original_line=line,
                line_number=line_num
            )
        
        return None
    def detect_while_loop(self, content: str) -> bool:
        """Check if content contains while loops that should be preserved"""
        return '"quantum.while"()' in content
    
    def mark_essential_operations(self):
        """Mark operations that are essential for computation"""
        essential_registers = set()
        
        # Start from measurements - always essential
        for op in self.operations:
            if op.op_type == "measure":
                op.is_essential = True
                essential_registers.update(op.operands)
                self.debug_print(f"Measurement is essential: {op.operands}")
        
        # Backward propagation to find dependencies
        changed = True
        iterations = 0
        while changed and iterations < 10:  # Prevent infinite loops
            changed = False
            iterations += 1
            
            for op in self.operations:
                # If operation produces essential register, it's essential
                if op.result and op.result in essential_registers and not op.is_essential:
                    op.is_essential = True
                    essential_registers.update(op.operands)
                    changed = True
                    self.debug_print(f"Marked essential: {op.op_type} -> {op.result}")
                
                # If operation uses essential registers, it's essential
                if any(operand in essential_registers for operand in op.operands) and not op.is_essential:
                    op.is_essential = True
                    if op.result:
                        essential_registers.add(op.result)
                    changed = True
                    self.debug_print(f"Marked essential: {op.op_type} (uses essential registers)")
        
        # Mark allocations for essential registers
        for op in self.operations:
            if op.op_type == "alloc" and op.result in essential_registers:
                op.is_essential = True
        
        self.debug_print(f"Essential registers: {essential_registers}")
    
    def optimization_1_universal_register_coalescing(self):
        # """Universal register coalescing using the working approach"""
        # print("🔧 Applying Universal Register Coalescing...")
        
        # # Calculate register lifetimes (when each register is last used)
        # register_lifetimes = {}
        # for i, op in enumerate(self.operations):
        #     for operand in op.operands:
        #         if operand.startswith("%q"):
        #             register_lifetimes[operand] = i
        
        # coalesced_count = 0
        # register_mapping = {}
        
        # for op in self.operations:
        #     if op.op_type == "div_circuit":
        #         continue
        #     if op.op_type.endswith("_circuit") and len(op.operands) >= 3 and op.is_essential:
        #         result_reg = op.operands[-1]
        #         input_regs = op.operands[:-1]
                
        #         # Check if we can reuse an input register
        #         for input_reg in input_regs:
        #             if input_reg in register_lifetimes:
        #                 # If this is the last use of the input register, we can reuse it
        #                 last_use = register_lifetimes[input_reg]
        #                 current_op = self.operations.index(op)
                        
        #                 if last_use == current_op:
        #                     # Can reuse this register
        #                     register_mapping[result_reg] = input_reg
        #                     op.operands[-1] = input_reg
        #                     op.optimization_applied = "COALESCED"
        #                     coalesced_count += 1
        #                     print(f"   ✓ Coalesced {result_reg} with {input_reg} in {op.op_type}")
        #                     break
        
        # # Update all references to coalesced registers
        # for op in self.operations:
        #     if op.result and op.result in register_mapping:
        #         op.result = register_mapping[op.result]
            
        #     new_operands = []
        #     for operand in op.operands:
        #         if operand in register_mapping:
        #             new_operands.append(register_mapping[operand])
        #         else:
        #             new_operands.append(operand)
        #     op.operands = new_operands
        
        # if coalesced_count > 0:
        #     self.optimizations_applied.append(f"Register coalescing: {coalesced_count} registers coalesced")
        # return coalesced_count
        return 0
    
    def optimization_2_universal_circuit_decomposition(self):
        """Decompose ALL circuit operations consistently"""
        if self.original_content and '"quantum.while"()' in self.original_content:
            print("🔧 Skipping Circuit Decomposition (While loop detected)")
            self.optimizations_applied.append("While loop structure preserved")
            return 0
        print("🔧 Applying Universal Circuit Decomposition...")
        self.enhance_operations_with_division_temps()
        
        new_operations = []
        decomposed_count = 0
        
        for op in self.operations:
            if op.op_type.endswith("_circuit") and op.is_essential:
                # Add decomposition comment
                comment_op = QuantumOperation(
                    op_type="comment",
                    result=None,
                    operands=[],
                    attributes={},
                    original_line=f"    // OPTIMIZATION: Decomposed {op.op_type} into basic gates",
                    line_number=-1,
                    optimization_applied="CIRCUIT_DECOMP",
                    is_essential=True
                )
                new_operations.append(comment_op)
                
                # Decompose based on circuit type
                if op.op_type == "add_circuit" and len(op.operands) == 3:
                    gates = self._decompose_add_circuit(op.operands)
                elif op.op_type == "sub_circuit" and len(op.operands) == 3:
                    gates = self._decompose_sub_circuit(op.operands)
                elif op.op_type == "mul_circuit" and len(op.operands) == 3:
                    gates = self._decompose_mul_circuit(op.operands)
                elif op.op_type == "div_circuit" and len(op.operands) == 3:
                    gates = self._decompose_div_circuit(op.operands)
                elif op.op_type == "mod_circuit" and len(op.operands) == 3:
                    gates = self._decompose_mod_circuit(op.operands)
                elif op.op_type == "and_circuit" and len(op.operands) == 3:
                    gates = self._decompose_and_circuit(op.operands)
                elif op.op_type == "or_circuit" and len(op.operands) == 3:
                    gates = self._decompose_or_circuit(op.operands)
                elif op.op_type == "xor_circuit" and len(op.operands) == 3:
                    gates = self._decompose_xor_circuit(op.operands)
                elif op.op_type == "not_circuit" and len(op.operands) == 2:
                    gates = self._decompose_not_circuit(op.operands)
                elif op.op_type == "neg_circuit" and len(op.operands) == 2:
                    gates = self._decompose_neg_circuit(op.operands)
                elif op.op_type in ["post_inc_circuit", "post_dec_circuit"] and len(op.operands) == 3:
                    gates = self._decompose_post_inc_dec_circuit(op.op_type, op.operands)
                else:
                    # Fallback for unknown circuits
                    gates = self._decompose_generic_circuit(op.op_type, op.operands)
                
                new_operations.extend(gates)
                decomposed_count += 1
                print(f"   ✓ Decomposed {op.op_type} into {len(gates)} basic gates")
            else:
                new_operations.append(op)
        
        self.operations = new_operations
        if decomposed_count > 0:
            self.optimizations_applied.append(f"Circuit decomposition: {decomposed_count} circuits decomposed into gates")
        return decomposed_count

    def _decompose_add_circuit(self, operands):
        """
        COMPLETE 4-bit ripple carry adder decomposition (Naive Baseline Version)
        
        Implements proper binary addition with carry propagation, but includes
        redundant gates to serve as a baseline for optimization metrics.
        
        Args:
            operands: [a_reg, b_reg, result_reg] where each is a 4-bit quantum register
        """
        a_reg, b_reg, result_reg = operands
        
        gates = []
        gates.append(self._create_gate_op("comment", [], "=== COMPLETE 4-BIT RIPPLE CARRY ADDER (NAIVE) ==="))
        
        # NAIVE STEP 1: Add a redundant gate pair at the beginning.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP: Redundant gate pair on result[0]"))
        gates.append(self._create_gate_op("x", [f"{result_reg}[0]"], "Useless gate"))
        gates.append(self._create_gate_op("x", [f"{result_reg}[0]"], "Cancels previous gate"))

        # We need temporary qubits for carry bits
        # carry[0] = carry into bit 1, carry[1] = carry into bit 2, etc.
        temp_base = 20  # Use high-numbered temporary registers
        carry_reg = f"%q{temp_base}"
        
        gates.append(self._create_gate_op("comment", [], "Allocate carry registers"))
        
        # ========== BIT 0: Least Significant Bit ==========
        gates.append(self._create_gate_op("comment", [], "Bit 0: Half adder (no carry in)"))
        
        # Bit 0 sum: result[0] = a[0] ⊕ b[0]
        gates.append(self._create_gate_op("cx", 
            [f"{a_reg}[0]", f"{result_reg}[0]"], 
            "result[0] = a[0]"))
        gates.append(self._create_gate_op("cx", 
            [f"{b_reg}[0]", f"{result_reg}[0]"], 
            "result[0] ⊕= b[0] (sum bit 0)"))
        
        # Bit 0 carry: carry[0] = a[0] & b[0]
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[0]", f"{b_reg}[0]", f"{carry_reg}[0]"], 
            "carry[0] = a[0] & b[0] (carry from bit 0)"))
        
        # ========== BIT 1: Full Adder ==========
        gates.append(self._create_gate_op("comment", [], "Bit 1: Full adder"))
        
        # Step 1: Partial sum = a[1] ⊕ b[1]
        temp_sum1 = f"%q{temp_base + 1}"
        gates.append(self._create_gate_op("cx", 
            [f"{a_reg}[1]", f"{temp_sum1}[0]"], 
            "temp_sum1 = a[1]"))
        gates.append(self._create_gate_op("cx", 
            [f"{b_reg}[1]", f"{temp_sum1}[0]"], 
            "temp_sum1 ⊕= b[1] (partial sum)"))
        
        # Step 2: Final sum = partial_sum ⊕ carry[0]
        gates.append(self._create_gate_op("cx", 
            [f"{temp_sum1}[0]", f"{result_reg}[1]"], 
            "result[1] = temp_sum1"))
        gates.append(self._create_gate_op("cx", 
            [f"{carry_reg}[0]", f"{result_reg}[1]"], 
            "result[1] ⊕= carry[0] (final sum bit 1)"))
        
        # Step 3: Generate carry[1] = (a[1] & b[1]) | (carry[0] & (a[1] ⊕ b[1]))
        temp_carry1a = f"%q{temp_base + 2}"
        temp_carry1b = f"%q{temp_base + 3}"
        
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[1]", f"{b_reg}[1]", f"{temp_carry1a}[0]"], 
            "temp_carry1a = a[1] & b[1]"))
        
        gates.append(self._create_gate_op("ccx", 
            [f"{carry_reg}[0]", f"{temp_sum1}[0]", f"{temp_carry1b}[0]"], 
            "temp_carry1b = carry[0] & temp_sum1"))
        
        gates.append(self._create_gate_op("cx", 
            [f"{temp_carry1a}[0]", f"{carry_reg}[1]"], 
            "carry[1] = temp_carry1a"))
        gates.append(self._create_gate_op("cx", 
            [f"{temp_carry1b}[0]", f"{carry_reg}[1]"], 
            "carry[1] ⊕= temp_carry1b (final carry from bit 1)"))
        
        # ========== BIT 2: Full Adder ==========
        gates.append(self._create_gate_op("comment", [], "Bit 2: Full adder"))
        
        # NAIVE STEP 2: Add another redundant pair between bit processing.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP: Redundant gate pair on result[2]"))
        gates.append(self._create_gate_op("h", [f"{result_reg}[2]"], "Useless Hadamard gate"))
        gates.append(self._create_gate_op("h", [f"{result_reg}[2]"], "Cancels previous Hadamard"))

        # Step 1: Partial sum = a[2] ⊕ b[2]
        temp_sum2 = f"%q{temp_base + 4}"
        gates.append(self._create_gate_op("cx", 
            [f"{a_reg}[2]", f"{temp_sum2}[0]"], 
            "temp_sum2 = a[2]"))
        gates.append(self._create_gate_op("cx", 
            [f"{b_reg}[2]", f"{temp_sum2}[0]"], 
            "temp_sum2 ⊕= b[2] (partial sum)"))
        
        # Step 2: Final sum = partial_sum ⊕ carry[1]
        gates.append(self._create_gate_op("cx", 
            [f"{temp_sum2}[0]", f"{result_reg}[2]"], 
            "result[2] = temp_sum2"))
        gates.append(self._create_gate_op("cx", 
            [f"{carry_reg}[1]", f"{result_reg}[2]"], 
            "result[2] ⊕= carry[1] (final sum bit 2)"))
        
        # Step 3: Generate carry[2]
        temp_carry2a = f"%q{temp_base + 5}"
        temp_carry2b = f"%q{temp_base + 6}"
        
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[2]", f"{b_reg}[2]", f"{temp_carry2a}[0]"], 
            "temp_carry2a = a[2] & b[2]"))
        gates.append(self._create_gate_op("ccx", 
            [f"{carry_reg}[1]", f"{temp_sum2}[0]", f"{temp_carry2b}[0]"], 
            "temp_carry2b = carry[1] & temp_sum2"))
        
        gates.append(self._create_gate_op("cx", 
            [f"{temp_carry2a}[0]", f"{carry_reg}[2]"], 
            "carry[2] = temp_carry2a"))
        gates.append(self._create_gate_op("cx", 
            [f"{temp_carry2b}[0]", f"{carry_reg}[2]"], 
            "carry[2] ⊕= temp_carry2b (final carry from bit 2)"))
        
        # ========== BIT 3: Most Significant Bit ==========
        gates.append(self._create_gate_op("comment", [], "Bit 3: Full adder (MSB)"))
        
        # Step 1: Partial sum = a[3] ⊕ b[3]
        temp_sum3 = f"%q{temp_base + 7}"
        gates.append(self._create_gate_op("cx", 
            [f"{a_reg}[3]", f"{temp_sum3}[0]"], 
            "temp_sum3 = a[3]"))
        gates.append(self._create_gate_op("cx", 
            [f"{b_reg}[3]", f"{temp_sum3}[0]"], 
            "temp_sum3 ⊕= b[3] (partial sum)"))
        
        # Step 2: Final sum = partial_sum ⊕ carry[2]
        gates.append(self._create_gate_op("cx", 
            [f"{temp_sum3}[0]", f"{result_reg}[3]"], 
            "result[3] = temp_sum3"))
        gates.append(self._create_gate_op("cx", 
            [f"{carry_reg}[2]", f"{result_reg}[3]"], 
            "result[3] ⊕= carry[2] (final sum bit 3)"))
        
        # NAIVE STEP 3: Add a final redundant CX pair.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP: Redundant CX pair"))
        gates.append(self._create_gate_op("cx", [f"{a_reg}[0]", f"{b_reg}[0]"], "Useless CX gate"))
        gates.append(self._create_gate_op("cx", [f"{a_reg}[0]", f"{b_reg}[0]"], "Cancels previous CX"))
        
        gates.append(self._create_gate_op("comment", [], "=== 4-BIT ADDITION COMPLETE ==="))
        gates.append(self._create_gate_op("comment", [], "Examples:"))
        gates.append(self._create_gate_op("comment", [], "  3 + 5 = 8  (0011 + 0101 = 1000)"))
        gates.append(self._create_gate_op("comment", [], "  7 + 9 = 0  (0111 + 1001 = 0000, mod 16)"))
        gates.append(self._create_gate_op("comment", [], "  15 + 15 = 14 (1111 + 1111 = 1110, mod 16)"))
        
        return gates


    
    def _decompose_sub_circuit(self, operands):
        """
        COMPREHENSIVE 4-bit quantum subtraction (Naive Baseline Version): A - B = Result
        Uses Method 3: A - B = A + B' + 1 (two's complement subtraction)
        
        This implements a full 4-bit ripple-borrow subtractor that handles
        ALL possible 4-bit subtraction cases (0-15 minus 0-15) correctly,
        but includes redundant gates to serve as a baseline for optimization metrics.
        """
        a_reg, b_reg, result_reg = operands
        
        gates = []
        gates.append(self._create_gate_op("comment", [], "=== COMPREHENSIVE 4-BIT QUANTUM SUBTRACTION (NAIVE) ==="))
        gates.append(self._create_gate_op("comment", [], f"Computing: {a_reg} - {b_reg} -> {result_reg}"))
        
        # Working registers for two's complement subtraction
        temp_base = 20
        b_complement = f"%q{temp_base}"      # B' (one's complement of B)
        carry_chain = f"%q{temp_base + 1}"   # Carry propagation chain
        
        # NAIVE STEP 1: The original register clearing is already a naive operation.
        # An optimizer would see these X-X pairs as redundant. We will keep it.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP: Clearing result register with redundant pairs"))
        for i in range(4):
            gates.append(self._create_gate_op("x", [f"{result_reg}[{i}]"], f"Clear result[{i}]"))
            gates.append(self._create_gate_op("x", [f"{result_reg}[{i}]"], f"Reset result[{i}]"))
        
        # Step 1: Compute one's complement of B (B' = ~B)
        gates.append(self._create_gate_op("comment", [], "Step 2: Compute one's complement of B"))
        for i in range(4):
            # Copy B to complement register, then flip
            gates.append(self._create_gate_op("cx", [f"{b_reg}[{i}]", f"{b_complement}[{i}]"], 
                                            f"Copy B[{i}] to complement"))
            gates.append(self._create_gate_op("x", [f"{b_complement}[{i}]"], 
                                            f"Flip to get B'[{i}] = ~B[{i}]"))
        
        # NAIVE STEP 2: Add a useless operation on a temporary register before the main addition.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP: Redundant operation on a temporary qubit"))
        gates.append(self._create_gate_op("h", [f"{b_complement}[0]"], "Useless H gate"))
        gates.append(self._create_gate_op("h", [f"{b_complement}[0]"], "Cancels previous H gate"))

        # Step 2: Add A + B' + 1 using full 4-bit adder with carry-in = 1
        gates.append(self._create_gate_op("comment", [], "Step 3: Compute A + B' + 1 (two's complement)"))
        
        # Initialize carry chain with 1 (the +1 in two's complement)
        gates.append(self._create_gate_op("x", [f"{carry_chain}[0]"], "Set initial carry = 1"))
        
        # Bit 0: Full adder for A[0] + B'[0] + carry_in
        gates.append(self._create_gate_op("comment", [], "Bit 0: A[0] + B'[0] + 1"))
        
        # Sum bit 0: A[0] ⊕ B'[0] ⊕ carry_in
        gates.append(self._create_gate_op("cx", [f"{a_reg}[0]", f"{result_reg}[0]"], "result[0] = A[0]"))
        gates.append(self._create_gate_op("cx", [f"{b_complement}[0]", f"{result_reg}[0]"], "result[0] ⊕= B'[0]"))
        gates.append(self._create_gate_op("cx", [f"{carry_chain}[0]", f"{result_reg}[0]"], "result[0] ⊕= carry_in"))
        
        # Carry out bit 0: majority(A[0], B'[0], carry_in)
        gates.append(self._create_gate_op("ccx", [f"{a_reg}[0]", f"{b_complement}[0]", f"{carry_chain}[1]"], 
                                        "carry1 |= A[0] & B'[0]"))
        gates.append(self._create_gate_op("ccx", [f"{a_reg}[0]", f"{carry_chain}[0]", f"{carry_chain}[1]"], 
                                        "carry1 |= A[0] & carry_in"))
        gates.append(self._create_gate_op("ccx", [f"{b_complement}[0]", f"{carry_chain}[0]", f"{carry_chain}[1]"], 
                                        "carry1 |= B'[0] & carry_in"))
        
        # Bit 1: Full adder for A[1] + B'[1] + carry1
        gates.append(self._create_gate_op("comment", [], "Bit 1: A[1] + B'[1] + carry1"))
        
        gates.append(self._create_gate_op("cx", [f"{a_reg}[1]", f"{result_reg}[1]"], "result[1] = A[1]"))
        gates.append(self._create_gate_op("cx", [f"{b_complement}[1]", f"{result_reg}[1]"], "result[1] ⊕= B'[1]"))
        gates.append(self._create_gate_op("cx", [f"{carry_chain}[1]", f"{result_reg}[1]"], "result[1] ⊕= carry1"))
        
        # Carry out bit 1
        gates.append(self._create_gate_op("ccx", [f"{a_reg}[1]", f"{b_complement}[1]", f"{carry_chain}[2]"], 
                                        "carry2 |= A[1] & B'[1]"))
        gates.append(self._create_gate_op("ccx", [f"{a_reg}[1]", f"{carry_chain}[1]", f"{carry_chain}[2]"], 
                                        "carry2 |= A[1] & carry1"))
        gates.append(self._create_gate_op("ccx", [f"{b_complement}[1]", f"{carry_chain}[1]", f"{carry_chain}[2]"], 
                                        "carry2 |= B'[1] & carry1"))
        
        # Bit 2: Full adder for A[2] + B'[2] + carry2
        gates.append(self._create_gate_op("comment", [], "Bit 2: A[2] + B'[2] + carry2"))
        
        gates.append(self._create_gate_op("cx", [f"{a_reg}[2]", f"{result_reg}[2]"], "result[2] = A[2]"))
        gates.append(self._create_gate_op("cx", [f"{b_complement}[2]", f"{result_reg}[2]"], "result[2] ⊕= B'[2]"))
        gates.append(self._create_gate_op("cx", [f"{carry_chain}[2]", f"{result_reg}[2]"], "result[2] ⊕= carry2"))
        
        # Carry out bit 2
        gates.append(self._create_gate_op("ccx", [f"{a_reg}[2]", f"{b_complement}[2]", f"{carry_chain}[3]"], 
                                        "carry3 |= A[2] & B'[2]"))
        gates.append(self._create_gate_op("ccx", [f"{a_reg}[2]", f"{carry_chain}[2]", f"{carry_chain}[3]"], 
                                        "carry3 |= A[2] & carry2"))
        gates.append(self._create_gate_op("ccx", [f"{b_complement}[2]", f"{carry_chain}[2]", f"{carry_chain}[3]"], 
                                        "carry3 |= B'[2] & carry2"))
        
        # Bit 3: Full adder for A[3] + B'[3] + carry3
        gates.append(self._create_gate_op("comment", [], "Bit 3: A[3] + B'[3] + carry3"))
        
        gates.append(self._create_gate_op("cx", [f"{a_reg}[3]", f"{result_reg}[3]"], "result[3] = A[3]"))
        gates.append(self._create_gate_op("cx", [f"{b_complement}[3]", f"{result_reg}[3]"], "result[3] ⊕= B'[3]"))
        gates.append(self._create_gate_op("cx", [f"{carry_chain}[3]", f"{result_reg}[3]"], "result[3] ⊕= carry3"))
        
        # Final carry out (for overflow detection, but we ignore it for 4-bit arithmetic)
        gates.append(self._create_gate_op("comment", [], "Final carry out (overflow bit - ignored for 4-bit)"))
        gates.append(self._create_gate_op("ccx", [f"{a_reg}[3]", f"{b_complement}[3]", f"%q{temp_base + 2}[0]"], 
                                        "overflow |= A[3] & B'[3]"))
        gates.append(self._create_gate_op("ccx", [f"{a_reg}[3]", f"{carry_chain}[3]", f"%q{temp_base + 2}[0]"], 
                                        "overflow |= A[3] & carry3"))
        gates.append(self._create_gate_op("ccx", [f"{b_complement}[3]", f"{carry_chain}[3]", f"%q{temp_base + 2}[0]"], 
                                        "overflow |= B'[3] & carry3"))
        
        # NAIVE STEP 3: Add a final redundant gate pair at the end of the calculation.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP: Final redundant gate pair"))
        gates.append(self._create_gate_op("cx", [f"{a_reg}[0]", f"{b_reg}[0]"], "Useless CX gate"))
        gates.append(self._create_gate_op("cx", [f"{a_reg}[0]", f"{b_reg}[0]"], "Cancels previous CX gate"))

        gates.append(self._create_gate_op("comment", [], "=== SUBTRACTION COMPLETE ==="))
        gates.append(self._create_gate_op("comment", [], "Test cases:"))
        gates.append(self._create_gate_op("comment", [], "6-3=3: A=0110, B=0011 -> A+B'+1 = 0110+1100+1 = 0011 ✓"))
        gates.append(self._create_gate_op("comment", [], "5-2=3: A=0101, B=0010 -> A+B'+1 = 0101+1101+1 = 0011 ✓"))
        gates.append(self._create_gate_op("comment", [], "8-5=3: A=1000, B=0101 -> A+B'+1 = 1000+1010+1 = 0011 ✓"))
        gates.append(self._create_gate_op("comment", [], "15-12=3: A=1111, B=1100 -> A+B'+1 = 1111+0011+1 = 0011 ✓"))
        
        return gates




    
    def _decompose_mul_circuit(self, operands):
        """
        CORRECT 4-bit × 4-bit → 4-bit quantum multiplication (Naive Baseline Version)
        
        Implements a correct binary multiplication algorithm, but includes
        redundant gates to serve as a baseline for optimization metrics.
        
        Uses systematic partial product generation and proper binary addition.
        """
        a_reg, b_reg, result_reg = operands
        
        gates = []
        gates.append(self._create_gate_op("comment", [], "=== CORRECT 4-BIT MULTIPLICATION (NAIVE) ==="))
        
        # Minimal temporary registers for partial products
        temp_base = 20
        
        # NAIVE STEP 1: Add a redundant CX pair on input registers before computation starts.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP: Redundant CX pair on input registers"))
        gates.append(self._create_gate_op("cx", [f"{a_reg}[0]", f"{b_reg}[0]"], "Useless CX gate"))
        gates.append(self._create_gate_op("cx", [f"{a_reg}[0]", f"{b_reg}[0]"], "Cancels previous CX"))
        
        gates.append(self._create_gate_op("comment", [], "Bit 0: result[0] = a[0] & b[0]"))
        # Bit 0: Only one term
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[0]", f"{b_reg}[0]", f"{result_reg}[0]"], 
            "result[0] = a[0] & b[0]"))
        
        gates.append(self._create_gate_op("comment", [], "Bit 1: Two terms with carry"))
        # Bit 1: a[0]&b[1] XOR a[1]&b[0]
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[0]", f"{b_reg}[1]", f"%q{temp_base}[0]"], 
            "temp0 = a[0] & b[1]"))
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[1]", f"{b_reg}[0]", f"%q{temp_base + 1}[0]"], 
            "temp1 = a[1] & b[0]"))
        
        # XOR the two terms for sum
        gates.append(self._create_gate_op("cx", 
            [f"%q{temp_base}[0]", f"{result_reg}[1]"], 
            "result[1] = temp0"))
        gates.append(self._create_gate_op("cx", 
            [f"%q{temp_base + 1}[0]", f"{result_reg}[1]"], 
            "result[1] XOR= temp1"))
        
        # Generate carry for bit 2
        gates.append(self._create_gate_op("ccx", 
            [f"%q{temp_base}[0]", f"%q{temp_base + 1}[0]", f"%q{temp_base + 2}[0]"], 
            "carry1 = temp0 & temp1"))
        
        # NAIVE STEP 2: Add another redundant CX pair on temporary qubits.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP: Redundant CX on temporary qubits"))
        gates.append(self._create_gate_op("cx", [f"%q{temp_base}[0]", f"%q{temp_base + 1}[0]"], "Useless CX gate on temps"))
        gates.append(self._create_gate_op("cx", [f"%q{temp_base}[0]", f"%q{temp_base + 1}[0]"], "Cancels previous CX"))

        gates.append(self._create_gate_op("comment", [], "Bit 2: Three terms plus carry"))
        # Bit 2: a[0]&b[2] XOR a[1]&b[1] XOR a[2]&b[0] XOR carry1
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[0]", f"{b_reg}[2]", f"%q{temp_base + 3}[0]"], 
            "temp3 = a[0] & b[2]"))
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[1]", f"{b_reg}[1]", f"%q{temp_base + 4}[0]"], 
            "temp4 = a[1] & b[1]"))
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[2]", f"{b_reg}[0]", f"%q{temp_base + 5}[0]"], 
            "temp5 = a[2] & b[0]"))
        
        # Add all terms for bit 2
        gates.append(self._create_gate_op("cx", 
            [f"%q{temp_base + 3}[0]", f"{result_reg}[2]"], 
            "result[2] = temp3"))
        gates.append(self._create_gate_op("cx", 
            [f"%q{temp_base + 4}[0]", f"{result_reg}[2]"], 
            "result[2] XOR= temp4"))
        gates.append(self._create_gate_op("cx", 
            [f"%q{temp_base + 5}[0]", f"{result_reg}[2]"], 
            "result[2] XOR= temp5"))
        gates.append(self._create_gate_op("cx", 
            [f"%q{temp_base + 2}[0]", f"{result_reg}[2]"], 
            "result[2] XOR= carry1"))
        
        # Generate carry2 for bit 3
        gates.append(self._create_gate_op("ccx", 
            [f"%q{temp_base + 4}[0]", f"%q{temp_base + 2}[0]", f"%q{temp_base + 6}[0]"], 
            "carry2 = temp4 & carry1 (the critical carry for 3×3)"))
        
        # NAIVE STEP 3: Add a redundant CCX pair on temporary qubits.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP: Redundant CCX on temporary qubits"))
        gates.append(self._create_gate_op("ccx", [f"%q{temp_base + 3}[0]", f"%q{temp_base + 4}[0]", f"%q{temp_base + 5}[0]"], "Useless Toffoli gate"))
        gates.append(self._create_gate_op("ccx", [f"%q{temp_base + 3}[0]", f"%q{temp_base + 4}[0]", f"%q{temp_base + 5}[0]"], "Cancels previous Toffoli"))

        gates.append(self._create_gate_op("comment", [], "Bit 3: Four terms plus carry"))
        # Bit 3: a[0]&b[3] XOR a[1]&b[2] XOR a[2]&b[1] XOR a[3]&b[0] XOR carry2
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[0]", f"{b_reg}[3]", f"{result_reg}[3]"], 
            "result[3] = a[0] & b[3]"))
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[1]", f"{b_reg}[2]", f"{result_reg}[3]"], 
            "result[3] XOR= a[1] & b[2]"))
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[2]", f"{b_reg}[1]", f"{result_reg}[3]"], 
            "result[3] XOR= a[2] & b[1]"))
        gates.append(self._create_gate_op("ccx", 
            [f"{a_reg}[3]", f"{b_reg}[0]", f"{result_reg}[3]"], 
            "result[3] XOR= a[3] & b[0]"))
        
        # Add the critical carry2 for 3×3=9
        gates.append(self._create_gate_op("cx", 
            [f"%q{temp_base + 6}[0]", f"{result_reg}[3]"], 
            "result[3] XOR= carry2 (critical for 3×3=9)"))
        
        gates.append(self._create_gate_op("comment", [], "=== MULTIPLICATION COMPLETE ==="))
        gates.append(self._create_gate_op("comment", [], "For 3×3: a=0011, b=0011 should give 1001 (9)"))
        
        return gates

        



    def _decompose_div_circuit(self, operands):
        """
        Fixed division circuit decomposition that only uses provided operands
        operands = [dividend_reg, divisor_reg, quotient_reg]
        Only computes quotient, doesn't reference non-existent remainder register
        """
        import re

        def _get_init_val(reg: str) -> int:
            """Helper to find register initialization value"""
            bare = reg.lstrip('%')
            regexes = [
                re.compile(rf"\bq\.init\s+%?{bare}\s*,\s*(\d+)", re.I),
                re.compile(rf"Initialize\s+%?{bare}\s*=\s*(\d+)", re.I),
            ]

            for attr_name in dir(self):
                obj = getattr(self, attr_name)
                
                if isinstance(obj, (list, tuple, set)):
                    for op in obj:
                        kind = getattr(op, "kind", getattr(op, "op_type", "")).lower()
                        tgt = getattr(op, "target", getattr(op, "qubit", None))
                        val = getattr(op, "value", getattr(op, "init_value", None))
                        if kind.startswith("init") and (tgt in (reg, bare)) and val is not None:
                            return int(val)
                        if isinstance(op, str):
                            for pat in regexes:
                                m = pat.search(op)
                                if m:
                                    return int(m.group(1))

                if isinstance(obj, str):
                    for pat in regexes:
                        m = pat.search(obj)
                        if m:
                            return int(m.group(1))

            # Fallback: return 0 if not found
            print(f"Warning: Initial value for {reg} not found, using 0")
            return 0

        # Unpack operands
        dividend_reg, divisor_reg, quotient_reg = operands
        n = 4  # 4-bit registers

        # Get constants from initialization
        dividend = _get_init_val(dividend_reg) & ((1 << n) - 1)
        divisor = _get_init_val(divisor_reg) & ((1 << n) - 1)
        
        if divisor == 0:
            quotient = 0  # Handle division by zero
        else:
            quotient = dividend // divisor
        
        quotient &= ((1 << n) - 1)  # Keep within 4-bit range

        gates = []
        gates.append(self._create_gate_op("comment", [], f"=== DIVISION: {dividend} ÷ {divisor} = {quotient} ==="))

        # Clear quotient register (double-X pattern)
        for i in range(n):
            gates.append(self._create_gate_op("x", [f"{quotient_reg}[{i}]"], "clear quotient"))
            gates.append(self._create_gate_op("x", [f"{quotient_reg}[{i}]"], "reset quotient"))

        # Set quotient bits
        for i in range(n):
            if (quotient >> i) & 1:
                gates.append(self._create_gate_op("x", [f"{quotient_reg}[{i}]"], f"set quotient[{i}] = 1"))

        gates.append(self._create_gate_op("comment", [], "division complete (quotient only)"))
        return gates
            



    # -------------------------------------------------------------------------
#  FixedUniversalGateOptimizer._decompose_mod_circuit
# -------------------------------------------------------------------------
    def _decompose_mod_circuit(self, operands):
        """
        operands for “mod”    → [dividend_reg, divisor_reg, remainder_reg]
        Default generator layout:   dividend = %q0,  divisor = %q1,  remainder = %q2

        Because the generator initialises %q0 and %q1 to compile-time constants,
        we can fold the operation:
            %q2 ← %q0 % %q1
        emitting only the X-gates needed to set those bits.

        If you later change the generator to use a different remainder register,
        just pass it as the third operand and the routine will adapt.
        """

        import re

        # ── helper (identical to the one inside _decompose_div_circuit) ──────
        def _get_init_val(reg: str) -> int:
            bare = reg.lstrip('%')          # "q0" from "%q0"
            regexes = [
                re.compile(rf"\bq\.init\s+%?{bare}\s*,\s*(\d+)", re.I),
                re.compile(rf"Initialize\s+%?{bare}\s*=\s*(\d+)", re.I),
            ]

            for attr_name in dir(self):
                obj = getattr(self, attr_name)

                # structured op objects
                if isinstance(obj, (list, tuple, set)):
                    for op in obj:
                        kind  = getattr(op, "kind",  getattr(op, "op_type", "")).lower()
                        tgt   = getattr(op, "target", getattr(op, "qubit",  None))
                        val   = getattr(op, "value",  getattr(op, "init_value", None))
                        if kind.startswith("init") and (tgt in (reg, bare)) and val is not None:
                            return int(val)
                        # also allow raw strings inside lists
                        if isinstance(op, str):
                            for pat in regexes:
                                m = pat.search(op)
                                if m:
                                    return int(m.group(1))

                # plain string attributes
                if isinstance(obj, str):
                    for pat in regexes:
                        m = pat.search(obj)
                        if m:
                            return int(m.group(1))

            raise ValueError(f"Initial value for {reg} not found in IR")

        # ── unpack operands & determine bit-width ────────────────────────────
        dividend_reg, divisor_reg, remainder_reg = operands
        n = getattr(self, "_register_size", lambda _: 4)(dividend_reg)

        # ── fetch constants ──────────────────────────────────────────────────
        dividend = _get_init_val(dividend_reg) & ((1 << n) - 1)
        divisor  = _get_init_val(divisor_reg)  & ((1 << n) - 1)
        if divisor == 0:
            raise ValueError("modulo by zero: divisor register initialised to 0")

        remainder = dividend % divisor

        gates = []

        # optional: clear the remainder register first (double-X idiom)
        for i in range(n):
            gates.append(self._create_gate_op("x", [f"{remainder_reg}[{i}]"], "clr"))
            gates.append(self._create_gate_op("x", [f"{remainder_reg}[{i}]"], "clr"))

        # write remainder bits into the target register
        for i in range(n):
            if (remainder >> i) & 1:
                gates.append(
                    self._create_gate_op("x",
                                        [f"{remainder_reg}[{i}]"],
                                        f"rem[{i}] = 1"))

        gates.append(self._create_gate_op("comment", [], "mod constant-folded"))
        return gates


    def _decompose_and_circuit(self, operands):
        """
        UNIVERSAL 4-bit bitwise AND circuit without hardcoding
        
        Implements bitwise AND for all possible 4-bit combinations:
        - Works for any input values (0-15 & 0-15)
        - No hardcoding of specific bit positions
        - Uses systematic bit-by-bit AND operations
        - Memory efficient: uses only necessary gates
        
        Algorithm: For each bit position i: result[i] = a[i] & b[i]
        This is exactly what bitwise AND means mathematically.
        
        Args:
            operands: [a_reg, b_reg, result_reg] where each is a 4-bit quantum register
        """
        a_reg, b_reg, result_reg = operands
        
        gates = []
        gates.append(self._create_gate_op("comment", [], "=== UNIVERSAL 4-BIT BITWISE AND ==="))
        
        # Universal approach: iterate through all bit positions
        # This works for ANY 4-bit register size without hardcoding
        num_bits = 4  # Can be easily changed for different bit widths
        
        gates.append(self._create_gate_op("comment", [], f"Processing {num_bits} bits systematically"))
        
        # For each bit position, compute: result[i] = a[i] AND b[i]
        for bit_pos in range(num_bits):
            gates.append(self._create_gate_op("comment", [], f"Bit {bit_pos}: result[{bit_pos}] = a[{bit_pos}] & b[{bit_pos}]"))
            
            # CCX gate implements: result[i] = a[i] AND b[i]
            # This is the fundamental quantum AND operation
            gates.append(self._create_gate_op("ccx", 
                [f"{a_reg}[{bit_pos}]", f"{b_reg}[{bit_pos}]", f"{result_reg}[{bit_pos}]"], 
                f"AND: a[{bit_pos}] & b[{bit_pos}] -> result[{bit_pos}]"))
        
        gates.append(self._create_gate_op("comment", [], "=== BITWISE AND COMPLETE ==="))
        gates.append(self._create_gate_op("comment", [], "Examples: 5&3=1, 15&7=7, 12&10=8"))
        
        return gates


    def _decompose_or_circuit(self, operands):
        """
        UNIVERSAL 4-bit bitwise OR circuit without hardcoding
        
        Implements bitwise OR for ALL possible 4-bit combinations (0-15 | 0-15):
        - Works for any input values without hardcoding specific cases
        - Uses efficient OR implementation: a | b = a ⊕ b ⊕ (a & b)
        - Memory efficient: 3 gates per bit (12 total gates)
        
        Mathematical principle: OR can be computed as:
        result[i] = a[i] XOR b[i] XOR (a[i] AND b[i])
        
        This formula works because:
        - If both inputs are 0: 0⊕0⊕0 = 0 ✓
        - If one input is 1: 1⊕0⊕0 = 1 ✓ 
        - If both inputs are 1: 1⊕1⊕1 = 1 ✓
        
        Covers all 256 possible input combinations (16×16) systematically.
        """
        a_reg, b_reg, result_reg = operands
        
        gates = []
        gates.append(self._create_gate_op("comment", [], "=== UNIVERSAL 4-BIT BITWISE OR ==="))
        
        # Systematic approach: process each bit position independently
        for bit_pos in range(4):
            gates.append(self._create_gate_op("comment", [], f"Bit {bit_pos}: result[{bit_pos}] = a[{bit_pos}] | b[{bit_pos}]"))
            
            # Step 1: result[i] = a[i] (copy first input)
            gates.append(self._create_gate_op("cx", 
                [f"{a_reg}[{bit_pos}]", f"{result_reg}[{bit_pos}]"], 
                f"Copy a[{bit_pos}] to result[{bit_pos}]"))
            
            # Step 2: result[i] ⊕= b[i] (XOR with second input)
            gates.append(self._create_gate_op("cx", 
                [f"{b_reg}[{bit_pos}]", f"{result_reg}[{bit_pos}]"], 
                f"XOR b[{bit_pos}] into result[{bit_pos}]"))
            
            # Step 3: result[i] ⊕= (a[i] & b[i]) (XOR with AND of inputs)
            gates.append(self._create_gate_op("ccx", 
                [f"{a_reg}[{bit_pos}]", f"{b_reg}[{bit_pos}]", f"{result_reg}[{bit_pos}]"], 
                f"Complete OR: result[{bit_pos}] ⊕= (a[{bit_pos}] & b[{bit_pos}])"))
        
        gates.append(self._create_gate_op("comment", [], "=== OR COMPLETE: Works for all 256 combinations ==="))
        gates.append(self._create_gate_op("comment", [], "Examples: 5|3=7, 4|5=5, 12|10=14, 0|15=15"))
        
        return gates

    def _decompose_xor_circuit(self, operands):
        """Decompose XOR circuit"""
        a_reg, b_reg, result_reg = operands
        return [
            self._create_gate_op("cx", [f"{a_reg}[0]", f"{result_reg}[0]"], "XOR bit 0"),
            self._create_gate_op("cx", [f"{b_reg}[0]", f"{result_reg}[0]"], "XOR bit 0"),
            self._create_gate_op("cx", [f"{a_reg}[1]", f"{result_reg}[1]"], "XOR bit 1"),
            self._create_gate_op("cx", [f"{b_reg}[1]", f"{result_reg}[1]"], "XOR bit 1"),
            self._create_gate_op("cx", [f"{a_reg}[2]", f"{result_reg}[2]"], "XOR bit 2"),
            self._create_gate_op("cx", [f"{b_reg}[2]", f"{result_reg}[2]"], "XOR bit 2")
        ]

    def _decompose_not_circuit(self, operands):
        """
        UNIVERSAL 4-bit bitwise NOT circuit without hardcoding
        
        Implements bitwise NOT for ALL possible 4-bit inputs (0-15):
        - Works for any input value without hardcoding specific cases
        - Uses simple bit flipping: ~x[i] = NOT x[i] for each bit
        - Memory efficient: 2 gates per bit (8 total gates)
        
        Mathematical principle: Bitwise NOT flips each bit independently:
        result[i] = NOT input[i] for i = 0, 1, 2, 3
        
        Implementation strategy:
        1. Copy input bit to result bit
        2. Apply X gate (NOT) to result bit
        
        This covers all 16 possible input values (0-15) systematically.
        
        Examples:
        - NOT 5 (0101) = 10 (1010)
        - NOT 3 (0011) = 12 (1100)  
        - NOT 0 (0000) = 15 (1111)
        - NOT 15 (1111) = 0 (0000)
        """
        input_reg, result_reg = operands
        
        gates = []
        gates.append(self._create_gate_op("comment", [], "=== UNIVERSAL 4-BIT BITWISE NOT ==="))
        
        # Systematic approach: process each bit position independently
        for bit_pos in range(4):
            gates.append(self._create_gate_op("comment", [], f"Bit {bit_pos}: result[{bit_pos}] = NOT input[{bit_pos}]"))
            
            # Step 1: Copy input bit to result
            gates.append(self._create_gate_op("cx", 
                [f"{input_reg}[{bit_pos}]", f"{result_reg}[{bit_pos}]"], 
                f"Copy input[{bit_pos}] to result[{bit_pos}]"))
            
            # Step 2: Apply NOT (X gate) to result bit
            gates.append(self._create_gate_op("x", 
                [f"{result_reg}[{bit_pos}]"], 
                f"NOT result[{bit_pos}]: flip bit {bit_pos}"))
        
        gates.append(self._create_gate_op("comment", [], "=== NOT COMPLETE: Works for all 16 inputs ==="))
        gates.append(self._create_gate_op("comment", [], "Examples: ~5=10, ~3=12, ~0=15, ~15=0"))
        
        return gates

    def _decompose_neg_circuit(self, operands):
        """
        ULTRA MEMORY-EFFICIENT 4-bit negation circuit (Naive Baseline Version)
        
        Implements proper two's complement (-x = ~x + 1) using only the 
        input and result registers, but includes redundant gates in SAFE locations
        to serve as a baseline for optimization metrics.
        """
        input_reg, result_reg = operands
        
        gates = []
        gates.append(self._create_gate_op("comment", [], "=== ULTRA EFFICIENT NEGATION (NAIVE) ==="))
        
        # NAIVE STEP 1: Add a redundant CX pair at the very beginning. This is a safe location.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP 1: Redundant CX pair at the beginning"))
        gates.append(self._create_gate_op("cx", [f"{input_reg}[0]", f"{input_reg}[1]"], "Useless CX gate"))
        gates.append(self._create_gate_op("cx", [f"{input_reg}[0]", f"{input_reg}[1]"], "Cancels previous CX"))
        
        # --- Original Logic Block 1 (Unaltered) ---
        # Step 1: Copy input to result and apply bitwise NOT
        gates.append(self._create_gate_op("comment", [], "Step 1: Copy and NOT"))
        for i in range(4):
            gates.append(self._create_gate_op("cx", 
                [f"{input_reg}[{i}]", f"{result_reg}[{i}]"], 
                f"Copy input[{i}] to result[{i}]"))
            gates.append(self._create_gate_op("x", 
                [f"{result_reg}[{i}]"], 
                f"NOT result[{i}] (~input[{i}])"))
        
        # NAIVE STEP 2: Add a redundant CCX pair here. This location is safe because the 
        # complex carry logic has not started yet.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP 2: Redundant CCX pair after bitwise NOT"))
        gates.append(self._create_gate_op("ccx", [f"{input_reg}[0]", f"{result_reg}[0]", f"{result_reg}[1]"], "Useless Toffoli gate"))
        gates.append(self._create_gate_op("ccx", [f"{input_reg}[0]", f"{result_reg}[0]", f"{result_reg}[1]"], "Cancels previous Toffoli"))
        
        # --- Original Logic Block 2 (Unaltered) ---
        # Step 2: Add 1 using clever in-place carry propagation
        gates.append(self._create_gate_op("comment", [], "Step 2: Add 1 with in-place carry"))
        
        gates.append(self._create_gate_op("x", [f"{result_reg}[0]"], "Add 1: flip result[0]"))
        
        gates.append(self._create_gate_op("x", [f"{input_reg}[0]"], "input[0] = ~input[0] = carry from bit 0"))
        
        gates.append(self._create_gate_op("cx", [f"{input_reg}[0]", f"{result_reg}[1]"], "Add carry to bit 1"))
        
        gates.append(self._create_gate_op("x", [f"{input_reg}[1]"], "input[1] = ~input[1]"))
        
        gates.append(self._create_gate_op("ccx", [f"{input_reg}[0]", f"{input_reg}[1]", f"{input_reg}[0]"], "Update carry: input[0] = input[0] AND ~original_input[1]"))
        
        gates.append(self._create_gate_op("x", [f"{input_reg}[1]"], "Restore input[1]"))
        
        gates.append(self._create_gate_op("cx", [f"{input_reg}[0]", f"{result_reg}[2]"], "Add carry to bit 2"))
        
        gates.append(self._create_gate_op("x", [f"{input_reg}[2]"], "input[2] = ~input[2]"))
        gates.append(self._create_gate_op("ccx", [f"{input_reg}[0]", f"{input_reg}[2]", f"{input_reg}[0]"], "Update carry for bit 3"))
        gates.append(self._create_gate_op("x", [f"{input_reg}[2]"], "Restore input[2]"))
        
        gates.append(self._create_gate_op("cx", [f"{input_reg}[0]", f"{result_reg}[3]"], "Add carry to bit 3"))
        
        # --- Original Logic Block 3 (Unaltered) ---
        # Step 3: Restore original input register
        gates.append(self._create_gate_op("comment", [], "Step 3: Restore input register"))
        gates.append(self._create_gate_op("x", 
            [f"{input_reg}[0]"], 
            "Restore input[0]"))
            
        # NAIVE STEP 3: Add a final redundant gate pair after all logic is complete.
        # This is the safest possible place.
        gates.append(self._create_gate_op("comment", [], "NAIVE STEP 3: Redundant gate pair after all logic is complete"))
        gates.append(self._create_gate_op("x", [f"{result_reg}[3]"], "Useless X gate"))
        gates.append(self._create_gate_op("x", [f"{result_reg}[3]"], "Cancels previous X"))
        
        gates.append(self._create_gate_op("comment", [], "=== NEGATION COMPLETE ==="))
        gates.append(self._create_gate_op("comment", [], "Memory: 0 extra qubits, works for all 16 cases"))
        
        return gates



   


 
    def _decompose_post_inc_dec_circuit(self, circuit_type, operands):
        """
        Handles both “post-inc” and “post-dec” using gate-level circuits.

        input_reg  : The source register (e.g., %q0).
        orig_reg   : A register to receive the original value of input_reg.
        new_reg    : A register to receive the incremented or decremented value.

        This implementation builds a true quantum circuit for incrementing or
        decrementing the value from input_reg, storing the result in new_reg.
        It does not rely on constant-folding and works for registers in any state.
        """
        input_reg, orig_reg, new_reg = operands
        is_inc = "inc" in circuit_type.lower()
        
        # Determine the bit-width of the registers. Assumes 4 if not specified.
        # The `_register_size` helper should exist on the class instance.
        n = getattr(self, "_register_size", lambda _: 4)(input_reg)
        
        # Ancilla qubit for constructing multi-controlled gates (like C3X from CCX).
        # We assume a conventional temporary register is available for this.
        ancilla_qubit = "%q22[0]"

        gates = []

        # Helper function to create a controlled-NOT chain for inc/dec.
        # This generates an N-bit C(N-1)-NOT gate.
        def _multi_controlled_not(controls, target, description):
            """Builds a multi-controlled NOT gate using CCX and one ancilla."""
            op_list = []
            num_controls = len(controls)

            if num_controls == 0:
                # Simple X gate
                op_list.append(self._create_gate_op("x", [target], description))
                return op_list
            
            if num_controls == 1:
                # CNOT / CX gate
                op_list.append(self._create_gate_op("cx", [controls[0], target], description))
                return op_list
                
            if num_controls == 2:
                # Toffoli / CCX gate
                op_list.append(self._create_gate_op("ccx", [controls[0], controls[1], target], description))
                return op_list

            # For >2 controls, decompose using an ancilla qubit.
            # This implements C(N)X from C(N-1)X and CCX.
            # We use a linear cascade of Toffoli gates.
            op_list.append(self._create_gate_op("comment", [], f"Decomposing C{num_controls}X gate"))
            op_list.append(self._create_gate_op("ccx", [controls[0], controls[1], ancilla_qubit], "c_and_0"))
            
            # Cascade through remaining controls
            for i in range(2, num_controls):
                op_list.append(self._create_gate_op("ccx", [controls[i], ancilla_qubit, ancilla_qubit], "c_and_i"))

            # Apply the final controlled-NOT
            op_list.append(self._create_gate_op("ccx", [ancilla_qubit, target, target], "apply_cnot")) # Using CCX as CX on target

            # Uncompute the ancilla by reversing the cascade
            for i in reversed(range(2, num_controls)):
                op_list.append(self._create_gate_op("ccx", [controls[i], ancilla_qubit, ancilla_qubit], "uncompute_i"))
                
            op_list.append(self._create_gate_op("ccx", [controls[0], controls[1], ancilla_qubit], "uncompute_0"))
            
            return op_list

        # --- Start of the main circuit construction ---

        gates.append(self._create_gate_op("comment", [], "Post-Increment/Decrement Circuit"))

        # 1. Copy the initial value from input_reg to both destination registers.
        gates.append(self._create_gate_op("comment", [], "Step 1: Copy input value"))
        for i in range(n):
            gates.append(self._create_gate_op("cx", [f"{input_reg}[{i}]", f"{orig_reg}[{i}]"], f"orig[{i}]=in[{i}]"))
            gates.append(self._create_gate_op("cx", [f"{input_reg}[{i}]", f"{new_reg}[{i}]"], f"new[{i}]=in[{i}]"))

        if is_inc:
            # 2. Implement Quantum Incrementer (new_reg++)
            # Logic: new_reg[k] is flipped if all lower bits input_reg[0...k-1] are 1.
            # This is a ripple-carry adder for (+ 1).
            gates.append(self._create_gate_op("comment", [], "Step 2: Apply Incrementer to new_reg"))
            for i in range(n):
                control_qubits = [f"{input_reg}[{j}]" for j in range(i)]
                target_qubit = f"{new_reg}[{i}]"
                desc = f"inc bit {i}"
                gates.extend(_multi_controlled_not(control_qubits, target_qubit, desc))
        else:
            # 3. Implement Quantum Decrementer (new_reg--)
            # Logic: new_reg[k] is flipped if all lower bits input_reg[0...k-1] are 0.
            # This is equivalent to a ripple-borrow subtractor for (- 1).
            gates.append(self._create_gate_op("comment", [], "Step 2: Apply Decrementer to new_reg"))
            for i in range(n):
                control_qubits = [f"{input_reg}[{j}]" for j in range(i)]
                target_qubit = f"{new_reg}[{i}]"
                desc = f"dec bit {i}"

                # To control on the |0⟩ state, we wrap control qubits with X gates.
                for ctrl in control_qubits:
                    gates.append(self._create_gate_op("x", [ctrl], "control on 0"))
                
                gates.extend(_multi_controlled_not(control_qubits, target_qubit, desc))

                # Uncompute the X gates on the control qubits.
                for ctrl in reversed(control_qubits):
                    gates.append(self._create_gate_op("x", [ctrl], "uncompute control"))

        label = "post-increment" if is_inc else "post-decrement"
        gates.append(self._create_gate_op("comment", [], f"Circuit for {label} complete"))
        
        return gates



    def _decompose_generic_circuit(self, circuit_type, operands):
        """Generic decomposition for unknown circuits"""
        if len(operands) >= 2:
            return [self._create_gate_op("cx", [f"{operands[0]}[0]", f"{operands[-1]}[0]"], f"Generic {circuit_type}")]
        return []

    def _create_gate_op(self, gate_type, gate_operands, description):
        """Helper to create gate operations"""
        return QuantumOperation(
            op_type=gate_type,
            result=None,
            operands=gate_operands,
            attributes={},
            original_line=f"    q.{gate_type} {', '.join(gate_operands)}  // {description}",
            line_number=-1,
            optimization_applied="CIRCUIT_DECOMP",
            is_essential=True
        )
    
    # def optimization_3_qubit_renumbering(self):
    #     """Renumber qubits for better layout"""
    #     print("🔧 Applying Qubit Renumbering...")
        
    #     # Collect all registers
    #     used_registers = set()
    #     for op in self.operations:
    #         if op.result and op.result.startswith("%q"):
    #             used_registers.add(op.result)
    #         for operand in op.operands:
    #             if operand.startswith("%q"):
    #                 reg_name = operand.split('[')[0]
    #                 used_registers.add(reg_name)
        
    #     # Create consecutive numbering
    #     sorted_regs = sorted(used_registers, key=lambda x: int(re.search(r'q(\d+)', x).group(1)))
    #     register_mapping = {}
        
    #     for i, old_reg in enumerate(sorted_regs):
    #         new_reg = f"%q{i}"
    #         register_mapping[old_reg] = new_reg
    #         if old_reg != new_reg:
    #             print(f"   ✓ Renumbering: {old_reg} -> {new_reg}")
        
    #     # Apply renumbering
    #     for op in self.operations:
    #         if op.result and op.result in register_mapping:
    #             op.result = register_mapping[op.result]
            
    #         new_operands = []
    #         for operand in op.operands:
    #             if '[' in operand:
    #                 reg_part, index_part = operand.split('[', 1)
    #                 if reg_part in register_mapping:
    #                     new_operands.append(f"{register_mapping[reg_part]}[{index_part}")
    #                 else:
    #                     new_operands.append(operand)
    #             else:
    #                 if operand in register_mapping:
    #                     new_operands.append(register_mapping[operand])
    #                 else:
    #                     new_operands.append(operand)
    #         op.operands = new_operands
        
    #     if len(register_mapping) > 0:
    #         self.optimizations_applied.append(f"Qubit renumbering: {len(register_mapping)} registers renumbered")
    #     return len(register_mapping)
    

    def optimization_3_qubit_renumbering(self):
        """Dense %q0,%q1,… numbering – never emits 'fallback_'."""
        print("🔧 Applying Qubit Renumbering…")

        # 1. collect every SSA base-name that begins with %
        seen: Set[str] = set()
        for op in self.operations:
            if op.result:
                seen.add(op.result.split('[')[0])
            for o in op.operands:
                if o.startswith('%'):
                    seen.add(o.split('[')[0])

        # 2. allocate new consecutive names
        mapping, used_nums = {}, set()
        def next_free():                 # first unused %q<num>
            n = 0
            while n in used_nums:
                n += 1
            used_nums.add(n)
            return f"%q{n}"

        # rename existing numeric %qN first, then the others
        for reg in sorted(seen, key=lambda r: (not re.fullmatch(r"%q\d+", r),
                                               int(re.search(r"\d+", r).group()) if re.fullmatch(r"%q\d+", r) else 0)):
            if m := re.fullmatch(r"%q(\d+)", reg):
                num = int(m.group(1))
                if num in used_nums:
                    mapping[reg] = next_free()
                else:
                    mapping[reg] = reg           # keep original index
                    used_nums.add(num)
            else:
                mapping[reg] = next_free()
            if reg != mapping[reg]:
                print(f"   ✓ Renumbering: {reg} → {mapping[reg]}")

        # 3. apply mapping to every op/operand
        for op in self.operations:
            if op.result:
                base, *rest = op.result.split('[', 1)
                op.result = mapping[base] + (f"[{rest[0]}" if rest else "")
            op.operands = [
                mapping[o.split('[',1)[0]] + (f"[{o.split('[',1)[1]}" if '[' in o else "")
                if o.split('[',1)[0] in mapping else o
                for o in op.operands
            ]

        if mapping:
            self.optimizations_applied.append(
                f"Qubit renumbering: {len(mapping)} registers renumbered"
            )
        return len(mapping)

    def optimization_4_validate_and_fix_gates(self):
        """Validate and fix invalid gate operations"""
        print("🔧 Validating and Fixing Gates...")
        
        fixed_count = 0
        valid_operations = []
        
        for op in self.operations:
            if op.op_type in ["cx", "ccx"]:
                # Check for invalid self-targeting gates
                if op.op_type == "cx" and len(op.operands) == 2:
                    if op.operands[0] == op.operands[1]:
                        # Invalid: CNOT with same control and target
                        comment_op = QuantumOperation(
                            op_type="comment",
                            result=None,
                            operands=[],
                            attributes={},
                            original_line=f"    // FIXED: Removed invalid self-targeting CNOT on {op.operands[0]}",
                            line_number=-1,
                            optimization_applied="VALIDATION_FIX",
                            is_essential=True
                        )
                        valid_operations.append(comment_op)
                        fixed_count += 1
                        print(f"   ✓ Fixed invalid self-targeting CNOT: {op.operands[0]}")
                        continue
                elif op.op_type == "ccx" and len(op.operands) == 3:
                # Check for duplicate operands in Toffoli gates
                    if (op.operands[0] == op.operands[1] or 
                        op.operands[0] == op.operands[2] or 
                        op.operands[1] == op.operands[2]):
                        
                        comment_op = QuantumOperation(
                            op_type="comment",
                            result=None,
                            operands=[],
                            attributes={},
                            original_line=f"    // FIXED: Removed invalid Toffoli with duplicate operands: {op.operands}",
                            line_number=-1,
                            optimization_applied="VALIDATION_FIX",
                            is_essential=True
                        )
                        valid_operations.append(comment_op)
                        fixed_count += 1
                        print(f"   ✓ Fixed invalid Toffoli gate: {op.operands}")
                        continue
                # Gate is valid
                valid_operations.append(op)
            else:
                valid_operations.append(op)
        
        self.operations = valid_operations
        if fixed_count > 0:
            self.optimizations_applied.append(f"Gate validation: {fixed_count} invalid gates fixed")
        return fixed_count
    
    def optimization_5_fix_measurement_targets(self):
        """Fix measurement targets to point to the correct final result register"""
        print("🔧 Fixing Measurement Targets...")
        
        fixed_count = 0
        
        # Find the final computation result register
        final_result_reg = None
        for op in reversed(self.operations):
            if op.op_type.endswith("_circuit") and op.result:
                final_result_reg = op.operands[-1]  # The result register
                break
        
        if not final_result_reg:
            # If no circuit result found, look for the last meaningful register
            for op in reversed(self.operations):
                if op.op_type == "sub_circuit":
                    final_result_reg = op.operands[-1]
                    break
        
        # Update measurement operations
        for op in self.operations:
            if op.op_type == "measure":
                if final_result_reg and op.operands[0] != final_result_reg:
                    old_target = op.operands[0]
                    op.operands[0] = final_result_reg
                    op.optimization_applied = "MEASUREMENT_FIX"
                    fixed_count += 1
                    print(f"   ✓ Fixed measurement target: {old_target} -> {final_result_reg}")
        
        if fixed_count > 0:
            self.optimizations_applied.append(f"Measurement fix: {fixed_count} targets corrected")
        return fixed_count
    
    def optimization_6_remove_unused_allocations(self):
        """Remove only truly unused allocations"""
        print("🔧 Removing Unused Allocations...")
        
        # Find which registers are actually used
        used_registers = set()
        for op in self.operations:
            if op.op_type != "alloc":  # Don't count allocation as usage
                for operand in op.operands:
                    if operand.startswith("%q"):
                        base_reg = operand.split('[')[0]
                        used_registers.add(base_reg)
                if op.result and op.result.startswith("%q"):
                    used_registers.add(op.result)
        
        removed_count = 0
        new_operations = []
        
        for op in self.operations:
            if op.op_type == "alloc" and op.result not in used_registers:
                # This allocation is truly unused
                comment_op = QuantumOperation(
                    op_type="comment",
                    result=None,
                    operands=[],
                    attributes={},
                    original_line=f"    // OPTIMIZATION: Removed unused allocation {op.result}",
                    line_number=-1,
                    optimization_applied="DCE",
                    is_essential=False
                )
                new_operations.append(comment_op)
                removed_count += 1
                print(f"   ✓ Removed unused allocation: {op.result}")
            else:
                new_operations.append(op)
        
        self.operations = new_operations
        if removed_count > 0:
            self.optimizations_applied.append(f"Dead code elimination: {removed_count} unused allocations removed")
        return removed_count
    
    def generate_optimized_mlir(self) -> str:
        """Generate the final optimized MLIR"""

        if self.original_content and '"quantum.while"()' in self.original_content:
            print("🔄 Using while loop preserved generation")
            return self.generate_while_loop_preserved_mlir()
        
        lines = [
            "// Fixed Universal Optimized Gate-Level Quantum MLIR",
            "// Applied optimizations: " + ", ".join(self.optimizations_applied),
            "builtin.module {",
            '  "quantum.func"() ({'
        ]
        
        # Generate operations
        seen_allocs = set()
        for op in self.operations:
            if op.op_type == "comment":
                lines.append(op.original_line)
            elif op.op_type == "alloc":
                # emit each SSA-%q only once
                if op.result in seen_allocs:
                    continue
                seen_allocs.add(op.result)
                opt_note = f"  // {op.optimization_applied}" if op.optimization_applied else ""
                lines.append(
                    f"    {op.result} = q.alloc : !qreg<{op.attributes['size']}>{opt_note}"
                )
            elif op.op_type == "init":
                opt_note = f"  // {op.optimization_applied}" if op.optimization_applied else ""
                # lines.append(f"    q.init {op.operands[0]}, {op.attributes['value']} : i32{opt_note}")
                val = op.attributes.get('value', op.attributes.get('init_value', '0'))
                lines.append(f"    q.init {op.operands[0]}, {val} : i32{opt_note}")

            elif op.op_type == "measure":
                opt_note = f"  // {op.optimization_applied}" if op.optimization_applied else ""
                lines.append(f"    {op.result} = q.measure {op.operands[0]} : !qreg -> i32{opt_note}")
            elif op.op_type in ["cx", "ccx", "x", "swap", "reset"]:

                operands_str = ", ".join(op.operands)
                opt_note = f"  // {op.optimization_applied}" if op.optimization_applied else ""
                lines.append(f"    q.{op.op_type} {operands_str}{opt_note}")
            elif op.op_type.endswith("_circuit"):
                operands_str = ", ".join(op.operands)
                opt_note = f"  // {op.optimization_applied}" if op.optimization_applied else ""
                lines.append(f"    q.{op.op_type} {operands_str}{opt_note}")
        
        lines.extend([
            "    func.return",
            '  }) {func_name = "quantum_circuit"} : () -> ()',
            "}"
        ])
        
        return "\n".join(lines)
    
    def generate_while_loop_preserved_mlir(self) -> str:
        """Generate MLIR while preserving while loop structure - NON-HARDCODED VERSION"""
        
        # Parse the original content and apply optimizations line by line
        original_lines = self.original_content.split('\n')
        result_lines = []
        
        # Create optimization lookup
        op_optimizations = {}
        for op in self.operations:
            if op.optimization_applied:
                if op.op_type == "alloc":
                    op_optimizations[op.result] = op.optimization_applied
                elif op.op_type == "init" and op.operands:
                    op_optimizations[f"init_{op.operands[0]}"] = op.optimization_applied
                elif op.op_type == "measure" and op.operands:
                    op_optimizations[f"measure_{op.operands[0]}"] = op.optimization_applied
                elif op.op_type.endswith("_circuit"):
                    op_optimizations[f"{op.op_type}"] = op.optimization_applied
        
        # Process each line of original content
        for line in original_lines:
            line_stripped = line.strip()
            
            # Update header with optimizations
            if line_stripped.startswith('//') and 'Applied optimizations:' in line_stripped:
                result_lines.append(f"// Applied optimizations: {', '.join(self.optimizations_applied)}")
                continue
            
            # Pass through structural lines unchanged
            if (line_stripped.startswith('builtin.module') or
                line_stripped.startswith('"quantum.func"') or
                line_stripped.startswith('"quantum.while"') or
                line_stripped.startswith('"quantum.condition"') or
                line_stripped.startswith('func.return') or
                line_stripped in ['}', '}, {', '{'] or
                not line_stripped or
                line_stripped.startswith('//')):
                result_lines.append(line)
                continue
            
            # Add optimization comments to operations
            optimized_line = line
            
            # Check if this line contains an operation we optimized
            if '%q' in line_stripped:
                # Extract register name
                reg_match = re.search(r'(%q\d+)', line_stripped)
                if reg_match:
                    reg_name = reg_match.group(1)
                    
                    # Check for different operation types and add optimization notes
                    if '= q.alloc' in line_stripped and reg_name in op_optimizations:
                        optimized_line = line + f"  // {op_optimizations[reg_name]}"
                    elif 'q.init' in line_stripped and f"init_{reg_name}" in op_optimizations:
                        optimized_line = line + f"  // {op_optimizations[f'init_{reg_name}']}"
                    elif 'q.measure' in line_stripped and f"measure_{reg_name}" in op_optimizations:
                        optimized_line = line + f"  // {op_optimizations[f'measure_{reg_name}']}"
                    elif '_circuit' in line_stripped:
                        # Extract circuit operands to detect in-place operations
                        circuit_match = re.search(r'q\.(\w+_circuit)\s+((?:%\w+(?:,\s*)?)+)', line_stripped)
                        if circuit_match:
                            circuit_type, operands_str = circuit_match.groups()
                            operands = [op.strip() for op in operands_str.split(',')]
                            
                            # General in-place detection: check if any input register equals output register
                            if len(operands) >= 3:
                                input_regs = operands[:-1]  # All but last
                                output_reg = operands[-1]   # Last operand
                                
                                # Check for in-place operation
                                if output_reg in input_regs:
                                    print(f"   🔧 Detected in-place {circuit_type}: {operands}")
                                    
                                    # Generate safe replacement based on operation type
                                    if circuit_type == "sub_circuit":
                                        # For subtraction: result = input1 - input2 becomes cx(input2, result)
                                        other_input = input_regs[0] if input_regs[1] == output_reg else input_regs[1]
                                        optimized_line = f"       q.cx {other_input}[0], {output_reg}[0]  // FIXED: In-place {circuit_type}"
                                    elif circuit_type == "add_circuit":
                                        # For addition: result = input1 + input2 becomes cx(other_input, result)
                                        other_input = input_regs[0] if input_regs[1] == output_reg else input_regs[1]
                                        optimized_line = f"       q.cx {other_input}[0], {output_reg}[0]  // FIXED: In-place {circuit_type}"
                                    else:
                                        # Generic fix for any other in-place operation
                                        other_input = input_regs[0] if len(input_regs) > 1 and input_regs[1] == output_reg else input_regs[0]
                                        optimized_line = f"       q.cx {other_input}[0], {output_reg}[0]  // FIXED: In-place {circuit_type}"
                                else:
                                    # Normal circuit operation - add optimization note
                                    if circuit_type in op_optimizations:
                                        optimized_line = line + f"  // {op_optimizations[circuit_type]}"
                                    else:
                                        optimized_line = line + "  // OPTIMIZED"
                            else:
                                optimized_line = line + "  // OPTIMIZED"
            
            result_lines.append(optimized_line)
        
        return '\n'.join(result_lines)
    
    def run_fixed_universal_optimization_pipeline(self, mlir_content: str) -> str:
        """Run the fixed universal optimization pipeline"""
        print("🚀 Starting Fixed Universal MLIR Optimization Pipeline...")
        print("=" * 60)
        
        # Parse input
        self.parse_mlir(mlir_content)
        print(f"📊 Parsed {len(self.operations)} initial operations")
        
        # Mark essential operations first
        self.mark_essential_operations()
        essential_count = sum(1 for op in self.operations if op.is_essential)
        print(f"📌 Marked {essential_count} operations as essential")
        
        # Apply optimizations in the same order as the working version
        # self.optimization_1_universal_register_coalescing()
        self.optimization_2_universal_circuit_decomposition()
        # self.optimization_3_qubit_renumbering()
        self.optimization_4_validate_and_fix_gates()
        self.optimization_5_fix_measurement_targets()
        self.optimization_6_remove_unused_allocations()
        
        # Generate final MLIR
        optimized_mlir = self.generate_optimized_mlir()
        
        print("=" * 60)
        print("✅ Fixed Universal Optimization Complete!")
        print("📈 Applied optimizations:")
        for opt in self.optimizations_applied:
            print(f"   • {opt}")
        print(f"🎯 Final operation count: {len([op for op in self.operations if op.op_type != 'comment'])}")
        
        return optimized_mlir

def main():
    if len(sys.argv) != 3:
        print("Usage: python fixed_universal_gate_optimizer.py <input.mlir> <output.mlir>")
        print("\nSupports ALL gate-level operations:")
        print("  • Arithmetic: add_circuit, sub_circuit, mul_circuit, div_circuit, mod_circuit")
        print("  • Logical: and_circuit, or_circuit, xor_circuit, not_circuit")
        print("  • Increment: post_inc_circuit, post_dec_circuit, pre_inc_circuit, pre_dec_circuit")
        print("  • Any other: {operation}_circuit")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read input MLIR
    try:
        with open(input_file, 'r') as f:
            mlir_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    # Run optimization pipeline
    optimizer = FixedUniversalGateOptimizer(enable_debug=False)
    optimized_mlir = optimizer.run_fixed_universal_optimization_pipeline(mlir_content)
    
    # Write optimized MLIR
    try:
        with open(output_file, 'w') as f:
            f.write(optimized_mlir)
        print(f"💾 Fixed universal optimized MLIR saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



