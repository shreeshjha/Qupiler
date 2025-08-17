// Fixed Universal Optimized Gate-Level Quantum MLIR
// Applied optimizations: Circuit decomposition: 2 circuits decomposed into gates
builtin.module {
  "quantum.func"() ({
    %q0 = q.alloc : !qreg<4>
    q.init %q0, 2 : i32
    %q1 = q.alloc : !qreg<4>
    q.init %q1, 7 : i32
    %q2 = q.alloc : !qreg<4>
    // OPTIMIZATION: Decomposed add_circuit into basic gates
    q.comment   // === COMPLETE 4-BIT RIPPLE CARRY ADDER (NAIVE) ===
    q.comment   // NAIVE STEP: Redundant gate pair on result[0]
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.comment   // Allocate carry registers
    q.comment   // Bit 0: Half adder (no carry in)
    q.cx %q0[0], %q2[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q0[0], %q1[0], %q20[0]  // CIRCUIT_DECOMP
    q.comment   // Bit 1: Full adder
    q.cx %q0[1], %q21[0]  // CIRCUIT_DECOMP
    q.cx %q1[1], %q21[0]  // CIRCUIT_DECOMP
    q.cx %q21[0], %q2[1]  // CIRCUIT_DECOMP
    q.cx %q20[0], %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q0[1], %q1[1], %q22[0]  // CIRCUIT_DECOMP
    q.ccx %q20[0], %q21[0], %q23[0]  // CIRCUIT_DECOMP
    q.cx %q22[0], %q20[1]  // CIRCUIT_DECOMP
    q.cx %q23[0], %q20[1]  // CIRCUIT_DECOMP
    q.comment   // Bit 2: Full adder
    q.comment   // NAIVE STEP: Redundant quantum gates for optimization testing
    q.h %q2[2]  // CIRCUIT_DECOMP
    q.h %q2[2]  // CIRCUIT_DECOMP
    q.t %q2[1]  // CIRCUIT_DECOMP
    q.t %q2[1]  // CIRCUIT_DECOMP
    q.t %q2[1]  // CIRCUIT_DECOMP
    q.t %q2[1]  // CIRCUIT_DECOMP
    q.t %q2[1]  // CIRCUIT_DECOMP
    q.t %q2[1]  // CIRCUIT_DECOMP
    q.t %q2[1]  // CIRCUIT_DECOMP
    q.t %q2[1]  // CIRCUIT_DECOMP
    q.s %q2[0]  // CIRCUIT_DECOMP
    q.s %q2[0]  // CIRCUIT_DECOMP
    q.s %q2[0]  // CIRCUIT_DECOMP
    q.s %q2[0]  // CIRCUIT_DECOMP
    q.z %q2[3]  // CIRCUIT_DECOMP
    q.z %q2[3]  // CIRCUIT_DECOMP
    q.h %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.h %q2[0]  // CIRCUIT_DECOMP
    q.cx %q0[2], %q24[0]  // CIRCUIT_DECOMP
    q.cx %q1[2], %q24[0]  // CIRCUIT_DECOMP
    q.cx %q24[0], %q2[2]  // CIRCUIT_DECOMP
    q.cx %q20[1], %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q0[2], %q1[2], %q25[0]  // CIRCUIT_DECOMP
    q.ccx %q20[1], %q24[0], %q26[0]  // CIRCUIT_DECOMP
    q.cx %q25[0], %q20[2]  // CIRCUIT_DECOMP
    q.cx %q26[0], %q20[2]  // CIRCUIT_DECOMP
    q.comment   // Bit 3: Full adder (MSB)
    q.cx %q0[3], %q27[0]  // CIRCUIT_DECOMP
    q.cx %q1[3], %q27[0]  // CIRCUIT_DECOMP
    q.cx %q27[0], %q2[3]  // CIRCUIT_DECOMP
    q.cx %q20[2], %q2[3]  // CIRCUIT_DECOMP
    q.comment   // NAIVE STEP: Redundant CX pair
    q.cx %q0[0], %q1[0]  // CIRCUIT_DECOMP
    q.cx %q0[0], %q1[0]  // CIRCUIT_DECOMP
    q.comment   // === 4-BIT ADDITION COMPLETE ===
    q.comment   // Examples:
    q.comment   //   3 + 5 = 8  (0011 + 0101 = 1000)
    q.comment   //   7 + 9 = 0  (0111 + 1001 = 0000, mod 16)
    q.comment   //   15 + 15 = 14 (1111 + 1111 = 1110, mod 16)
    %q3 = q.alloc : !qreg<4>
    q.init %q3, 3 : i32
    %q4 = q.alloc : !qreg<4>
    // OPTIMIZATION: Decomposed sub_circuit into basic gates
    q.comment   // === COMPREHENSIVE 4-BIT QUANTUM SUBTRACTION (NAIVE) ===
    q.comment   // Computing: %q2 - %q3 -> %q4
    q.comment   // NAIVE STEP: Clearing result register with redundant pairs
    q.x %q4[0]  // CIRCUIT_DECOMP
    q.x %q4[0]  // CIRCUIT_DECOMP
    q.x %q4[1]  // CIRCUIT_DECOMP
    q.x %q4[1]  // CIRCUIT_DECOMP
    q.x %q4[2]  // CIRCUIT_DECOMP
    q.x %q4[2]  // CIRCUIT_DECOMP
    q.x %q4[3]  // CIRCUIT_DECOMP
    q.x %q4[3]  // CIRCUIT_DECOMP
    q.comment   // Step 2: Compute one's complement of B
    q.cx %q3[0], %q30[0]  // CIRCUIT_DECOMP
    q.x %q30[0]  // CIRCUIT_DECOMP
    q.cx %q3[1], %q30[1]  // CIRCUIT_DECOMP
    q.x %q30[1]  // CIRCUIT_DECOMP
    q.cx %q3[2], %q30[2]  // CIRCUIT_DECOMP
    q.x %q30[2]  // CIRCUIT_DECOMP
    q.cx %q3[3], %q30[3]  // CIRCUIT_DECOMP
    q.x %q30[3]  // CIRCUIT_DECOMP
    q.comment   // NAIVE STEP: Redundant operation on a temporary qubit
    q.h %q30[0]  // CIRCUIT_DECOMP
    q.h %q30[0]  // CIRCUIT_DECOMP
    q.comment   // Step 3: Compute A + B' + 1 (two's complement)
    q.x %q31[0]  // CIRCUIT_DECOMP
    q.comment   // Bit 0: A[0] + B'[0] + 1
    q.cx %q2[0], %q4[0]  // CIRCUIT_DECOMP
    q.cx %q30[0], %q4[0]  // CIRCUIT_DECOMP
    q.cx %q31[0], %q4[0]  // CIRCUIT_DECOMP
    q.ccx %q2[0], %q30[0], %q31[1]  // CIRCUIT_DECOMP
    q.ccx %q2[0], %q31[0], %q31[1]  // CIRCUIT_DECOMP
    q.ccx %q30[0], %q31[0], %q31[1]  // CIRCUIT_DECOMP
    q.comment   // Bit 1: A[1] + B'[1] + carry1
    q.cx %q2[1], %q4[1]  // CIRCUIT_DECOMP
    q.cx %q30[1], %q4[1]  // CIRCUIT_DECOMP
    q.cx %q31[1], %q4[1]  // CIRCUIT_DECOMP
    q.ccx %q2[1], %q30[1], %q31[2]  // CIRCUIT_DECOMP
    q.ccx %q2[1], %q31[1], %q31[2]  // CIRCUIT_DECOMP
    q.ccx %q30[1], %q31[1], %q31[2]  // CIRCUIT_DECOMP
    q.comment   // Bit 2: A[2] + B'[2] + carry2
    q.cx %q2[2], %q4[2]  // CIRCUIT_DECOMP
    q.cx %q30[2], %q4[2]  // CIRCUIT_DECOMP
    q.cx %q31[2], %q4[2]  // CIRCUIT_DECOMP
    q.ccx %q2[2], %q30[2], %q31[3]  // CIRCUIT_DECOMP
    q.ccx %q2[2], %q31[2], %q31[3]  // CIRCUIT_DECOMP
    q.ccx %q30[2], %q31[2], %q31[3]  // CIRCUIT_DECOMP
    q.comment   // Bit 3: A[3] + B'[3] + carry3
    q.cx %q2[3], %q4[3]  // CIRCUIT_DECOMP
    q.cx %q30[3], %q4[3]  // CIRCUIT_DECOMP
    q.cx %q31[3], %q4[3]  // CIRCUIT_DECOMP
    q.comment   // Final carry out (overflow bit - ignored for 4-bit)
    q.ccx %q2[3], %q30[3], %q32[0]  // CIRCUIT_DECOMP
    q.ccx %q2[3], %q31[3], %q32[0]  // CIRCUIT_DECOMP
    q.ccx %q30[3], %q31[3], %q32[0]  // CIRCUIT_DECOMP
    q.comment   // NAIVE STEP: Final redundant gate pair
    q.cx %q2[0], %q3[0]  // CIRCUIT_DECOMP
    q.cx %q2[0], %q3[0]  // CIRCUIT_DECOMP
    q.comment   // === SUBTRACTION COMPLETE ===
    q.comment   // Test cases:
    q.comment   // 6-3=3: A=0110, B=0011 -> A+B'+1 = 0110+1100+1 = 0011 ✓
    q.comment   // 5-2=3: A=0101, B=0010 -> A+B'+1 = 0101+1101+1 = 0011 ✓
    q.comment   // 8-5=3: A=1000, B=0101 -> A+B'+1 = 1000+1010+1 = 0011 ✓
    q.comment   // 15-12=3: A=1111, B=1100 -> A+B'+1 = 1111+0011+1 = 0011 ✓
    %q5 = q.measure %q4 : !qreg -> i32
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}