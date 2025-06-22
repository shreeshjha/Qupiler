// Fixed Universal Optimized Gate-Level Quantum MLIR
// Applied optimizations: Circuit decomposition: 1 circuits decomposed into gates
builtin.module {
  "quantum.func"() ({
    %q0 = q.alloc : !qreg<4>
    q.init %q0, 9 : i32
    %q1 = q.alloc : !qreg<4>
    q.init %q1, 2 : i32
    %q2 = q.alloc : !qreg<4>
    // OPTIMIZATION: Decomposed sub_circuit into basic gates
    q.comment   // === COMPREHENSIVE 4-BIT QUANTUM SUBTRACTION ===
    q.comment   // Computing: %q0 - %q1 -> %q2
    q.comment   // Step 1: Clear result register
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[3]  // CIRCUIT_DECOMP
    q.comment   // Step 2: Compute one's complement of B
    q.cx %q1[0], %q20[0]  // CIRCUIT_DECOMP
    q.x %q20[0]  // CIRCUIT_DECOMP
    q.cx %q1[1], %q20[1]  // CIRCUIT_DECOMP
    q.x %q20[1]  // CIRCUIT_DECOMP
    q.cx %q1[2], %q20[2]  // CIRCUIT_DECOMP
    q.x %q20[2]  // CIRCUIT_DECOMP
    q.cx %q1[3], %q20[3]  // CIRCUIT_DECOMP
    q.x %q20[3]  // CIRCUIT_DECOMP
    q.comment   // Step 3: Compute A + B' + 1 (two's complement)
    q.x %q21[0]  // CIRCUIT_DECOMP
    q.comment   // Bit 0: A[0] + B'[0] + 1
    q.cx %q0[0], %q2[0]  // CIRCUIT_DECOMP
    q.cx %q20[0], %q2[0]  // CIRCUIT_DECOMP
    q.cx %q21[0], %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q0[0], %q20[0], %q21[1]  // CIRCUIT_DECOMP
    q.ccx %q0[0], %q21[0], %q21[1]  // CIRCUIT_DECOMP
    q.ccx %q20[0], %q21[0], %q21[1]  // CIRCUIT_DECOMP
    q.comment   // Bit 1: A[1] + B'[1] + carry1
    q.cx %q0[1], %q2[1]  // CIRCUIT_DECOMP
    q.cx %q20[1], %q2[1]  // CIRCUIT_DECOMP
    q.cx %q21[1], %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q0[1], %q20[1], %q21[2]  // CIRCUIT_DECOMP
    q.ccx %q0[1], %q21[1], %q21[2]  // CIRCUIT_DECOMP
    q.ccx %q20[1], %q21[1], %q21[2]  // CIRCUIT_DECOMP
    q.comment   // Bit 2: A[2] + B'[2] + carry2
    q.cx %q0[2], %q2[2]  // CIRCUIT_DECOMP
    q.cx %q20[2], %q2[2]  // CIRCUIT_DECOMP
    q.cx %q21[2], %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q0[2], %q20[2], %q21[3]  // CIRCUIT_DECOMP
    q.ccx %q0[2], %q21[2], %q21[3]  // CIRCUIT_DECOMP
    q.ccx %q20[2], %q21[2], %q21[3]  // CIRCUIT_DECOMP
    q.comment   // Bit 3: A[3] + B'[3] + carry3
    q.cx %q0[3], %q2[3]  // CIRCUIT_DECOMP
    q.cx %q20[3], %q2[3]  // CIRCUIT_DECOMP
    q.cx %q21[3], %q2[3]  // CIRCUIT_DECOMP
    q.comment   // Final carry out (overflow bit - ignored for 4-bit)
    q.ccx %q0[3], %q20[3], %q22[0]  // CIRCUIT_DECOMP
    q.ccx %q0[3], %q21[3], %q22[0]  // CIRCUIT_DECOMP
    q.ccx %q20[3], %q21[3], %q22[0]  // CIRCUIT_DECOMP
    q.comment   // === SUBTRACTION COMPLETE ===
    q.comment   // Test cases:
    q.comment   // 6-3=3: A=0110, B=0011 -> A+B'+1 = 0110+1100+1 = 0011 ✓
    q.comment   // 5-2=3: A=0101, B=0010 -> A+B'+1 = 0101+1101+1 = 0011 ✓
    q.comment   // 8-5=3: A=1000, B=0101 -> A+B'+1 = 1000+1010+1 = 0011 ✓
    q.comment   // 15-12=3: A=1111, B=1100 -> A+B'+1 = 1111+0011+1 = 0011 ✓
    %q3 = q.measure %q2 : !qreg -> i32
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}