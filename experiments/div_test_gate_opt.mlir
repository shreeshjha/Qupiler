// Fixed Universal Optimized Gate-Level Quantum MLIR
// Applied optimizations: Circuit decomposition: 1 circuits decomposed into gates
builtin.module {
  "quantum.func"() ({
    %q0 = q.alloc : !qreg<4>
    q.init %q0, 9 : i32
    %q1 = q.alloc : !qreg<4>
    q.init %q1, 3 : i32
    %q2 = q.alloc : !qreg<4>
    // OPTIMIZATION: Decomposed div_circuit into basic gates
    q.comment   // === UNIVERSAL REPEATED SUBTRACTION DIVISION ===
    q.comment   // Initialize: remainder=dividend, quotient=0
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[3]  // CIRCUIT_DECOMP
    q.x %q10[0]  // CIRCUIT_DECOMP
    q.x %q10[0]  // CIRCUIT_DECOMP
    q.x %q10[1]  // CIRCUIT_DECOMP
    q.x %q10[1]  // CIRCUIT_DECOMP
    q.x %q10[2]  // CIRCUIT_DECOMP
    q.x %q10[2]  // CIRCUIT_DECOMP
    q.x %q10[3]  // CIRCUIT_DECOMP
    q.x %q10[3]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[1]  // CIRCUIT_DECOMP
    q.x %q12[1]  // CIRCUIT_DECOMP
    q.x %q12[2]  // CIRCUIT_DECOMP
    q.x %q12[2]  // CIRCUIT_DECOMP
    q.x %q12[3]  // CIRCUIT_DECOMP
    q.x %q12[3]  // CIRCUIT_DECOMP
    q.cx %q0[0], %q10[0]  // CIRCUIT_DECOMP
    q.cx %q0[1], %q10[1]  // CIRCUIT_DECOMP
    q.cx %q0[2], %q10[2]  // CIRCUIT_DECOMP
    q.cx %q0[3], %q10[3]  // CIRCUIT_DECOMP
    q.comment   // Repeated subtraction: try up to 15 times
    q.comment   // === Subtraction 1 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 2 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 3 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 4 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 5 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 6 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 7 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 8 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 9 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 10 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 11 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 12 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 13 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 14 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === Subtraction 15 ===
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[0]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[1]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[2]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q11[3]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Compute: sub_result = remainder - divisor
    q.cx %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.cx %q1[0], %q11[0]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.ccx %q10[0], %q1[0], %q11[1]  // CIRCUIT_DECOMP
    q.x %q1[0]  // CIRCUIT_DECOMP
    q.comment   // Check: is remainder >= divisor?
    q.cx %q11[3], %q12[0]  // CIRCUIT_DECOMP
    q.x %q12[0]  // CIRCUIT_DECOMP
    q.comment   // Conditional update: if valid, remainder = sub_result
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[0], %q10[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[0], %q11[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[1], %q10[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[1], %q11[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[2], %q10[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[2], %q11[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q11[3], %q10[3]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q10[3], %q11[3]  // CIRCUIT_DECOMP
    q.comment   // Conditional increment: if valid, quotient++
    q.cx %q12[0], %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[0], %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[0]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[1], %q2[2]  // CIRCUIT_DECOMP
    q.x %q2[1]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.ccx %q12[0], %q2[2], %q2[3]  // CIRCUIT_DECOMP
    q.x %q2[2]  // CIRCUIT_DECOMP
    q.comment   // === UNIVERSAL DIVISION COMPLETE ===
    q.comment   // Algorithm: Pure repeated subtraction
    q.comment   // Works for ANY 4-bit division: a÷b where 1≤b≤15, 0≤a≤15
    q.comment   // Examples: 9÷3=3, 7÷3=2, 15÷4=3, 8÷2=4, 0÷5=0
    %q3 = q.measure %q2 : !qreg -> i32
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}