// Advanced Quantum Gate Optimized MLIR
// Original gates: 91
// Optimized gates: 57
// Applied optimizations: X(10), CX(4), CCX(0), DCE(0)
builtin.module {
  "quantum.func"() ({
    %q0 = q.alloc : !qreg<4>
    %q1 = q.alloc : !qreg<4>
    %q2 = q.alloc : !qreg<4>
    %q3 = q.alloc : !qreg<4>
    %q4 = q.alloc : !qreg<4>
    q.init %q0, 2 : i32
    q.init %q1, 7 : i32
    q.init %q3, 3 : i32
    q.cx %q0[0], %q2[0]
    q.cx %q1[0], %q2[0]
    q.ccx %q0[0], %q1[0], %q20[0]
    q.cx %q0[1], %q21[0]
    q.cx %q1[1], %q21[0]
    q.cx %q21[0], %q2[1]
    q.cx %q20[0], %q2[1]
    q.ccx %q0[1], %q1[1], %q22[0]
    q.ccx %q20[0], %q21[0], %q23[0]
    q.cx %q22[0], %q20[1]
    q.cx %q23[0], %q20[1]
    q.z %q2[0]  // HXH_TO_Z_CONVERTED
    q.cx %q0[2], %q24[0]
    q.cx %q1[2], %q24[0]
    q.cx %q24[0], %q2[2]
    q.cx %q20[1], %q2[2]
    q.ccx %q0[2], %q1[2], %q25[0]
    q.ccx %q20[1], %q24[0], %q26[0]
    q.cx %q25[0], %q20[2]
    q.cx %q26[0], %q20[2]
    q.cx %q0[3], %q27[0]
    q.cx %q1[3], %q27[0]
    q.cx %q27[0], %q2[3]
    q.cx %q20[2], %q2[3]
    q.cx %q3[0], %q30[0]
    q.x %q30[0]
    q.cx %q3[1], %q30[1]
    q.x %q30[1]
    q.cx %q3[2], %q30[2]
    q.x %q30[2]
    q.cx %q3[3], %q30[3]
    q.x %q30[3]
    q.x %q31[0]
    q.cx %q2[0], %q4[0]
    q.cx %q30[0], %q4[0]
    q.cx %q31[0], %q4[0]
    q.ccx %q2[0], %q30[0], %q31[1]
    q.ccx %q2[0], %q31[0], %q31[1]
    q.ccx %q30[0], %q31[0], %q31[1]
    q.cx %q2[1], %q4[1]
    q.cx %q30[1], %q4[1]
    q.cx %q31[1], %q4[1]
    q.ccx %q2[1], %q30[1], %q31[2]
    q.ccx %q2[1], %q31[1], %q31[2]
    q.ccx %q30[1], %q31[1], %q31[2]
    q.cx %q2[2], %q4[2]
    q.cx %q30[2], %q4[2]
    q.cx %q31[2], %q4[2]
    q.ccx %q2[2], %q30[2], %q31[3]
    q.ccx %q2[2], %q31[2], %q31[3]
    q.ccx %q30[2], %q31[2], %q31[3]
    q.cx %q2[3], %q4[3]
    q.cx %q30[3], %q4[3]
    q.cx %q31[3], %q4[3]
    q.ccx %q2[3], %q30[3], %q32[0]
    q.ccx %q2[3], %q31[3], %q32[0]
    q.ccx %q30[3], %q31[3], %q32[0]
    %q5 = q.measure %q4 : !qreg -> i32
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}