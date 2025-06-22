// Comprehensive Gate-Level Quantum MLIR
// Converted 5 high-level operations
// Operation types: add, init, measure, neg
// Total quantum registers: 5
builtin.module {
  "quantum.func"() ({
    %q0 = q.alloc : !qreg<4>
    q.init %q0, 1 : i32
    %q1 = q.alloc : !qreg<4>
    q.init %q1, 5 : i32
    %q2 = q.alloc : !qreg<4>
    q.neg_circuit %q1, %q2
    %q3 = q.alloc : !qreg<4>
    q.add_circuit %q0, %q2, %q3
    %q4 = q.measure %q3 : !qreg -> i32
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}