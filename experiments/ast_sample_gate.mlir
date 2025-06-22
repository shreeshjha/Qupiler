// Comprehensive Gate-Level Quantum MLIR
// Converted 4 high-level operations
// Operation types: init, measure, sub
// Total quantum registers: 4
builtin.module {
  "quantum.func"() ({
    %q0 = q.alloc : !qreg<4>
    q.init %q0, 9 : i32
    %q1 = q.alloc : !qreg<4>
    q.init %q1, 2 : i32
    %q2 = q.alloc : !qreg<4>
    q.sub_circuit %q0, %q1, %q2
    %q3 = q.measure %q2 : !qreg -> i32
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}