// Comprehensive Gate-Level Quantum MLIR
// Converted 6 high-level operations
// Operation types: add, init, measure, sub
// Total quantum registers: 6
builtin.module {
  "quantum.func"() ({
    %q0 = q.alloc : !qreg<4>
    q.init %q0, 2 : i32
    %q1 = q.alloc : !qreg<4>
    q.init %q1, 7 : i32
    %q2 = q.alloc : !qreg<4>
    q.add_circuit %q0, %q1, %q2
    %q3 = q.alloc : !qreg<4>
    q.init %q3, 3 : i32
    %q4 = q.alloc : !qreg<4>
    q.sub_circuit %q2, %q3, %q4
    %q5 = q.measure %q4 : !qreg -> i32
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}