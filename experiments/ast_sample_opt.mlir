// Optimized Quantum MLIR
// Expected classical result: 7
builtin.module {
  "quantum.func"() ({
    %0 = "quantum.init"() {type = i32, value = 9 : i32} : () -> i32
    %1 = "quantum.init"() {type = i32, value = 2 : i32} : () -> i32
    %2 = "quantum.sub"(%0, %1) : (i32, i32) -> i32
    %3 = "quantum.measure"(%2) : (i32) -> i1
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}