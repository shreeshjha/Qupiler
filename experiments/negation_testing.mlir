builtin.module {
  "quantum.func"() ({
    %0 = "quantum.init"() {type = i32, value = 1 : i32} : () -> i32
    %1 = "quantum.neg"(%0) : (i32) -> i32
    %2 = "quantum.init"() {type = i32, value = 5 : i32} : () -> i32
    %3 = "quantum.neg"(%2) : (i32) -> i32
    %4 = "quantum.add"(%0, %3) : (i32, i32) -> i32
    %5 = "quantum.measure"(%4) : (i32) -> i1
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}