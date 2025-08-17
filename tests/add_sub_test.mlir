builtin.module {
  "quantum.func"() ({
    %0 = "quantum.init"() {type = i32, value = 2 : i32} : () -> i32
    %1 = "quantum.init"() {type = i32, value = 7 : i32} : () -> i32
    %2 = "quantum.add"(%0, %1) : (i32, i32) -> i32
    %3 = "quantum.init"() {type = i32, value = 3 : i32} : () -> i32
    %4 = "quantum.sub"(%2, %3) : (i32, i32) -> i32
    %5 = "quantum.measure"(%4) : (i32) -> i1
    func.return
  }) {func_name = "quantum_circuit"} : () -> ()
}