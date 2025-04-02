#include "dialect.hpp"
#include "utils.hpp"
#include <vector>

void emit_qubit_alloc(QMLIR_Function& func, const std::string& tmp, int size) {
    func.ops.push_back({QOpKind::Custom, tmp, "", "", 0, "q.alloc", size});
}

void emit_qubit_init(QMLIR_Function& func, const std::string& qubit_tmp, int value, int size) {
    func.ops.push_back({QOpKind::Custom, qubit_tmp, "", "", value, "q.init", size});
}

void emit_measure(QMLIR_Function& func, const std::string& qubit_tmp, const std::string& result_tmp) {
    func.ops.push_back({QOpKind::Custom, result_tmp, qubit_tmp, "", 0, "q.measure"});
}

void emit_quantum_adder(QMLIR_Function& func, const std::string& result,
                        const std::string& a, const std::string& b, int num_bits) {
     // Create a carry register with num_bits+1 qubits (including carry-in bit)
     std::string carry = new_tmp("c");
     emit_qubit_alloc(func, carry, num_bits + 1);
     
     // Implement a proper quantum ripple-carry adder
     
     // Bit 0 (least significant bit)
     // First step: compute a[0] XOR b[0] and store in result[0]
     func.ops.push_back({QOpKind::Custom, "", a + "[0]", result + "[0]", 0, "q.cx"});
     func.ops.push_back({QOpKind::Custom, "", b + "[0]", result + "[0]", 0, "q.cx"});
     
     // Second step: compute carry-out c[1] = a[0] AND b[0]
     func.ops.push_back({QOpKind::Custom, carry + "[1]", a + "[0]", b + "[0]", 0, "q.ccx"});
     
     // For bits 1 through num_bits-1
     for (int i = 1; i < num_bits; i++) {
         // First compute the XOR: result[i] = a[i] XOR b[i]
         func.ops.push_back({QOpKind::Custom, "", a + "[" + std::to_string(i) + "]", 
                             result + "[" + std::to_string(i) + "]", 0, "q.cx"});
         func.ops.push_back({QOpKind::Custom, "", b + "[" + std::to_string(i) + "]", 
                             result + "[" + std::to_string(i) + "]", 0, "q.cx"});
         
         // Compute the carry-out: c[i+1] = (a[i] AND b[i]) OR (c[i] AND (a[i] OR b[i]))
         // We're using the fact that (a[i] OR b[i]) = (a[i] XOR b[i]) XOR (a[i] AND b[i])
         
         // First part: a[i] AND b[i] -> c[i+1]
         func.ops.push_back({QOpKind::Custom, carry + "[" + std::to_string(i+1) + "]", 
                             a + "[" + std::to_string(i) + "]", 
                             b + "[" + std::to_string(i) + "]", 0, "q.ccx"});
         
         // Second part: c[i] AND (a[i] XOR b[i]) -> temp
         // Note: result[i] already contains a[i] XOR b[i]
         func.ops.push_back({QOpKind::Custom, carry + "[" + std::to_string(i+1) + "]", 
                             result + "[" + std::to_string(i) + "]", 
                             carry + "[" + std::to_string(i) + "]", 0, "q.ccx"});
         
         // Third part: complete the carry-out computation by XORing with the carry-in
         func.ops.push_back({QOpKind::Custom, "", carry + "[" + std::to_string(i) + "]", 
                             result + "[" + std::to_string(i) + "]", 0, "q.cx"});
    }
}

void emit_quantum_subtractor(QMLIR_Function& func, const std::string& result,
                             const std::string& a, const std::string& b, int num_bits) {
    // Compute ~b.
    std::string b_inv = new_tmp("inv");
    emit_qubit_alloc(func, b_inv, num_bits);
    for (int i = 0; i < num_bits; ++i) {
        func.ops.push_back({QOpKind::Custom, "", b + "[" + std::to_string(i) + "]",
                            b_inv + "[" + std::to_string(i) + "]", 0, "q.cx"});
        func.ops.push_back({QOpKind::Custom, "", b_inv + "[" + std::to_string(i) + "]", "", 0, "q.x"});
    }
    // Create constant 1.
    std::string plus_one = new_tmp("one");
    emit_qubit_alloc(func, plus_one, num_bits);
    emit_qubit_init(func, plus_one, 1, num_bits);
    // Compute two's complement of b: ~b + 1.
    std::string b_twos = new_tmp("b2");
    emit_qubit_alloc(func, b_twos, num_bits);
    emit_quantum_adder(func, b_twos, b_inv, plus_one, num_bits);
    // Now, subtract: a + (~b + 1)
    emit_quantum_adder(func, result, a, b_twos, num_bits);
}

