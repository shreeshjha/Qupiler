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
    std::vector<std::string> carry_bits;
    for (int i = 0; i < num_bits; ++i) {
        std::string c = new_tmp("anc");
        emit_qubit_alloc(func, c, 1);
        carry_bits.push_back(c);
    }

    // Copy register 'a' into result via CNOTs.
    for (int i = 0; i < num_bits; ++i) {
        func.ops.push_back({QOpKind::Custom, "", a + "[" + std::to_string(i) + "]",
                            result + "[" + std::to_string(i) + "]", 0, "q.cx"});
    }

    // Add b into result.
    for (int i = 0; i < num_bits; ++i) {
        func.ops.push_back({QOpKind::Custom, "", b + "[" + std::to_string(i) + "]",
                            result + "[" + std::to_string(i) + "]", 0, "q.cx"});
        func.ops.push_back({QOpKind::Custom, carry_bits[i] + "[0]",
                            result + "[" + std::to_string(i) + "]",
                            b + "[" + std::to_string(i) + "]", 0, "q.ccx"});
    }

    // Propagate carries.
    for (int i = 1; i < num_bits; ++i) {
        func.ops.push_back({QOpKind::Custom, "", carry_bits[i - 1] + "[0]",
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

