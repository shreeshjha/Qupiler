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

// Shifts a register 'src' by 'shift' bits into register 'dst'.
// Both src and dst should have at least (shift + num_bits) qubits allocated.
void emit_quantum_shift(QMLIR_Function& func,
                        const std::string& src,
                        const std::string& dst,
                        int shift,
                        int num_bits) {
    // Copy src[i] into dst[i+shift] using CX gates.
    // We assume dst is already allocated and 0-initialized.
    for (int i = 0; i < num_bits; i++) {
        // Move src[i] → dst[i+shift]
        // First do dst[i+shift] = dst[i+shift] XOR src[i].
        // Because dst[i+shift] is initially 0, it just becomes src[i].
        func.ops.push_back({
            QOpKind::Custom,
            "",                           // no explicit 'result' variable
            src + "[" + std::to_string(i) + "]",
            dst + "[" + std::to_string(i + shift) + "]",
            0,
            "q.cx"
        });
    }
}

// void emit_quantum_multiplier(QMLIR_Function& func, const std::string& result, 
//                            const std::string& a, const std::string& b, int num_bits) {
//     // For small numbers like 1×2, 2×3, etc., we can use a very direct approach
//     // that essentially mimics the classical multiplication algorithm
    
//     // Create an accumulator register (initialized to 0)
//     std::string acc = new_tmp("acc");
//     emit_qubit_alloc(func, acc, 2 * num_bits);
    
//     // For bit 0 of the multiplier (b)
//     // If b[0] is 1, add a to the accumulator
//     for (int j = 0; j < num_bits; j++) {
//         // Create a control qubit
//         std::string ctl0 = new_tmp("ctl");
//         emit_qubit_alloc(func, ctl0, 1);
        
//         // Set ctl0 to 1 if b[0] AND a[j] is 1
//         func.ops.push_back({QOpKind::Custom, ctl0 + "[0]", 
//                            b + "[0]", 
//                            a + "[" + std::to_string(j) + "]", 
//                            0, "q.ccx"});
        
//         // XOR ctl0 into the accumulator at position j
//         func.ops.push_back({QOpKind::Custom, "", 
//                            ctl0 + "[0]", 
//                            acc + "[" + std::to_string(j) + "]", 
//                            0, "q.cx"});
//     }
    
//     // For bit 1 of the multiplier (b)
//     // If b[1] is 1, add a << 1 (shifted left by 1) to the accumulator
//     for (int j = 0; j < num_bits; j++) {
//         // Create a control qubit
//         std::string ctl1 = new_tmp("ctl");
//         emit_qubit_alloc(func, ctl1, 1);
        
//         // Set ctl1 to 1 if b[1] AND a[j] is 1
//         func.ops.push_back({QOpKind::Custom, ctl1 + "[0]", 
//                            b + "[1]", 
//                            a + "[" + std::to_string(j) + "]", 
//                            0, "q.ccx"});
        
//         // XOR ctl1 into the accumulator at position j+1
//         func.ops.push_back({QOpKind::Custom, "", 
//                            ctl1 + "[0]", 
//                            acc + "[" + std::to_string(j+1) + "]", 
//                            0, "q.cx"});
//     }
    
//     // For bit 2 of the multiplier (b)
//     // If b[2] is 1, add a << 2 (shifted left by 2) to the accumulator
//     for (int j = 0; j < num_bits; j++) {
//         if (j+2 < 2*num_bits) {
//             // Create a control qubit
//             std::string ctl2 = new_tmp("ctl");
//             emit_qubit_alloc(func, ctl2, 1);
            
//             // Set ctl2 to 1 if b[2] AND a[j] is 1
//             func.ops.push_back({QOpKind::Custom, ctl2 + "[0]", 
//                                b + "[2]", 
//                                a + "[" + std::to_string(j) + "]", 
//                                0, "q.ccx"});
            
//             // XOR ctl2 into the accumulator at position j+2
//             func.ops.push_back({QOpKind::Custom, "", 
//                                ctl2 + "[0]", 
//                                acc + "[" + std::to_string(j+2) + "]", 
//                                0, "q.cx"});
//         }
//     }
    
//     // For bit 3 of the multiplier (b)
//     // If b[3] is 1, add a << 3 (shifted left by 3) to the accumulator
//     for (int j = 0; j < num_bits; j++) {
//         if (j+3 < 2*num_bits) {
//             // Create a control qubit
//             std::string ctl3 = new_tmp("ctl");
//             emit_qubit_alloc(func, ctl3, 1);
            
//             // Set ctl3 to 1 if b[3] AND a[j] is 1
//             func.ops.push_back({QOpKind::Custom, ctl3 + "[0]", 
//                                b + "[3]", 
//                                a + "[" + std::to_string(j) + "]", 
//                                0, "q.ccx"});
            
//             // XOR ctl3 into the accumulator at position j+3
//             func.ops.push_back({QOpKind::Custom, "", 
//                                ctl3 + "[0]", 
//                                acc + "[" + std::to_string(j+3) + "]", 
//                                0, "q.cx"});
//         }
//     }
    
//     // Now copy the lower bits of the accumulator to the result
//     for (int i = 0; i < num_bits; i++) {
//         func.ops.push_back({QOpKind::Custom, "", 
//                            acc + "[" + std::to_string(i) + "]", 
//                            result + "[" + std::to_string(i) + "]", 
//                            0, "q.cx"});
//     }
// }

void emit_quantum_multiplier(QMLIR_Function& func, const std::string& result, 
                             const std::string& a, const std::string& b, int num_bits) {
    int total_bits = 2 * num_bits;

    // Create accumulator
    std::string acc = new_tmp("acc");
    emit_qubit_alloc(func, acc, total_bits);

    // Loop over bits of multiplier
    for (int i = 0; i < num_bits; ++i) {
        // Shifted partial product result
        std::string partial = new_tmp("pp");
        emit_qubit_alloc(func, partial, total_bits);

        for (int j = 0; j < num_bits; ++j) {
            int k = i + j;
            if (k >= total_bits) continue;

            // control = a[j] AND b[i]
            std::string ctl = new_tmp("ctl");
            emit_qubit_alloc(func, ctl, 1);
            func.ops.push_back({QOpKind::Custom, ctl + "[0]", b + "[" + std::to_string(i) + "]",
                                a + "[" + std::to_string(j) + "]", 0, "q.ccx"});

            // partial[k] ^= ctl
            func.ops.push_back({QOpKind::Custom, "", ctl + "[0]", partial + "[" + std::to_string(k) + "]", 0, "q.cx"});
        }

        // Add partial to accumulator with carry
        std::string new_acc = new_tmp("acc");
        emit_qubit_alloc(func, new_acc, total_bits);
        emit_quantum_adder(func, new_acc, acc, partial, total_bits);

        acc = new_acc;
    }

    // Copy lower `num_bits` bits from accumulator to result
    for (int i = 0; i < num_bits; ++i) {
        func.ops.push_back({QOpKind::Custom, "", acc + "[" + std::to_string(i) + "]",
                            result + "[" + std::to_string(i) + "]", 0, "q.cx"});
    }
}
