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

// Helper function to implement greater-than-or-equal comparison
void emit_quantum_geq(QMLIR_Function& func,
                      const std::string& a,
                      const std::string& b,
                      const std::string& result, // 1-qubit output
                      int num_bits) {
    // 1) Build two's complement of b
    std::string b_inv = new_tmp("b_inv");
    emit_qubit_alloc(func, b_inv, num_bits);
    for (int i = 0; i < num_bits; i++) {
        // Copy b into b_inv
        func.ops.push_back({
            QOpKind::Custom, "", 
            b + "[" + std::to_string(i) + "]",
            b_inv + "[" + std::to_string(i) + "]", 
            0, "q.cx"
        });
        // Invert bits
        func.ops.push_back({
            QOpKind::Custom, "", 
            b_inv + "[" + std::to_string(i) + "]", 
            "", 
            0, "q.x"
        });
    }

    // 2) b_twos = b_inv + 1
    std::string one = new_tmp("one");
    emit_qubit_alloc(func, one, num_bits);
    emit_qubit_init(func, one, 1, num_bits);

    std::string b_twos = new_tmp("b2");
    emit_qubit_alloc(func, b_twos, num_bits);
    emit_quantum_adder(func, b_twos, b_inv, one, num_bits);

    // 3) diff = a + (b_twos) = a - b
    std::string diff = new_tmp("diff");
    emit_qubit_alloc(func, diff, num_bits);
    emit_quantum_adder(func, diff, a, b_twos, num_bits);

    // 4) diff[MSB] == 1 means negative => a < b
    // We want geq=1 if a >= b => invert that bit
    func.ops.push_back({
        QOpKind::Custom, "", 
        diff + "[" + std::to_string(num_bits - 1) + "]", 
        result + "[0]", 
        0, "q.cx"
    });

    // Invert to get geq
    func.ops.push_back({
        QOpKind::Custom, "", 
        result + "[0]", 
        "", 
        0, "q.x"
    });
}

void emit_quantum_divider(QMLIR_Function& func,
                          const std::string& result,
                          const std::string& a,
                          const std::string& b,
                          int num_bits) {
    // 1) Allocate registers for dividend, divisor, and the quotient
    std::string dividend = new_tmp("dvd");
    std::string divisor  = new_tmp("dvs");
    std::string quotient = new_tmp("q");

    emit_qubit_alloc(func, dividend, num_bits);
    emit_qubit_alloc(func, divisor,  num_bits);
    emit_qubit_alloc(func, quotient, num_bits);

    // 2) Copy 'a' into 'dividend', 'b' into 'divisor'
    for (int i = 0; i < num_bits; i++) {
        func.ops.push_back({
            QOpKind::Custom, "", 
            a + "[" + std::to_string(i) + "]",
            dividend + "[" + std::to_string(i) + "]", 
            0, "q.cx"
        });
        func.ops.push_back({
            QOpKind::Custom, "", 
            b + "[" + std::to_string(i) + "]",
            divisor + "[" + std::to_string(i) + "]", 
            0, "q.cx"
        });
    }

    // 3) Remainder is 2 * num_bits (for shifting, top half vs. bottom half)
    std::string remainder = new_tmp("r");
    emit_qubit_alloc(func, remainder, 2 * num_bits);

    // 4) Main loop: from MSB down to LSB
    for (int i = num_bits - 1; i >= 0; i--) {
        // 4a) Shift remainder left by 1
        std::string shifted_r = new_tmp("shifted_r");
        emit_qubit_alloc(func, shifted_r, 2 * num_bits);

        // Shift bits [0..(2N-2)] -> [1..(2N-1)]
        emit_quantum_shift(func, remainder, shifted_r, 1, 2 * num_bits - 1);

        // Bring down next dividend bit into shifted_r[0]
        func.ops.push_back({
            QOpKind::Custom, "",
            dividend + "[" + std::to_string(i) + "]",
            shifted_r + "[0]",
            0, "q.cx"
        });

        // 4b) Make updated_r = shifted_r (copy fully)
        std::string updated_r = new_tmp("upd_r");
        emit_qubit_alloc(func, updated_r, 2 * num_bits);
        for (int j = 0; j < 2 * num_bits; j++) {
            func.ops.push_back({
                QOpKind::Custom, "",
                shifted_r + "[" + std::to_string(j) + "]",
                updated_r + "[" + std::to_string(j) + "]",
                0, "q.cx"
            });
        }

        // 4c) Extract the top half of updated_r into upper_r
        std::string upper_r = new_tmp("upper_r");
        emit_qubit_alloc(func, upper_r, num_bits);
        for (int j = 0; j < num_bits; j++) {
            func.ops.push_back({
                QOpKind::Custom, "",
                updated_r + "[" + std::to_string(j + num_bits) + "]",
                upper_r + "[" + std::to_string(j) + "]",
                0, "q.cx"
            });
        }

        // 4d) Compare: upper_r >= divisor => cmp
        std::string cmp = new_tmp("cmp");
        emit_qubit_alloc(func, cmp, 1);
        emit_quantum_geq(func, upper_r, divisor, cmp, num_bits);

        // Copy cmp into quotient[i]
        func.ops.push_back({
            QOpKind::Custom, "",
            cmp + "[0]",
            quotient + "[" + std::to_string(i) + "]",
            0, "q.cx"
        });

        // 4e) Compute sub_result = upper_r - divisor
        //     = upper_r + two's complement of divisor
        //  i) Build neg_divisor
        std::string neg_divisor = new_tmp("neg_dvs");
        emit_qubit_alloc(func, neg_divisor, num_bits);

        for (int j = 0; j < num_bits; j++) {
            func.ops.push_back({
                QOpKind::Custom, "",
                divisor + "[" + std::to_string(j) + "]",
                neg_divisor + "[" + std::to_string(j) + "]",
                0, "q.cx"
            });
            // X to invert the bits
            func.ops.push_back({
                QOpKind::Custom, "",
                neg_divisor + "[" + std::to_string(j) + "]",
                "",
                0, "q.x"
            });
        }

        //   ii) Add 1 => two's complement
        std::string one = new_tmp("one");
        emit_qubit_alloc(func, one, num_bits);
        emit_qubit_init(func, one, 1, num_bits);

        std::string twos_comp = new_tmp("twos");
        emit_qubit_alloc(func, twos_comp, num_bits);
        emit_quantum_adder(func, twos_comp, neg_divisor, one, num_bits);

        //   iii) sub_result = upper_r + twos_comp
        std::string sub_result = new_tmp("sub_result");
        emit_qubit_alloc(func, sub_result, num_bits);
        emit_quantum_adder(func, sub_result, upper_r, twos_comp, num_bits);

        // 4f) Conditionally update remainder
        // new_r = updated_r, but with top half conditionally replaced
        std::string new_r = new_tmp("new_r");
        emit_qubit_alloc(func, new_r, 2 * num_bits);

        // First, copy the lower half (bits [0..(N-1)]) unconditionally
        for (int j = 0; j < num_bits; j++) {
            func.ops.push_back({
                QOpKind::Custom, "",
                updated_r + "[" + std::to_string(j) + "]",
                new_r + "[" + std::to_string(j) + "]",
                0, "q.cx"
            });
        }

        // For the upper half, conditionally choose between upper_r and sub_result
        // Start by copying upper_r for all bits
        for (int j = 0; j < num_bits; j++) {
            func.ops.push_back({
                QOpKind::Custom, "",
                upper_r + "[" + std::to_string(j) + "]",
                new_r + "[" + std::to_string(j + num_bits) + "]",
                0, "q.cx"
            });
        }

        // Then conditionally XOR with (sub_result XOR upper_r) when cmp==1
        for (int j = 0; j < num_bits; j++) {
            // tmpX = sub_result[j] XOR upper_r[j]
            std::string tmpX = new_tmp("tmpX");
            emit_qubit_alloc(func, tmpX, 1);

            func.ops.push_back({
                QOpKind::Custom, "",
                sub_result + "[" + std::to_string(j) + "]",
                tmpX + "[0]",
                0, "q.cx"
            });
            func.ops.push_back({
                QOpKind::Custom, "",
                upper_r + "[" + std::to_string(j) + "]",
                tmpX + "[0]",
                0, "q.cx"
            });

            // Now we do a controlled XOR with cmp
            std::string final_ctrl = new_tmp("final_ctrl");
            emit_qubit_alloc(func, final_ctrl, 1);

            // final_ctrl = cmp && tmpX
            func.ops.push_back({
                QOpKind::Custom,
                final_ctrl + "[0]",
                cmp + "[0]",
                tmpX + "[0]",
                0, "q.ccx"
            });

            // XOR final_ctrl into new_r's upper bit [j + num_bits]
            func.ops.push_back({
                QOpKind::Custom, "",
                final_ctrl + "[0]",
                new_r + "[" + std::to_string(j + num_bits) + "]",
                0, "q.cx"
            });
        }

        // 4g) remainder = new_r
        remainder = new_r;
    }

    // 5) Copy final quotient to 'result'
    for (int i = 0; i < num_bits; i++) {
        func.ops.push_back({
            QOpKind::Custom, "",
            quotient + "[" + std::to_string(i) + "]",
            result + "[" + std::to_string(i) + "]",
            0, "q.cx"
        });
    }
}
