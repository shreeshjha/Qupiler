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




void emit_quantum_divider(QMLIR_Function& func,
                          const std::string& result,
                          const std::string& a,
                          const std::string& b,
                          int num_bits) {
    // IMPORTANT: In quantum circuits with LSB-first representation:
    // - Qubit[0] is the least significant bit (LSB)
    // - Qubit[num_bits-1] is the most significant bit (MSB)
    // - Division algorithm processes bits from MSB to LSB
    
    // Initialize quotient and remainder registers
    std::string quotient = new_tmp("quot");
    emit_qubit_alloc(func, quotient, num_bits);
    emit_qubit_init(func, quotient, 0, num_bits);
    
    std::string remainder = new_tmp("rem");
    emit_qubit_alloc(func, remainder, num_bits);
    emit_qubit_init(func, remainder, 0, num_bits);
    
    // Process bits from MSB down to LSB
    for (int i = num_bits - 1; i >= 0; i--) {
        // Create ancillas for this iteration
        std::string shifted_rem = new_tmp("srem");
        emit_qubit_alloc(func, shifted_rem, num_bits);
        emit_qubit_init(func, shifted_rem, 0, num_bits);
        
        // Step 1: Shift remainder left by 1 bit
        // (copy remainder[j] → shifted_rem[j+1], from top down to avoid overwriting)
        for (int j = num_bits - 2; j >= 0; j--) {
            func.ops.push_back({
                QOpKind::Custom, "", 
                remainder + "[" + std::to_string(j) + "]",
                shifted_rem + "[" + std::to_string(j + 1) + "]",
                0, "q.cx"
            });
        }
        
        // Step 2: Bring down next bit from dividend into the LSB of shifted_rem
        func.ops.push_back({
            QOpKind::Custom, "",
            a + "[" + std::to_string(i) + "]",
            shifted_rem + "[0]",
            0, "q.cx"
        });
        
        // Step 3: Compare shifted_rem with divisor by subtracting:  diff = shifted_rem - b
        std::string diff = new_tmp("diff");
        emit_qubit_alloc(func, diff, num_bits);
        emit_quantum_subtractor(func, diff, shifted_rem, b, num_bits);
        
        // Step 4: geq = NOT( diff[MSB] )  -- sign bit is diff[num_bits-1]
        // If sign bit is 1 => remainder went negative => shifted_rem < divisor
        // If sign bit is 0 => shifted_rem >= divisor
        std::string geq = new_tmp("geq");
        emit_qubit_alloc(func, geq, 1);
        
        // Copy sign bit into geq and then invert
        func.ops.push_back({
            QOpKind::Custom, "",
            diff + "[" + std::to_string(num_bits - 1) + "]",
            geq + "[0]",
            0, "q.cx"
        });
        func.ops.push_back({
            QOpKind::Custom, "",
            geq + "[0]",
            "",
            0, "q.x" // invert
        });
        
        // Step 5: Set the quotient bit at position i
        func.ops.push_back({
            QOpKind::Custom, "",
            geq + "[0]",
            quotient + "[" + std::to_string(i) + "]",
            0, "q.cx"
        });
        
        // Step 6: Update remainder = (geq == 1) ? diff : shifted_rem
        
        // 6a. Allocate new remainder
        std::string new_rem = new_tmp("nrem");
        emit_qubit_alloc(func, new_rem, num_bits);
        
        // 6b. Copy shifted_rem into new_rem
        for (int j = 0; j < num_bits; j++) {
            func.ops.push_back({
                QOpKind::Custom, "",
                shifted_rem + "[" + std::to_string(j) + "]",
                new_rem + "[" + std::to_string(j) + "]",
                0, "q.cx"
            });
        }
        
        // 6c. Multiplex each bit: if geq=1, new_rem[j] should become diff[j]
        for (int j = 0; j < num_bits; j++) {
            // delta = diff[j] XOR new_rem[j]
            std::string delta = new_tmp("delta");
            emit_qubit_alloc(func, delta, 1);

            // delta ← diff[j] ⊕ new_rem[j]
            func.ops.push_back({QOpKind::Custom, "", diff + "[" + std::to_string(j) + "]", delta + "[0]", 0, "q.cx"});
            func.ops.push_back({QOpKind::Custom, "", new_rem + "[" + std::to_string(j) + "]", delta + "[0]", 0, "q.cx"});

            // temp = geq AND delta
            std::string temp = new_tmp("temp");
            emit_qubit_alloc(func, temp, 1);

            func.ops.push_back({QOpKind::Custom, temp + "[0]", geq + "[0]", delta + "[0]", 0, "q.ccx"});

            // new_rem[j] ^= temp
            func.ops.push_back({QOpKind::Custom, "", temp + "[0]", new_rem + "[" + std::to_string(j) + "]", 0, "q.cx"});
        }
        
        // Update remainder for the next iteration
        remainder = new_rem;
    }
    
    // Finally, copy the quotient out into `result`
    for (int i = 0; i < num_bits; i++) {
        func.ops.push_back({
            QOpKind::Custom, "",
            quotient + "[" + std::to_string(i) + "]",
            result + "[" + std::to_string(i) + "]",
            0, "q.cx"
        });
    }
}


void emit_quantum_modulo(QMLIR_Function& func, const std::string& result, const std::string& a, const std::string& b, int num_bits) {
    // a % b = a - (b * (a / b)) 
    // Step 1: Compute quotient = a / b
    std::string quotient = new_tmp("quot");
    emit_qubit_alloc(func, quotient, num_bits);
    emit_quantum_divider(func, quotient, a, b, num_bits);
    
    // Step 2: Compute product = b * quotient
    std::string product = new_tmp("prod");
    emit_qubit_alloc(func, product, num_bits);
    emit_quantum_multiplier(func, product, b, quotient, num_bits);
    
    // Step 3: Compute remainder = a - product
    emit_quantum_subtractor(func, result, a, product, num_bits);   
}

void emit_quantum_negate(QMLIR_Function& func, const std::string& result, 
                          const std::string& input, int num_bits) {
    // Two's complement negation: invert all bits and add 1
    
    // Step 1: Create a temporary register for the inverted bits
    std::string inverted = new_tmp("inv");
    emit_qubit_alloc(func, inverted, num_bits);
    
    // Step 2: Invert all bits (apply X gate to each qubit)
    for (int i = 0; i < num_bits; i++) {
        // First copy the input to inverted using CNOT
        func.ops.push_back({QOpKind::Custom, "", input + "[" + std::to_string(i) + "]", 
                           inverted + "[" + std::to_string(i) + "]", 0, "q.cx"});
        
        // Then invert the bits using X gate
        func.ops.push_back({QOpKind::Custom, "", inverted + "[" + std::to_string(i) + "]", 
                           "", 0, "q.x"});
    }
    
    // Step 3: Create constant 1
    std::string plus_one = new_tmp("one");
    emit_qubit_alloc(func, plus_one, num_bits);
    emit_qubit_init(func, plus_one, 1, num_bits);
    
    // Step 4: Add 1 to the inverted bits to get two's complement
    emit_quantum_adder(func, result, inverted, plus_one, num_bits);
}

void emit_quantum_increment(QMLIR_Function& func, const std::string& result, 
                            const std::string& input, int num_bits) {
    // Create constant 1
    std::string plus_one = new_tmp("one");
    emit_qubit_alloc(func, plus_one, num_bits);
    emit_qubit_init(func, plus_one, 1, num_bits);
    
    // Add 1 to input
    emit_quantum_adder(func, result, input, plus_one, num_bits);
}

void emit_quantum_decrement(QMLIR_Function& func, const std::string& result, 
                            const std::string& input, int num_bits) {
    // Create constant 1
    std::string minus_one = new_tmp("one");
    emit_qubit_alloc(func, minus_one, num_bits);
    emit_qubit_init(func, minus_one, 1, num_bits);
    
    // Subtract 1 from input
    emit_quantum_subtractor(func, result, input, minus_one, num_bits);
}


void emit_quantum_and(QMLIR_Function& func, const std::string& result,
                      const std::string& a, const std::string& b, int num_bits) {
    // Perform bitwise AND between a and b, storing the result in result
    // Each bit in result will be the AND of the corresponding bits in a and b
    
    for (int i = 0; i < num_bits; i++) {
        // We need a temporary qubit for each bit position
        std::string temp = new_tmp("and_tmp");
        emit_qubit_alloc(func, temp, 1);
        
        // Use Toffoli gate (CCNOT) to compute a[i] AND b[i] into temp[0]
        // The Toffoli gate performs target = target ⊕ (control1 AND control2)
        // Since temp[0] is initially 0, it becomes (a[i] AND b[i])
        func.ops.push_back({
            QOpKind::Custom,
            temp + "[0]",  // target
            a + "[" + std::to_string(i) + "]",  // control1
            b + "[" + std::to_string(i) + "]",  // control2
            0,
            "q.ccx"  // Toffoli gate (CCNOT)
        });
        
        // Copy the result from temp to the output register
        func.ops.push_back({
            QOpKind::Custom,
            "",  // no explicit result
            temp + "[0]",  // control
            result + "[" + std::to_string(i) + "]",  // target
            0,
            "q.cx"  // CNOT gate
        });
    }
}