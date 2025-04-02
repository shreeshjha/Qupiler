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

// void emit_quantum_subtractor(QMLIR_Function& func, const std::string& result,
//                              const std::string& a, const std::string& b, int num_bits) {
//     // Compute ~b.
//     std::string b_inv = new_tmp("inv");
//     emit_qubit_alloc(func, b_inv, num_bits);
//     for (int i = 0; i < num_bits; ++i) {
//         func.ops.push_back({QOpKind::Custom, "", b + "[" + std::to_string(i) + "]",
//                             b_inv + "[" + std::to_string(i) + "]", 0, "q.cx"});
//         func.ops.push_back({QOpKind::Custom, "", b_inv + "[" + std::to_string(i) + "]", "", 0, "q.x"});
//     }
//     // Create constant 1.
//     std::string plus_one = new_tmp("one");
//     emit_qubit_alloc(func, plus_one, num_bits);
//     emit_qubit_init(func, plus_one, 1, num_bits);
//     // Compute two's complement of b: ~b + 1.
//     std::string b_twos = new_tmp("b2");
//     emit_qubit_alloc(func, b_twos, num_bits);
//     emit_quantum_adder(func, b_twos, b_inv, plus_one, num_bits);
//     // Now, subtract: a + (~b + 1)
//     emit_quantum_adder(func, result, a, b_twos, num_bits);
// }

// Memory-efficient quantum subtractor
// void emit_quantum_subtractor(QMLIR_Function& func, const std::string& result,
//     const std::string& a, const std::string& b, int num_bits) {
// // Create a register for ~b (instead of copying b and then inverting)
// std::string b_inv = new_tmp("inv");
// emit_qubit_alloc(func, b_inv, num_bits);

// // Initialize b_inv as the inverse of b
// for (int i = 0; i < num_bits; ++i) {
// // Copy b to b_inv
// func.ops.push_back({QOpKind::Custom, "", b + "[" + std::to_string(i) + "]",
//    b_inv + "[" + std::to_string(i) + "]", 0, "q.cx"});
// // Invert b_inv
// func.ops.push_back({QOpKind::Custom, "", b_inv + "[" + std::to_string(i) + "]", "", 0, "q.x"});
// }

// // Create constant 1 with a single-qubit register
// std::string one = new_tmp("one");
// emit_qubit_alloc(func, one, 1);
// // Set one[0] to 1
// func.ops.push_back({QOpKind::Custom, "", one + "[0]", "", 0, "q.x"});

// // Compute a + ~b + 1 directly (instead of creating an intermediate register)
// // We do this by extending our adder to handle a non-power-of-2 sized input

// // First, add a[0] and b_inv[0]
// func.ops.push_back({QOpKind::Custom, "", a + "[0]", result + "[0]", 0, "q.cx"});
// func.ops.push_back({QOpKind::Custom, "", b_inv + "[0]", result + "[0]", 0, "q.cx"});

// // Add 1 to result[0] (part of the two's complement)
// func.ops.push_back({QOpKind::Custom, "", one + "[0]", result + "[0]", 0, "q.cx"});

// // Create a single-bit carry register
// std::string carry = new_tmp("c");
// emit_qubit_alloc(func, carry, 1);

// // Calculate initial carry from first bit
// // Carry is 1 if at least two of a[0], b_inv[0], one[0] are 1
// func.ops.push_back({QOpKind::Custom, carry + "[0]", a + "[0]", b_inv + "[0]", 0, "q.ccx"});
// std::string temp = new_tmp("t");
// emit_qubit_alloc(func, temp, 1);
// func.ops.push_back({QOpKind::Custom, temp + "[0]", a + "[0]", one + "[0]", 0, "q.ccx"});
// func.ops.push_back({QOpKind::Custom, "", temp + "[0]", carry + "[0]", 0, "q.cx"});
// func.ops.push_back({QOpKind::Custom, temp + "[0]", b_inv + "[0]", one + "[0]", 0, "q.ccx"});
// func.ops.push_back({QOpKind::Custom, "", temp + "[0]", carry + "[0]", 0, "q.cx"});

// // Process the remaining bits
// for (int i = 1; i < num_bits; i++) {
// // Add a[i] and b_inv[i] to result[i]
// func.ops.push_back({QOpKind::Custom, "", a + "[" + std::to_string(i) + "]", 
//    result + "[" + std::to_string(i) + "]", 0, "q.cx"});
// func.ops.push_back({QOpKind::Custom, "", b_inv + "[" + std::to_string(i) + "]", 
//    result + "[" + std::to_string(i) + "]", 0, "q.cx"});

// // Add carry to result[i]
// func.ops.push_back({QOpKind::Custom, "", carry + "[0]", 
//    result + "[" + std::to_string(i) + "]", 0, "q.cx"});

// // Calculate new carry (if i < num_bits - 1)
// if (i < num_bits - 1) {
// // Clean up previous temp calculations
// func.ops.push_back({QOpKind::Custom, temp + "[0]", b_inv + "[" + std::to_string(i-1) + "]", 
//        one + "[0]", 0, "q.ccx"});
// func.ops.push_back({QOpKind::Custom, temp + "[0]", a + "[" + std::to_string(i-1) + "]", 
//        one + "[0]", 0, "q.ccx"});

// // Calculate new carry = majority(a[i], b_inv[i], carry[0])
// func.ops.push_back({QOpKind::Custom, temp + "[0]", a + "[" + std::to_string(i) + "]", 
//        b_inv + "[" + std::to_string(i) + "]", 0, "q.ccx"});
// func.ops.push_back({QOpKind::Custom, "", temp + "[0]", carry + "[0]", 0, "q.cx"});
// func.ops.push_back({QOpKind::Custom, temp + "[0]", a + "[" + std::to_string(i) + "]", 
//        carry + "[0]", 0, "q.ccx"});
// func.ops.push_back({QOpKind::Custom, "", temp + "[0]", carry + "[0]", 0, "q.cx"});
// func.ops.push_back({QOpKind::Custom, temp + "[0]", b_inv + "[" + std::to_string(i) + "]", 
//        carry + "[0]", 0, "q.ccx"});
// func.ops.push_back({QOpKind::Custom, "", temp + "[0]", carry + "[0]", 0, "q.cx"});
// }
// }
// }

void emit_quantum_subtractor(QMLIR_Function& func, const std::string& result,
    const std::string& a, const std::string& b, int num_bits) {
// Create a register for ~b (instead of copying b and then inverting)
std::string b_inv = new_tmp("inv");
emit_qubit_alloc(func, b_inv, num_bits);

// Initialize b_inv as the inverse of b
for (int i = 0; i < num_bits; ++i) {
func.ops.push_back({QOpKind::Custom, "", b + "[" + std::to_string(i) + "]",
   b_inv + "[" + std::to_string(i) + "]", 0, "q.cx"});
func.ops.push_back({QOpKind::Custom, "", b_inv + "[" + std::to_string(i) + "]", "", 0, "q.x"});
}

// Create a standard full adder implementation to compute a + (~b + 1)
std::string carry = new_tmp("c");
emit_qubit_alloc(func, carry, num_bits + 1);

// Set the initial carry to 1 (for two's complement)
func.ops.push_back({QOpKind::Custom, "", carry + "[0]", "", 0, "q.x"});

// CRITICAL FIX: Reverse the bit order in the result register to match expected output
for (int i = 0; i < num_bits; i++) {
// Map bit i to position (num_bits-1-i) to reverse the bit order
int result_pos = num_bits - 1 - i;

// First compute the sum without considering the carry
func.ops.push_back({QOpKind::Custom, "", a + "[" + std::to_string(i) + "]", 
   result + "[" + std::to_string(result_pos) + "]", 0, "q.cx"});
func.ops.push_back({QOpKind::Custom, "", b_inv + "[" + std::to_string(i) + "]", 
   result + "[" + std::to_string(result_pos) + "]", 0, "q.cx"});

// Now incorporate the carry from the previous bit
func.ops.push_back({QOpKind::Custom, "", carry + "[" + std::to_string(i) + "]", 
   result + "[" + std::to_string(result_pos) + "]", 0, "q.cx"});

// Calculate carry[i+1]
std::string temp = new_tmp("t" + std::to_string(i));
emit_qubit_alloc(func, temp, 1);

func.ops.push_back({QOpKind::Custom, carry + "[" + std::to_string(i+1) + "]", 
   a + "[" + std::to_string(i) + "]", 
   b_inv + "[" + std::to_string(i) + "]", 0, "q.ccx"});

func.ops.push_back({QOpKind::Custom, temp + "[0]", 
   result + "[" + std::to_string(result_pos) + "]", 
   carry + "[" + std::to_string(i) + "]", 0, "q.ccx"});

func.ops.push_back({QOpKind::Custom, "", temp + "[0]", 
   carry + "[" + std::to_string(i+1) + "]", 0, "q.cx"});
}
}