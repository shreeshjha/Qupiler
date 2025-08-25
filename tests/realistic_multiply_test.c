#include <stdio.h>

int quantum_circuit() {
    int a = 3;
    int b = 3; 
    int result = a * b;  // This will trigger _decompose_mul_circuit with T-gates
    return result;       // Expected: 9 (3 × 3 = 9)
}

int main() {
    int result = quantum_circuit();
    printf("3 × 3 = %d\n", result);
    return 0;
}