#include <stdio.h>

int quantum_circuit() {
    int dividend = 12;
    int divisor = 4;
    int result = dividend / divisor;  // This will trigger _decompose_div_circuit with T-gates
    return result;                    // Expected: 3 (12 ÷ 4 = 3)
}

int main() {
    int result = quantum_circuit();
    printf("12 ÷ 4 = %d\n", result);
    return 0;
}