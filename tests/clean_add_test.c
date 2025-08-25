#include <stdio.h>

int quantum_circuit() {
    int a = 5;
    int b = 7;
    int result = a + b;  // This will use clean addition circuit (no T-gates)
    return result;       // Expected: 12 (5 + 7 = 12)
}

int main() {
    int result = quantum_circuit();
    printf("5 + 7 = %d\n", result);
    return 0;
}