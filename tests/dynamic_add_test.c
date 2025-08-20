#include <stdio.h>

int quantum_circuit() {
    int a = 8;
    int b = 6;
    int result = a + b;  // Different numbers: 8 + 6 = 14
    return result;       
}

int main() {
    int result = quantum_circuit();
    printf("8 + 6 = %d\n", result);
    return 0;
}