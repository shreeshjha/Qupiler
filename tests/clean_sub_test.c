#include <stdio.h>

int quantum_circuit() {
    int a = 9;
    int b = 3;
    int result = a - b;  // This should be 9 - 3 = 6
    return result;
}

int main() {
    int result = quantum_circuit();
    printf("9 - 3 = %d\n", result);
    return 0;
}