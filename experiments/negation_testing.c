#include <stdio.h>

void quantum_circuit() {
    int a = 1;
    int neg_a = -a;  // Computed using unary minus
    int b = 3;      // Positive value
    int neg_b = -b; // Computed using unary minus
    int sum = a + neg_b; // This should be 5 + (-3) = 2   
    printf("%d\n", sum);
}

int main() {
    quantum_circuit();
    return 0;
}
