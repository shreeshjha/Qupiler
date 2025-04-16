#include <stdio.h>

void quantum_circuit() {
    int x = 1;
    int y = 3;
    while (x < y) {
        x = x + 1;
    }
    int sum = x;
    printf("%d\n",sum);
}

int main() {
    quantum_circuit();
    return 0;
}
