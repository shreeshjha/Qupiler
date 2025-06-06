#include <stdio.h>

void quantum_circuit() {
    int x = 5;
    int y = 1;
    while (x > y) {
        x = x - 1;
    }
    int sub = x;
    printf("%d\n",sub);
}

int main() {
    quantum_circuit();
    return 0;
}
