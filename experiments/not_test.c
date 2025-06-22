#include <stdio.h>

void quantum_circuit() {
  int a = 8;
  int b = ~a;
  printf("%d\n", b);
}

int main() {
  quantum_circuit();
  return 0;
}
