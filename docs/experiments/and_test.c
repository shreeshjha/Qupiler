#include <stdio.h>

void quantum_circuit() {
  int a = 6;
  int b = 3;
  int res = a && b;
  printf("%d\n", res);
}

int main() {
  quantum_circuit();
  return 0;
}
