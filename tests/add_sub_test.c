#include <stdio.h>

void quantum_circuit() {
  int a = 2;
  int b = 7;
  int sum = a + b;
  int c = 3;
  int minus = sum - c;
  printf("%d\n", minus);
}
int main() {
  quantum_circuit();
  return 0;
}
