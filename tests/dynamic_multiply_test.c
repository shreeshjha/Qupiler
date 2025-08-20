#include <stdio.h>

int quantum_circuit() {
  int a = 5;
  int b = 3;
  int result = a * b; // Different numbers: 5 × 2 = 10
  return result;
}

int main() {
  int result = quantum_circuit();
  printf("5 × 3 = %d\n", result);
  return 0;
}
