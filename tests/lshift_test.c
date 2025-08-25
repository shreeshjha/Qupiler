#include <stdio.h>

int quantum_circuit() {
  int a = 1;               // Binary: 0011
  int shift = 3;           // Shift left by 2 positions
  int result = a << shift; // Expected: 12 (1100)
  return result;
}

int main() {
  int result = quantum_circuit();
  printf("1 << 3 = %d\n", result);
  return 0;
}
