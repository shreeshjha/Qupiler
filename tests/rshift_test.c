#include <stdio.h>

int quantum_circuit() {
  int a = 9;               // Binary: 0110
  int shift = 1;           // Shift right by 1 position
  int result = a >> shift; // Expected: 1 (0011)
  return result;
}

int main() {
  int result = quantum_circuit();
  printf("9 >> 1 = %d\n", result);
  return 0;
}
