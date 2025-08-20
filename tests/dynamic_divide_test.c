#include <stdio.h>

int quantum_circuit() {
  int dividend = 6;
  int divisor = 3;
  int result = dividend / divisor; // Different numbers: 15 ÷ 3 = 5
  return result;
}

int main() {
  int result = quantum_circuit();
  printf("6 ÷ 3 = %d\n", result);
  return 0;
}
