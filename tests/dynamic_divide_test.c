#include <stdio.h>

int quantum_circuit() {
  int dividend = 2;
  int divisor = 2;
  int result = dividend / divisor; // Different numbers: 15 ÷ 3 = 5
  return result;
}

int main() {
  int result = quantum_circuit();
  printf("2 ÷ 2 = %d\n", result);
  return 0;
}
