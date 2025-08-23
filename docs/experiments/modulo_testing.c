#include <stdio.h>

void quantum_circuit() {
  int a = 9;
  int b = 5;

  int ans = a % b;
  printf("%d\n", ans);
}

int main() {
  quantum_circuit();
  return 0;
}
