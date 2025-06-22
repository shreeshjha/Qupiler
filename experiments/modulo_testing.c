#include <stdio.h>

void quantum_circuit() {
  int a = 7;
  int b = 3;

  int ans = a % b;
  printf("%d\n", ans);
}

int main() {
  quantum_circuit();
  return 0;
}
