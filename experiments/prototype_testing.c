#include <stdio.h>

void quantum_circuit(){
  int a=7;
  int b=2;
  int c=3;
  int sum = a + b;
  int ans = sum - c;
  printf("%d\n",ans);
}

int main(){
  quantum_circuit();
  return 0;
}
