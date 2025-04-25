#include <stdio.h>

void quantum_circuit(){
  int a=4;
  int b=5;
  int res= a || b;
  printf("%d\n",res);
}

int main(){
  quantum_circuit();
  return 0;
}
