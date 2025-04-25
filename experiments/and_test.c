#include <stdio.h>

void quantum_circuit(){
  int a=5;
  int b=4;
  int res=a && b;
  printf("%d\n",res);
}

int main(){
  quantum_circuit();
  return 0;
}
