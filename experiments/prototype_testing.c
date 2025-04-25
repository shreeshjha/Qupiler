#include <stdio.h>

void quantum_circuit(){
  int a=1;
  int b=2;
  int c=3;
  int temp_c = c++;
  int temp_a = a--;
  int sum = temp_a + b;
  int ans = sum - temp_c;
  printf("%d\n",ans);
}

int main(){
  quantum_circuit();
  return 0;
}
