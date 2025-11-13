#include "layers/Activation.h"
#include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"

int main() {
  Tensor<int> t1 = {1,2};
  t1.Push_back(3);
  t1.print(); // [3]

  Tensor<Tensor<int>> t2;
  t2.Push_back(Tensor<int>({1, 2, 3}));
  t2.print(); // [[1, 2, 3]]

  Tensor<Tensor<Tensor<int>>> t3;
  t3.Push_back(Tensor<Tensor<int>>({Tensor<int>({4, 5})}));
  t3.print(); // [[[4, 5]]]}
}
