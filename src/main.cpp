#include "layers/Activation.h"
#include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"

int main() {
  Tensor<int> t1 = {1, 2, 3};
  Tensor<Tensor<int>> t2 = {t1, t1};
  Tensor<Tensor<Tensor<int>>> t3 = {t2, t2};
  Tensor<Tensor<Tensor<int>>> t4 = {t2, t2};

  t1.print(); // [1, 2, 3]
  std::cout << "\n";
  t2.print(); // [[1, 2, 3], [1, 2, 3]]
  std::cout << "\n";
  t3.print(); // [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]
  Tensor<Tensor<int>> output = Matmul(t3, t4);
  output.print();
}
