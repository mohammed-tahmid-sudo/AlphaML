#include "layers/Activation.h"
#include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"

int main() {

  Tensor<Tensor<int>> A = {{1, 2, 3}, {4, 5, 6}};
  // 3x2 matrix B
  Tensor<Tensor<int>> B = {{7, 8}, {9, 10}, {11, 12}};

  Tensor<Tensor<int>> C = Matmul(A, B);

  C.print();

  return 0;
}
