#include "layers/Activation.h"
#include <optimizers/Adam.h>
// #include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"
#include <losses/CrossEntropyLoss.h>
// #include "utils/sequential.h"
#include <layers/flatten.h>

int main() {
  // x = [[[0,1],[2,3]], [[4,5],[6,7]]]
  Tensor<Tensor<Tensor<int>>> x;
  x = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  flatten3d<int> flatten;
  Tensor<int> y = flatten.Forward(x);
  y.print();
}
