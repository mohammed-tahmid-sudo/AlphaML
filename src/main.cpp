#include "layers/Activation.h"
#include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"

int main() {
  Tensor<int> t1 = {1, 2};
  Tensor<Tensor<int>> t2 = {t1, t1};
  Tensor<Tensor<Tensor<int>>> t3 = {t2, t2};
  t1.print();
  t2.print();
  t3.print();
	return 0;
}
