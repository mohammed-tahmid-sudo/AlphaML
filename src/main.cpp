#include "layers/Activation.h"
// #include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"
#include <losses/CrossEntropyLoss.h>
// #include "utils/sequential.h"

int main() {
  Tensor<double> Ylogits = {0.2, 0.7, 0.1}; // predicted probabilities
  Tensor<double> Ytrue = {0.0, 1.0, 0.0};   // true label (one-hot)

  double loss = CrossEntropyLoss(Ylogits, Ytrue);
  std::cout << loss << std::endl;
}
