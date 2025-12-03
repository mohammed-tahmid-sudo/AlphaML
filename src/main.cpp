#include "layers/Activation.h"
#include <optimizers/Adam.h>
// #include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"
#include <losses/CrossEntropyLoss.h>
// #include "utils/sequential.h"

int main() {
  Tensor<double> params = {0.5, -0.3, 0.8};
  Tensor<double> grads = {0.1, -0.2, 0.3};

  AdamOptimizer optimizer(params.size(), 0.01);

  for (int step = 0; step < 5; step++) {
    optimizer.update(params, grads);

    std::cout << "Step " << step + 1 << ": ";
    // for (double p : params) std::cout << p << " ";
    params.print();
    std::cout << std::endl;
  }
  return 0;
}
