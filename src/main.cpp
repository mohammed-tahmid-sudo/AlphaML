#include "layers/Activation.h"
#include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"

int main() {
  int in_features = 3;
  int out_features = 1;

  Dense1D<double> layer(in_features, out_features);

  // Initialize weights and bias
  layer.weights = Tensor<Tensor<double>>(
      out_features, Tensor<double>(in_features, 1.0) // all 1s
  );

  layer.bias = Tensor<double>(out_features, 0.5); // all 0.5

  // Set input
  layer.input = Tensor<double>(in_features, 2.0); // all 2s

  // Forward pass
  Tensor<double> output = layer.Forward();

  output.print();
  return 0;
}
