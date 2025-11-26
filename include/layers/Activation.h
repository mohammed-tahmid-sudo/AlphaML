// Activation.h / Activation.cpp
// Activation functions: ReLU, Sigmoid, Tanh.

#include "layers/Layer.h"
#include "utils/Tensor.h"

#include <cmath>

// ReLU for multi-dimensional Tensor
template <typename T> class ReLU : public Layer<T> {
public:
  Tensor<T> tensor;
  ReLU(Tensor<T> data) : tensor(data) {}

  Tensor<T> Forward() {
    Tensor<T> y;
    for (int i = 0; i < tensor.size(); i++) {
      y.Push_back(std::max(T(0), tensor[i]));
    }
    return y;
  }
};

template <typename T> class Sigmoid : public Layer<T> {
public:
  Tensor<T> tensor;
  Sigmoid(Tensor<T> data) : tensor(data) {}

  Tensor<T> Forward() {
    Tensor<T> y;
    for (int i = 0; i < tensor.size(); i++) {
      y.Push_back(T(1) / (T(1) + std::exp(-tensor[i])));
    }
    return y;
  }
};
