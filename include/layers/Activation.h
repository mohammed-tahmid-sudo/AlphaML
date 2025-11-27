// Activation.h / Activation.cpp
// Activation functions: ReLU, Sigmoid, Tanh.

#include "layers/Layer.h"
#include "utils/Tensor.h"

#include <cmath>

// ReLU for multi-dimensional Tensor
template <typename T> class ReLU : public Layer<T> {
public:
  // ReLU(Tensor<T> data) : x(data) {}

  Tensor<T> Forward(const Tensor<T> &x) {
    Tensor<T> y;
    for (int i = 0; i < x.size(); i++) {
      y.Push_back(std::max(T(0), x[i]));
    }
    return y;
  }
};

template <typename T> class Sigmoid : public Layer<T> {
public:
  // Sigmoid(Tensor<T> data) : tensor(data) {}

  Tensor<T> Forward(const Tensor<T> &x) {
    Tensor<T> y;
    for (int i = 0; i < x.size(); i++) {
      y.Push_back(T(1) / (T(1) + std::exp(-x[i])));
    }
    return y;
  }
};
