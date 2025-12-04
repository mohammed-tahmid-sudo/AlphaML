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
  Tensor<T> Backward(const Tensor<T> &x, const Tensor<T> &grad_out) {
    Tensor<T> grad_in = grad_out;
    for (int i = 0; i < x.size(); i++) {
      grad_in[i] *= (x[i] > 0 ? 1 : 0);
    }
    return grad_in;
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
  Tensor<T> Backward(const Tensor<T> &y, const Tensor<T> &grad_out) {
    Tensor<T> grad_in = grad_out;
    for (int i = 0; i < y.size(); i++) {
      grad_in[i] *= y[i] * (1 - y[i]);
    }
    return grad_in;
  }
};
