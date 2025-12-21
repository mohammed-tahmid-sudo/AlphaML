// Activation.h / Activation.cpp
// Activation functions: ReLU, Sigmoid, Tanh.

#include "layers/Layer.h"
#include "utils/Tensor.h"

#include <cmath>

template <typename T>
class ReLU : public Layer<T> {
    Tensor<T> x_cache;

public:
    Tensor<T> Forward(const Tensor<T>& x) override {
        x_cache = x;
        Tensor<T> y(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            y[i] = std::max(T(0), x[i]);
        return y;
    }

    Tensor<T> Backward(const Tensor<T>& grad_out) override {
        Tensor<T> grad_in(grad_out.size());
        for (size_t i = 0; i < grad_out.size(); ++i)
            grad_in[i] = grad_out[i] * (x_cache[i] > 0 ? 1 : 0);
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
