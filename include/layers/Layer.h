// Layer.h
// Abstract base class for all neural network layers.
// Requires forward() and backward() methods.
#include <utils/Tensor.h>
#pragma once

template <typename T> struct Layer {

  virtual Tensor<T> Forward(const Tensor<T>& x) = 0;
  virtual Tensor<T> Backward(const Tensor<T>& x) = 0;
  virtual ~Layer() = default;
};
