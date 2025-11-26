// Layer.h
// Abstract base class for all neural network layers.
// Requires forward() and backward() methods.
#include <utils/Tensor.h>
#pragma once

template <typename T> struct Layer {

  // TODO: Add Backwards
  // virtual Tensor<T> Forward(T X) = 0;
  virtual Tensor<T> Forward() = 0;
  virtual ~Layer() = default;
};
