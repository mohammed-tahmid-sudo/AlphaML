// Layer.h
// Abstract base class for all neural network layers.
// Requires forward() and backward() methods.
#include <utils/Tensor.h>
#pragma once

template <typename T, typename input_forward = Tensor<T>,
          typename input_backward = Tensor<T>,
          typename backward_type = Tensor<T>>
struct Layer {

  virtual Tensor<T> Forward(const input_forward &x) = 0;
  virtual backward_type Backward(const input_backward &x) = 0;

  virtual void UpdateParameters(T lr) {} 
  virtual ~Layer() = default;
};
