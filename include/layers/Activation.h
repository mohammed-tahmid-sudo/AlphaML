// Activation.h / Activation.cpp
// Activation functions: ReLU, Sigmoid, Tanh.

#include "utils/Tensor.h"

#include <cmath>
#include <type_traits>

// ReLU for multi-dimensional Tensor
template <typename T> void ReLU(Tensor<T> &tensor) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    if constexpr (std::is_class_v<T>) {
      // Recurse if element is another Tensor
      ReLU(tensor[i]);
    } else {
      // Base case: apply ReLU
      tensor[i] = (tensor[i] > 0) ? tensor[i] : 0;
    }
  }
}

template <typename T> void Sigmoid(Tensor<T> &tensor) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    if constexpr (std::is_class_v<T>) {
      // Recurse if element is another Tensor
      Sigmoid(tensor[i]);
    } else {
      // Base case: apply ReLU
      tensor[i] = 1.0 / (1.0 + std::exp(-i));
    }
  }
}


