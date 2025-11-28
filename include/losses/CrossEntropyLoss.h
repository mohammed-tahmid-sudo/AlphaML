// CrossEntropyLoss.h / CrossEntropyLoss.cpp
// Cross-Entropy loss for classification.

#include <cmath>
#include <layers/Layer.h>
#include <stdexcept>
#include <utils/Tensor.h>

template <typename T>
T CrossEntropyLoss(const Tensor<T> Ylogits, const Tensor<T> Ytrue) {
  T output;

  if (Ylogits.size() != Ytrue.size()) {
    throw std::runtime_error(
        "LIBRARY ERROR: YTrue and Ylogits Must have to have the same value ");
  }

  for (int i = 0; i < Ylogits.size(); i++) {
    // main calculation
    output += Ytrue[i] * std::log(Ylogits[i]);
  }
  return -output;
}
