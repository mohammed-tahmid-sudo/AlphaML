// Dense.h / Dense.cpp
// Fully connected layer. Implements forward and backward pass.
#pragma once
#include <layers/Layer.h>
#include <utils/Tensor.h>
#include <vector>

// Derived class
template <typename T> class Dense2D : public Layer<T> {

public:
  int in_features, out_features;
  Tensor<T> weights;
  Tensor<T> bias;

  Dense2D(int in_feat, int out_feat)
      : in_features(in_feat), out_features(out_feat) {}
};
