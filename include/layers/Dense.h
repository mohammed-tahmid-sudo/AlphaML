// Dense.h / Dense.cpp
// Fully connected layer. Implements forward and backward pass.
#pragma once
#include <layers/Layer.h>
#include <utils/Tensor.h>

// Derived class
template <typename T> class Dense1D : public Layer<T> {

public:
  int in_features, out_features;
  Dense1D(int in_feat, int out_feat)
      : in_features(in_feat), out_features(out_feat) {}

  Tensor<Tensor<T>> weights; // [out_features][in_features]
  Tensor<T> bias;            // [out_features]
  Tensor<T> input;           // [in_features]

  Tensor<T> Forward() override {
    Tensor<T> y(out_features, 0.0);

    for (int i = 0; i < out_features; i++) {
      for (int j = 0; j < in_features; j++) {
        y[i] += weights[i][j] * input[j]; // FIXED
      }
      y[i] += bias[i];
    }

    return y;
  }
};
