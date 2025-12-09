// Dense.h / Dense.cpp
// Fully connected layer. Implements forward and backward pass.
#pragma once
#include <layers/Layer.h>
#include <tuple>
#include <utils/Tensor.h>

template <typename T, typename input = const Tensor<T> &,
          typename input_back = const Tensor<T> &,
          typename backward =
              std::tuple<Tensor<T>, Tensor<Tensor<T>>, Tensor<T>>>
class Dense1D : public Layer<T> {

public:
  int in_features, out_features;
  Dense1D(int in_feat, int out_feat)
      : in_features(in_feat), out_features(out_feat),
        weights(out_feat, Tensor<T>(in_feat, 0.0)), // zeros for now
        bias(out_feat, 0.0) {}                      // zeros for now

  Tensor<Tensor<T>> weights; // [out_features][in_features]
  Tensor<T> bias;            // [out_features]
  Tensor<T> last_x;

  Tensor<T> Forward(input x) override {

    last_x = x;
    Tensor<T> y(out_features, 0.0);

    for (int i = 0; i < out_features; i++) {
      for (int j = 0; j < in_features; j++) {
        y[i] += weights[i][j] * x[j]; // FIXED
      }
      y[i] += bias[i];
    }
    y.print();

    return y;
  }
  backward Backward(input_back dy) {

    Tensor<T> dx(in_features, 0.0);
    Tensor<Tensor<T>> dW(out_features, Tensor<T>(in_features, 0.0));
    Tensor<T> db(out_features, 0.0);

    // db = dy
    for (int i = 0; i < out_features; i++) {
      db[i] = dy[i];
    }

    // dW[i][j] = dy[i] * x[j]
    for (int i = 0; i < out_features; i++) {
      for (int j = 0; j < in_features; j++) {
        dW[i][j] = dy[i] * last_x[j];
      }
    }

    // dx[j] = sum_i dy[i] * W[i][j]
    for (int j = 0; j < in_features; j++) {
      T sum = 0;
      for (int i = 0; i < out_features; i++) {
        sum += dy[i] * weights[i][j];
      }
      dx[j] = sum;
    }

    return {dx, dW, db};
  }
};
