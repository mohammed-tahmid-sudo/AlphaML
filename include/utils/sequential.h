// Sequential Function

#pragma once

#include "layers/Layer.h"
#include "utils/Tensor.h"
#include <algorithm>
#include <initializer_list>
#include <losses/CrossEntropyLoss.h>
#include <vector>

template <typename T>
void compute_gradients_linear(
    const Tensor<T> &x, // input vector
    const Tensor<T>
        &grad_output,  // gradient of loss w.r.t output (y_pred - y_true)
    Tensor<T> &grad_W, // gradient w.r.t weights
    Tensor<T> &grad_b, // gradient w.r.t biases
    size_t input_size, size_t output_size) {
  // grad_W is output_size x input_size
  for (size_t i = 0; i < output_size; ++i) {
    for (size_t j = 0; j < input_size; ++j) {
      grad_W[i * input_size + j] = grad_output[i] * x[j];
    }
    grad_b[i] = grad_output[i];
  }
}

template <typename T> class Sequential : public Layer<T> {
public:
  Sequential() = default;

  std::vector<Layer<T> *> layers;

  Sequential(std::initializer_list<Layer<T> *> list) {
    for (auto l : list)
      layers.push_back(l);
  }
  Tensor<T> Forward(const Tensor<T> &x) override {
    Tensor<T> output = x;
    for (auto *l : layers) {
      output = l->Forward(output);
    }
    return output;
  }

  Tensor<T> Backward(const Tensor<T> &grad) override {
    Tensor<T> g = grad;
    for (int i = layers.size() - 1; i >= 0; --i)
      g = layers[i]->Backward(g);
    return g;
  }

  Tensor<Tensor<Tensor<Tensor<T>>>>
  CreateBatch(const Tensor<Tensor<Tensor<T>>> &data, int batchsize) {
    Tensor<Tensor<Tensor<Tensor<T>>>> y;

    for (int i = 0; i < data.size(); i += batchsize) {
      Tensor<Tensor<Tensor<int>>> batch;
      for (int j = i; j < std::min(i + batchsize, (int)data.size()); j++)
        batch.Push_back(data[j]);

      y.Push_back(batch);
    }
  }
};
