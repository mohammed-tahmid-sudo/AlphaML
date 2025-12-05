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

  // THIS TRANING IS COMPLEATLY FLAWED. TODO: FIX IT LATER
  void train(Layer<T> &Model, Tensor<Tensor<Tensor<Tensor<T>>>> &train_x,
             Tensor<Tensor<Tensor<Tensor<T>>>> &test_x, Tensor<T> &train_y,
             Tensor<T> &test_y, size_t epochs, int batch_size = 32,
             float Learning_rate = 0.1, std::string loss = "crossentropy",
             const std::string optimizer = "adam",
             const std::string device = "CPU") {

    std::cout << "Creating batch\n";
    auto batches_x = CreateBatch(train_x, batch_size);
    auto batches_y = CreateBatch(train_y, batch_size);
    std::cout << "Batching Compleate\n";

    for (int epoch = 0; epoch < epochs; epoch++) {

      for (int batch = 0; batch < batches_y.size(); batch++) {
        Tensor<Tensor<Tensor<T>>> batch_x = batches_x[batch];
        auto batch_y = batches_y[batch];
        for (int i = 0; i < batch_y.size(); i++) {
          auto x_true = batch_x[i];
          auto y_true = batches_y[i];

          Tensor<T> y_preds = Model.Forward(x_true);
          T loss = CrossEntropyLoss(y_preds, y_true);

          Tensor<T> grad(y_preds.size());
          for (int j = 0; j < y_preds.size(); j++) {
            grad[j] = y_preds[j] - y_true[j];
            Tensor<T> param_grads = compute_gradients(x_true, grad, params);
						// ??????????????????????????????????????????????????????????????????????????
						// This thing is SHIT ????????????????????????????
						// I'm GONNA REWRITE ALL THIS SHIT IN A NICER WAY 
						// SOO MUCH CODING FOR TODAY. NOW I'M GONNA HIT PYSIDE

          }
        }
      }
    }
  }
};
