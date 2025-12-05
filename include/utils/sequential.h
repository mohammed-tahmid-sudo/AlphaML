// Sequential Function

// Pragma once
#pragma once

#include "layers/Layer.h"
#include "utils/Tensor.h"
#include <initializer_list>
#include <vector>

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

  std::vector<std::vector<std::vector<std::vector<T>>>>
  create_batches(const std::vector<std::vector<std::vector<T>>> &data,
                 size_t batch_size) {
    std::vector<std::vector<std::vector<std::vector<T>>>> batches;
    size_t n = data.size();

    for (size_t i = 0; i < n; i += batch_size) {
      size_t end = std::min(i + batch_size, n);
      std::vector<std::vector<std::vector<T>>> batch(data.begin() + i,
                                                     data.begin() + end);
      batches.push_back(batch);
    }

    void train(Layer<T> & Model, Tensor<Tensor<Tensor<Tensor<T>>>> & train_x,
               Tensor<Tensor<Tensor<Tensor<T>>>> & test_x, Tensor<T> & train_y,
               Tensor<T> & test_y, size_t epoch, int batch_size = 32,
               float Learning_rate = 0.1, std::string loss = "crossentropy",
               std::string optimizer = "adam",
               const std::string device = "CPU") {

      auto batches_x = create_batches(train_x, batch_size);
      auto batches_y = create_batches(train_y, batch_size);

      for (size_t i = 0; i < epoch, i++) {
      }
    }
  }
};
