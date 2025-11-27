// Sequential Function

#include "layers/Layer.h"
#include "utils/Tensor.h"
#include <initializer_list>

#pragma once

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
};
