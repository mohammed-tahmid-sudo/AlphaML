// CrossEntropyLoss.h / CrossEntropyLoss.cpp
// Cross-Entropy loss for classification.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <layers/Layer.h>
#include <stdexcept>
#include <utils/Tensor.h>
#include <vector>

// template <typename T>
// T CrossEntropyLoss(const Tensor<T> Ylogits, const Tensor<T> Ytrue) {
//   T output = 0;
//
//   if (Ylogits.size() != Ytrue.size()) {
//     throw std::runtime_error(
//         "LIBRARY ERROR: YTrue and Ylogits Must have to have the same value ");
//   }
//
//   for (int i = 0; i < Ylogits.size(); i++) {
//     // main calculation
//     output += Ytrue[i] * std::log(Ylogits[i]);
//   }
//   return -output;
// }

template <typename T>
T CrossEntropyLoss(const Tensor<T>& logits, const Tensor<T>& y_true) {
    T maxv = logits[0];
    for (size_t i = 1; i < logits.size(); ++i)
        if (logits[i] > maxv) maxv = logits[i];

    T sum = 0;
    for (size_t i = 0; i < logits.size(); ++i)
        sum += std::exp(logits[i] - maxv);

    T loss = 0;
    for (size_t i = 0; i < logits.size(); ++i)
        loss += y_true[i] * (std::log(sum) + maxv - logits[i]);

    return loss;
}


template <typename T> T LogSumExp(const std::vector<T> &logits) {
  T max_val = *std::max_element(logits.begin(), logits.end());
  T sum = 0;
  for (auto l : logits)
    sum += std::exp(l - max_val);
  return max_val + std::log(sum);
}

Tensor<double> CrossEntropyGradTensor(const Tensor<double>& logits, int target_class) {
    size_t C = logits.size();
    Tensor<double> grad(C);
    double log_sum = *std::max_element(logits.data.begin(), logits.data.end());
    double sum_exp = 0;
    for (size_t i = 0; i < C; ++i) sum_exp += std::exp(logits[i] - log_sum);
    for (size_t i = 0; i < C; ++i) {
        double prob = std::exp(logits[i] - log_sum) / sum_exp;
        grad[i] = prob - (i == target_class ? 1.0 : 0.0);
    }
    return grad;
}

