// MSELoss.h / MSELoss.cpp
#pragma once
#include <stdexcept>   // for std::runtime_error
#include <cstddef>     // for size_t
#include "utils/Tensor.h"  // your Tensor class

// Mean Squared Error Loss
template <typename T>
T MSELoss(const Tensor<T>& y_pred, const Tensor<T>& y_true) {
    if (y_pred.size() != y_true.size())
        throw std::runtime_error("Size mismatch in MSELoss");

    T sum = 0;
    for (size_t i = 0; i < y_pred.size(); ++i) {
        T diff = y_pred[i] - y_true[i];
        sum += diff * diff;
    }
    return sum / y_pred.size();
}

// // Gradient of MSE
// template <typename T>
// Tensor<T> MSEGrad(const Tensor<T>& y_pred, const Tensor<T>& y_true) {
//     if (y_pred.size() != y_true.size())
//         throw std::runtime_error("Size mismatch in MSEGrad");

//     Tensor<T> grad(y_pred.size());
//     T inv = 1.0 / y_pred.size();
//     for (size_t i = 0; i < y_pred.size(); ++i)
//         grad[i] = 2 * (y_pred[i] - y_true[i]) * inv;

//     return grad;
// }

