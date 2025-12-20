// Dense.h / Dense.cpp
// Fully connected layer. Implements forward and backward pass.
#pragma once
#include <layers/Layer.h>
#include <tuple>
#include <utils/Tensor.h>

// template <typename T, typename input = const Tensor<T> &,
//           typename input_back = const Tensor<T> &,
//           typename backward =
//               std::tuple<Tensor<T>, Tensor<Tensor<T>>, Tensor<T>>>
// class Dense : public Layer<T> {
//
// public:
//   int in_features, out_features;
//   Tensor<T> biases;
//   Tensor<Tensor<T>> weights;
//   Tensor<T> last_x; // store input from forward pass
//   Tensor<T> dW, db; // gradients
//
//   Dense(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
//     // resize biases and gradient
//     biases.resize(out_f, 0.0);
//     db.resize(out_f, 0.0);
//
//     // resize weights and weight gradients
//     weights.resize(out_f);
//     dW.resize(out_f);
//     for (size_t i = 0; i < out_f; ++i) {
//       weights[i].resize(in_f, 0.0);
//       dW[i].resize(in_f, 0.0);
//     }
//   }
//
//   Tensor<T> Forward(input x) override {
//
//     last_x = x;
//     Tensor<T> y(out_features, 0.0);
//
//     for (int i = 0; i < out_features; i++) {
//       for (int j = 0; j < in_features; j++) {
//         y[i] += weights[i][j] * x[j]; // FIXED
//       }
//       y[i] += biases[i];
//     }
//     y.print();
//
//     return y;
//   }
//
//   Tensor<T> Backward(const Tensor<T> &dy) {
//     // dx has the same size as input
//     Tensor<T> dx(last_x.size());
//
//     // dW: out_features x in_features
//     dW.resize(out_features);
//     for (size_t i = 0; i < out_features; ++i) {
//       dW[i].resize(in_features);
//       for (size_t j = 0; j < in_features; ++j) {
//         dW[i][j] = dy[i] * last_x[j]; // gradient w.r.t weights
//       }
//     }
//
//     // db: same size as out_features
//     db.resize(out_features);
//     for (size_t i = 0; i < out_features; ++i) {
//       db[i] = dy[i]; // gradient w.r.t bias
//     }
//
//     // dx[j] = sum_i dy[i] * weights[i][j]
//     for (size_t j = 0; j < in_features; ++j) {
//       T sum = 0;
//       for (size_t i = 0; i < out_features; ++i) {
//         sum += dy[i] * weights[i][j];
//       }
//       dx[j] = sum;
//     }
//
//     return dx; // this goes to the previous layer
//   }
// };








#pragma once
#include <layers/Layer.h>
#include <tuple>
#include <utils/Tensor.h>

template <typename T, typename input = const Tensor<T>&,
          typename input_back = const Tensor<T>&,
          typename backward = std::tuple<Tensor<T>, Tensor<Tensor<T>>, Tensor<T>>>
class Dense : public Layer<T> {
public:
    int in_features, out_features;
    Tensor<T> biases;
    Tensor<Tensor<T>> weights;
    Tensor<T> last_x;    // store input from forward pass
    Tensor<Tensor<T>> dW; // gradient w.r.t weights
    Tensor<T> db;        // gradient w.r.t biases

    Dense(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
        // initialize biases and gradient
        biases.resize(out_f, 0.0);
        db.resize(out_f, 0.0);

        // initialize weights and weight gradients
        weights.resize(out_f, Tensor<T>(in_f, 0.0));
        dW.resize(out_f, Tensor<T>(in_f, 0.0));
    }

    Tensor<T> Forward(input x) override {
        last_x = x;
        Tensor<T> y(out_features, 0.0);

        for (int i = 0; i < out_features; i++) {
            for (int j = 0; j < in_features; j++) {
                y[i] += weights[i][j] * x[j];
            }
            y[i] += biases[i];
        }

        return y;
    }

    Tensor<T> Backward(const Tensor<T>& dy) override {
        Tensor<T> dx(in_features, 0.0);

        // compute gradients w.r.t weights
        for (int i = 0; i < out_features; i++) {
            for (int j = 0; j < in_features; j++) {
                dW[i][j] = dy[i] * last_x[j];
            }
        }

        // compute gradients w.r.t biases
        for (int i = 0; i < out_features; i++) {
            db[i] = dy[i];
        }

        // compute gradient w.r.t input
        for (int j = 0; j < in_features; j++) {
            T sum = 0;
            for (int i = 0; i < out_features; i++) {
                sum += dy[i] * weights[i][j];
            }
            dx[j] = sum;
        }

        return dx;
    }

    // optional: parameter update function
    void UpdateParameters(T lr) {
        for (int i = 0; i < out_features; i++) {
            for (int j = 0; j < in_features; j++) {
                weights[i][j] -= lr * dW[i][j];
            }
            biases[i] -= lr * db[i];
        }
    }
};

