// main_fixed.cpp
#include "layers/Dense.h"
#include "losses/CrossEntropyLoss.h" // optional if not already included
#include "utils/sequential.h"
#include <algorithm>
#include <cmath>
#include <iostream>

template <typename T>
T CrossEntropyLossFromLogits(const Tensor<T> &logits, int target_class) {
  T maxv = logits[0];
  for (size_t i = 1; i < logits.size(); ++i)
    if (logits[i] > maxv)
      maxv = logits[i];
  T sum = 0;
  for (size_t i = 0; i < logits.size(); ++i)
    sum += std::exp(logits[i] - maxv);
  T lse = maxv + std::log(sum);
  return lse - logits[target_class];
}

template <typename T>
Tensor<T> CrossEntropyGradFromLogits(const Tensor<T> &logits,
                                     int target_class) {
  size_t C = logits.size();
  Tensor<T> grad(C);
  T maxv = logits[0];
  for (size_t i = 1; i < C; ++i)
    if (logits[i] > maxv)
      maxv = logits[i];
  T sum = 0;
  for (size_t i = 0; i < C; ++i)
    sum += std::exp(logits[i] - maxv);
  for (size_t i = 0; i < C; ++i) {
    T prob = std::exp(logits[i] - maxv) / sum;
    grad[i] = prob - (i == (size_t)target_class ? (T)1 : (T)0);
  }
  return grad;
}

template <typename T> int argmax(const Tensor<T> &t) {
  int idx = 0;
  T best = t[0];
  for (size_t i = 1; i < t.size(); ++i)
    if (t[i] > best) {
      best = t[i];
      idx = (int)i;
    }
  return idx;
}

int main() {
  using T = double;

  // Data (your provided tensors)
  Tensor<Tensor<int>> X = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                           {2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                           {3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                           {4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
                           {5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
                           {6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                           {7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                           {8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
                           {9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                           {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                           {11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
                           {12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
                           {13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
                           {14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                           {15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                           {16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                           {17, 18, 19, 20, 21, 22, 23, 24, 25, 26},
                           {18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
                           {19, 20, 21, 22, 23, 24, 25, 26, 27, 28},
                           {20, 21, 22, 23, 24, 25, 26, 27, 28, 29},
                           {21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
                           {22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
                           {23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
                           {24, 25, 26, 27, 28, 29, 30, 31, 32, 33},
                           {25, 26, 27, 28, 29, 30, 31, 32, 33, 34},
                           {26, 27, 28, 29, 30, 31, 32, 33, 34, 35},
                           {27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
                           {28, 29, 30, 31, 32, 33, 34, 35, 36, 37},
                           {29, 30, 31, 32, 33, 34, 35, 36, 37, 38},
                           {30, 31, 32, 33, 34, 35, 36, 37, 38, 39},
                           {31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
                           {32, 33, 34, 35, 36, 37, 38, 39, 40, 41},
                           {33, 34, 35, 36, 37, 38, 39, 40, 41, 42},
                           {34, 35, 36, 37, 38, 39, 40, 41, 42, 43},
                           {35, 36, 37, 38, 39, 40, 41, 42, 43, 44},
                           {36, 37, 38, 39, 40, 41, 42, 43, 44, 45},
                           {37, 38, 39, 40, 41, 42, 43, 44, 45, 46},
                           {38, 39, 40, 41, 42, 43, 44, 45, 46, 47},
                           {39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
                           {40, 41, 42, 43, 44, 45, 46, 47, 48, 49},
                           {31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
                           {32, 33, 34, 35, 36, 37, 38, 39, 40, 41},
                           {33, 34, 35, 36, 37, 38, 39, 40, 41, 42},
                           {34, 35, 36, 37, 38, 39, 40, 41, 42, 43},
                           {35, 36, 37, 38, 39, 40, 41, 42, 43, 44},
                           {36, 37, 38, 39, 40, 41, 42, 43, 44, 45},
                           {37, 38, 39, 40, 41, 42, 43, 44, 45, 46},
                           {38, 39, 40, 41, 42, 43, 44, 45, 46, 47},
                           {39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
                           {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}};

  Tensor<int> y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  const size_t N = X.size();
  if (N != y.size()) {
    std::cerr << "X/y size mismatch\n";
    return 1;
  }

  // model
  Sequential<T> model(
      {new Dense<T>(10, 16), new Dense<T>(16, 8), new Dense<T>(8, 2)});

  double lr = 1e-2;
  int epochs = 2000;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double epoch_loss = 0.0;
    int correct = 0;

    for (size_t si = 0; si < N; ++si) {
      const Tensor<int> &xs_int = X[si];
      Tensor<T> x(xs_int.size());
      for (size_t j = 0; j < xs_int.size(); ++j)
        x[j] = (T)xs_int[j];

      Tensor<T> logits = model.Forward(x);

      int target = y[si];

      // --- create one-hot Tensor<T> target for the project's CrossEntropyLoss
      // ---
      Tensor<T> y_onehot(logits.size());
      for (size_t k = 0; k < logits.size(); ++k)
        y_onehot[k] = 0;
      y_onehot[target] = 1;

      // --- use your project's CrossEntropyLoss that expects (Tensor, Tensor)
      // ---
      T loss = CrossEntropyLoss(logits, y_onehot);

      // gradient: use stable softmax difference (compatible with
      // Dense::Backward)
      Tensor<T> grad = CrossEntropyGradFromLogits(logits, target);

      model.Backward(grad);
      model.UpdateParameters(lr);

      epoch_loss += (double)loss;
      if (argmax(logits) == target)
        ++correct;
    }

    if (epoch % 10 == 0) {
      std::cout << "Epoch " << epoch << " avg_loss=" << (epoch_loss / N)
                << " acc=" << (double)correct / N << "\n";
    }
  }

  // final accuracy
  int correct_final = 0;
  for (size_t si = 0; si < N; ++si) {
    const Tensor<int> &xs_int = X[si];
    Tensor<T> x(xs_int.size());
    for (size_t j = 0; j < xs_int.size(); ++j)
      x[j] = (T)xs_int[j];
    Tensor<T> logits = model.Forward(x);
    if (argmax(logits) == y[si])
      ++correct_final;
  }
  std::cout << "Final accuracy = " << (double)correct_final / N << "\n";
  return 0;
}
