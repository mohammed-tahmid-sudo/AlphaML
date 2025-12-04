#include <cstddef>
#include <layers/Layer.h>
#include <utils/Tensor.h>

template <typename T>
class flatten3d : public Layer<T, Tensor<Tensor<Tensor<T>>>, Tensor<T>,
                               Tensor<Tensor<Tensor<T>>>> {
  size_t d1, d2, d3;

public:
  Tensor<T> Forward(const Tensor<Tensor<Tensor<T>>> &x) {
    Tensor<T> y;
    for (size_t i = 0; i < x.size(); ++i) {
      for (size_t j = 0; j < x[i].size(); ++j) {
        for (size_t k = 0; k < x[i][j].size(); ++k) {
          y.Push_back(x[i][j][k]);
        }
      }
    }
    return y;
  }

  Tensor<Tensor<Tensor<T>>> Backward(const Tensor<T> &grad_out) {
    Tensor<Tensor<Tensor<T>>> grad_in(d1);

    size_t idx = 0;
    for (size_t i = 0; i < d1; i++) {
      grad_in[i].Resize(d2);
      for (size_t j = 0; j < d2; j++) {
        grad_in[i][j].Resize(d3);
        for (size_t k = 0; k < d3; k++) {
          grad_in[i][j][k] = grad_out[idx++];
        }
      }
    }
    return grad_in;
  }
};
