// Adam.h / Adam.cpp
// Adam optimizer.

#include <cmath>
#include <utils/Tensor.h>

template <typename T>
class AdamOptimizer {
private:
    Tensor<T> m; // first moment
    Tensor<T> v; // second moment
    int t;       // timestep
    T lr;
    T beta1;
    T beta2;
    T epsilon;

public:
    AdamOptimizer(int param_size, T lr_ = 0.001, T beta1_ = 0.9, T beta2_ = 0.999, T epsilon_ = 1e-8)
        : m(param_size, static_cast<T>(0)), v(param_size, static_cast<T>(0)),
          t(0), lr(lr_), beta1(beta1_), beta2(beta2_), epsilon(epsilon_) {}

    // Use Tensor<T> for both params and grads
    void update(Tensor<T>& params, const Tensor<T>& grads) {
        t++;
        for (size_t i = 0; i < params.size(); i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i];

            T m_hat = m[i] / (1 - std::pow(beta1, t));
            T v_hat = v[i] / (1 - std::pow(beta2, t));

            params[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
};

