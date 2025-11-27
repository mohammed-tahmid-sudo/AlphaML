#include "layers/Activation.h"
#include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"
#include "utils/sequential.h"

int main() {
// --- Layer definitions ---
    Dense1D<double> l1(3, 3);
    Dense1D<double> l2(3, 2);
    Dense1D<double> l3(2, 1);

    // --- Assign weights and biases ---
    l1.weights = {
        {0.5, -1.0, 0.3},
        {0.2, 0.1, -0.5},
        {-0.4, 0.6, -0.2}
    };
    l1.bias = {0.0, 0.1, -0.1};

    l2.weights = {
        {0.4, -0.6, 0.2},
        {-0.3, 0.7, 0.5}
    };
    l2.bias = {0.2, -0.1};

    l3.weights = {
        {0.6, -0.2}
    };
    l3.bias = {0.5};

    // --- Input ---
    Tensor<double> x{1.0, 2.0, 3.0};

    // --- Sequential model ---
    Sequential<double> model{{&l1, &l2, &l3}}; // No ReLU

    // --- Forward pass ---
    auto y = model.Forward(x);

    // --- Print final output ---
    std::cout << "Final output: ";
    y.print();

    return 0;}
