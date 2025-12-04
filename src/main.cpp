#include "layers/Activation.h"
#include <optimizers/Adam.h>
// #include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"
#include <losses/CrossEntropyLoss.h>
// #include "utils/sequential.h"
#include <layers/flatten.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(std::time(nullptr));

    // Create a 3D tensor [2, 3, 4]
    Tensor<float> x({2, 3, 4});

    // Fill with random values
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(std::rand()) / RAND_MAX;

    std::cout << "Original tensor:\n";
    x.print();

    // Flatten
    flatten3d<float> flatten;
    Tensor<float> out = flatten.Forward(x);

    std::cout << "Flattened tensor:\n";
    out.print();

    // Backward with gradient = 1
    Tensor<float> grad_output(out.size());
    for (size_t i = 0; i < grad_output.size(); ++i)
        grad_output[i] = 1.0f;

    Tensor<float> grad_input = flatten.Backward(grad_output);
    std::cout << "Gradient reshaped back:\n";
    grad_input.print();

    return 0;
}

