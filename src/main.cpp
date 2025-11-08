#include <iostream>
#include "utils/Tensor.h"
#include "layers/Dense.h"
#include "layers/Activation.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"

int main() {
    std::cout << "Torch Alt ML Library Test\n";

    // Example stub usage
    std::cout << "Creating a Tensor...\n";
    // Tensor t({2, 3}); // Uncomment when Tensor class is implemented

    std::cout << "Adding a Dense layer...\n";
    // Dense d(3, 2); // Uncomment when Dense class is implemented

    std::cout << "Applying ReLU activation...\n";
    // ReLU relu; // Uncomment when Activation class is implemented

    std::cout << "ML Library test finished!\n";
    return 0;
}

