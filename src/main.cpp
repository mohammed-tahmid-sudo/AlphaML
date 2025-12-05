#include "layers/Activation.h"
#include <optimizers/Adam.h>
#include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"
#include <losses/CrossEntropyLoss.h>
#include "utils/sequential.h"
#include <layers/flatten.h>

int main() {
    // 1. Create layers
    flatten3d<float> flatten;
    Dense1D<float>* dense1 = new Dense1D<float>(6, 3);  // input size 6 -> output 3
    Dense1D<float>* dense2 = new Dense1D<float>(3, 2);  // output size 3 -> 2

    // 2. Create sequential model
    Sequential<float> model({&flatten, dense1, dense2});

    // 3. Dummy 3D input data (batch size 2, 2x3)
    Tensor<Tensor<Tensor<float>>> train_x(2);  // batch of 2
    for (int b = 0; b < 2; ++b) {
        Tensor<Tensor<float>> sample(2);  // 2x3
        for (int i = 0; i < 2; ++i) {
            Tensor<float> row(3);
            for (int j = 0; j < 3; ++j)
                row[j] = (float)(b + i + j);  // fill with some numbers
            sample[i] = row;
        }
        train_x[b] = sample;
    }

    // Dummy labels
    Tensor<Tensor<float>> train_y(2);
    for (int i = 0; i < 2; ++i) {
        Tensor<float> label(2);
        label[0] = 1.0; label[1] = 0.0;  // dummy one-hot
        train_y[i] = label;
    }

    // 4. Train (single epoch, learning rate 0.01)
    model.train(train_x, train_y, 1, 2, 0.01);

    // 5. Test
    std::cout << "Testing on first sample:\n";
    Tensor<float> output = model.Forward(train_x[0]);
    output.print();

    // 6. Clean up
    delete dense1;
    delete dense2;

    return 0;
}
