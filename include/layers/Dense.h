// Dense.h / Dense.cpp
// Fully connected layer. Implements forward and backward pass.
#include <utils/Tensor.h>
#include <layers/Layer.h>



// Derived class
template <typename T>
class LinearLayer : public Layer<T> {
public:
    T Forward(T X) override {
        std::cout << "Forward called with: " << X << "\n";
        return X; // just an example
    }
};
