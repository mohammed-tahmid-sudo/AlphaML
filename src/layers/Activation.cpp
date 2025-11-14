// Activation.cpp
// Implementation of activation functions.


#include <iostream>
#include <vector>
#include <algorithm>
#include <type_traits>

// Base case: arithmetic types
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
relu(const T& x) {
    return std::max(T(0), x);
}

// Recursive case: containers
template <typename Container>
auto relu(const Container& input) {
    using ValueType = decltype(relu(*input.begin())); // type of element after ReLU
    Container output;
    output.reserve(input.size());
    for (const auto& elem : input) {
        output.push_back(relu(elem)); // recursive call
    }
    return output;
}

int main() {
    // 1D example
    std::vector<float> v1 = {-1.0, 2.0, -3.5};
    auto r1 = relu(v1);

    for (float val : r1) std::cout << val << " ";
    std::cout << "\n";

    // 2D example
    std::vector<std::vector<float>> v2 = {{-1, 2}, {-3, 4}};
    auto r2 = relu(v2);

    for (const auto& row : r2) {
        for (float val : row) std::cout << val << " ";
        std::cout << "\n";
    }
}

