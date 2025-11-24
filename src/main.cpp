#include "layers/Activation.h"
#include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"
#include <algorithm>

int main() {
    // 1D example
    Tensor<int> v1{1,2,3};
    Tensor<int> v2{4,5,6};
    std::cout << "1D dot product: " << Matmul1D(v1,v2) << "\n";

    // 2D example
    Tensor<Tensor<int>> m1{{1,2},{3,4}};
    Tensor<Tensor<int>> m2{{5,6},{7,8}};
    auto m2d = Matmul2D(m1,m2);
    m2d.print();

    // 3D example
    Tensor<Tensor<Tensor<int>>> t3d1{{{1,2},{3,4}}, {{5,6},{7,8}}};
    Tensor<Tensor<Tensor<int>>> t3d2{{{1,0},{0,1}}, {{1,0},{0,1}}};
    auto t3d = Matmul3D(t3d1, t3d2);
    t3d.print();

    // 4D example
    Tensor<Tensor<Tensor<Tensor<int>>>> t4d1{{{{1,2},{3,4}}}};
    Tensor<Tensor<Tensor<Tensor<int>>>> t4d2{{{{1,0},{0,1}}}};
    auto t4d = Matmul4D(t4d1, t4d2);
    t4d.print();

    return 0;
}

