#include "layers/Activation.h"
#include "layers/Dense.h"
#include "losses/Loss.h"
#include "optimizers/Optimizer.h"
#include "utils/Tensor.h"
#include "utils/sequential.h"
#include "utils/Utils.h"
#include <layers/flatten.h>
#include <losses/CrossEntropyLoss.h>
#include <optimizers/Adam.h>


int main() {
    Tensor<Tensor<Tensor<Tensor<int>>>> data = {
        {{{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, {16,17,18,19,20}},
          {{21,22,23,24,25}, {26,27,28,29,30}, {31,32,33,34,35}, {36,37,38,39,40}},
          {{41,42,43,44,45}, {46,47,48,49,50}, {51,52,53,54,55}, {56,57,58,59,60}}},

         {{{61,62,63,64,65}, {66,67,68,69,70}, {71,72,73,74,75}, {76,77,78,79,80}},
          {{81,82,83,84,85}, {86,87,88,89,90}, {91,92,93,94,95}, {96,97,98,99,100}},
          {{101,102,103,104,105}, {106,107,108,109,110}, {111,112,113,114,115}, {116,117,118,119,120}}}}
    };

    auto batches = batch(data, 1); // batch size 1 for testing

    for (size_t i = 0; i < batches.size(); i++) {
        std::cout << "Batch " << i << ":\n";
        for (auto& mat3d : batches[i]) {
            for (auto& mat2d : mat3d) {
                for (auto& row : mat2d) {
                    for (int x : row) std::cout << x << ' ';
                    std::cout << "\n";
                }
                std::cout << "---\n";
            }
            std::cout << "=====\n";
        }
        std::cout << "********\n";
    }
}

