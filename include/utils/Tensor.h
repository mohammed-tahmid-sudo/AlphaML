// Tensor.h / Tensor.cpp
// Defines Tensor class for multi-dimensional arrays.
// Supports basic math operations and broadcasting.

// Tensor.cpp
// Implementation of Tensor class methods.
// tensor_print.hpp
// tensor_nd_matmul.hpp

#pragma once

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <vector>

// ---------------- Tensor class ----------------
template <typename T> class Tensor {
public:
  static_assert(std::is_arithmetic_v<T> || std::is_class_v<T>,
                "Tensor only supports arithmetic or nested Tensor types.");

private:
  std::vector<T> data;

public:
  Tensor() = default;
  Tensor(std::initializer_list<T> list) : data(list) {}

  void print() {
    std::cout << "[";
    for (size_t i = 0; i < data.size(); ++i) {
      if constexpr (std::is_class_v<T>) // if element is another Tensor
        data[i].print();
      else
        std::cout << data[i];

      if (i + 1 < data.size())
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
  void Push_back(const T &dt) { data.push_back(dt); }
};
