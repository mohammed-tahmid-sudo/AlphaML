// Tensor.h / Tensor.cpp
// Defines Tensor class for multi-dimensional arrays.
// Supports basic math operations and broadcasting.

// Tensor.cpp
// Implementation of Tensor class methods.
// tensor_print.hpp
// tensor_nd_matmul.hpp

#pragma once

#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

// ---------------- Tensor class ----------------
template <typename T> class Tensor {
public:
  static_assert(std::is_arithmetic_v<T>,
                "Tensor only supports arithmetic or nested Tensor types.");

private:
  std::vector<T> data;

public:
  Tensor() = default;
  Tensor(std::initializer_list<T> list) : data(list) {}

};
