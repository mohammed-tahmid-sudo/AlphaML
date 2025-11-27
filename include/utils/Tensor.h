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
  Tensor(size_t n, const T &value = T()) : data(n, value) {}

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
  size_t size() const { return data.size(); }
  T &Front() { return data.front(); }
  T &Back() { return data.back(); }

  T &operator[](size_t i) { return data[i]; }
  const T &operator[](size_t i) const { return data[i]; }
};

template <typename T> Tensor<T> Add(const Tensor<T> &A, const Tensor<T> &B) {
  Tensor<T> result;
  for (size_t i = 0; i < A.size(); ++i) {
    if constexpr (std::is_class_v<T>)
      result.Push_back(Add(A[i], B[i])); // recurse for nested tensors
    else
      result.Push_back(A[i] + B[i]); // base case
  }
  return result;
}

template <typename T> Tensor<T> Sub(const Tensor<T> &A, const Tensor<T> &B) {
  Tensor<T> result;
  for (size_t i = 0; i < A.size(); ++i) {
    if constexpr (std::is_class_v<T>)
      result.Push_back(Sub(A[i], B[i])); // recurse for nested tensors
    else
      result.Push_back(A[i] - B[i]); // base case
  }
  return result;
}

// 1D vector dot product

template <typename T> 
Tensor<T> Matmul1D(const Tensor<int> &a, const Tensor<int> &b);

// 2D matrix multiplication
template <typename T>
Tensor<Tensor<T>> Matmul2D(const Tensor<Tensor<int>> &A,
                           const Tensor<Tensor<int>> &B);

// 3D tensor multiplication along last two axes
template <typename T>
Tensor<Tensor<Tensor<int>>> Matmul3D(const Tensor<Tensor<Tensor<int>>> &A,
                                     const Tensor<Tensor<Tensor<int>>> &B);

// 4D tensor multiplication along last two axes
template <typename T>
Tensor<Tensor<Tensor<Tensor<int>>>>
Matmul4D(const Tensor<Tensor<Tensor<Tensor<int>>>> &A,
         const Tensor<Tensor<Tensor<Tensor<int>>>> &B);
