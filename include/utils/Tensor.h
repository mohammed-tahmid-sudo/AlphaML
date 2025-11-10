// Tensor.h / Tensor.cpp
// Defines Tensor class for multi-dimensional arrays.
// Supports basic math operations and broadcasting.

// Tensor.cpp
// Implementation of Tensor class methods.

#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <typename T> class Tensor {
  static_assert(std::is_arithmetic<T>::value,
                "Tensor Only supports Numeric Values not `Strings`");

  std::vector<T> data;

public:
  Tensor() = default;
  Tensor(std::initializer_list<T> list) : data(list) {}

  size_t len() const { return data.size(); }

  // void push_back(const Tensor<T> &value) { data.push_back(value); }
  void push_back(const T &value) { data.push_back(value); }

  T &operator[](size_t i) { return data[i]; }
  const T &operator[](size_t i) const { return data[i]; }

  // Prints the value recursively
  void print() const {
    std::cout << "[";
    for (size_t i = 0; i < data.size(); ++i) {
      std::cout << data[i];
      if (i != data.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]";
  }

  // This is for the GPU
  T *raw_ptr() { return data.data(); }
};

// Recursive case: Tensor of Tensors
template <typename T> class Tensor<Tensor<T>> {
  std::vector<Tensor<T>> data;

public:
  Tensor() = default;
  Tensor(std::initializer_list<Tensor<T>> list) : data(list) {}

  void push_back(const Tensor<T> &value) { data.push_back(value); }

  size_t len() const { return data.size(); }

  Tensor<T> &operator[](size_t i) { return data[i]; }
  const Tensor<T> &operator[](size_t i) const { return data[i]; }

  void print() const {
    std::cout << "[";
    for (size_t i = 0; i < data.size(); ++i) {
      data[i].print();
      if (i != data.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
};

template <typename T>
Tensor<Tensor<T>> Matmul(const Tensor<Tensor<T>> &A,
                         const Tensor<Tensor<T>> &B) {
  if (A.len() == 0 || B.len() == 0) {
    throw std::invalid_argument("Matmul: input matrices must not be empty.");
  }

  // dimensions: A is m x n, B is n x p
  size_t m = A.len();
  size_t n = A[0].len();
  size_t b_rows = B.len();
  size_t p = B[0].len();

  // validate rectangular shape for A
  for (size_t i = 0; i < m; ++i) {
    if (A[i].len() != n)
      throw std::invalid_argument(
          "Matmul: matrix A is ragged (rows have different lengths).");
  }
  // validate rectangular shape for B
  for (size_t i = 0; i < b_rows; ++i) {
    if (B[i].len() != p)
      throw std::invalid_argument(
          "Matmul: matrix B is ragged (rows have different lengths).");
  }

  if (n != b_rows) {
    throw std::invalid_argument(
        "Matmul: inner dimensions must match (A.cols == B.rows).");
  }

  Tensor<Tensor<T>> C;
  // initialize C (m x p) with zeros
  for (size_t i = 0; i < m; ++i) {
    Tensor<T> row;
    for (size_t j = 0; j < p; ++j)
      row.push_back(T(0));
    C.push_back(row);
  }

  // multiplication: iterate so we reuse A[i][k] across j
  for (size_t i = 0; i < m; ++i) {
    for (size_t k = 0; k < n; ++k) {
      T a_ik = A[i][k];
      const Tensor<T> &Brow = B[k];
      Tensor<T> &Crow = C[i];
      for (size_t j = 0; j < p; ++j) {
        Crow[j] += a_ik * Brow[j];
      }
    }
  }

  return C;
}
