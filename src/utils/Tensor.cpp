#include "utils/Tensor.h"
#include <cstddef>

int Matmul1D(const Tensor<int> &a, const Tensor<int> &b) {
  if (a.size() != b.size())
    throw std::runtime_error("size mismatch");
  int sum = 0;
  for (size_t i = 0; i < a.size(); ++i)
    sum += a[i] * b[i];
  return sum;
}

// ---------- 2D matrix multiplication ----------
Tensor<Tensor<int>> Matmul2D(const Tensor<Tensor<int>> &A,
                             const Tensor<Tensor<int>> &B) {
  if (A[0].size() != B.size())
    throw std::runtime_error("Matrix dimension mismatch");
  Tensor<Tensor<int>> result;
  for (size_t i = 0; i < A.size(); ++i) {
    Tensor<int> row;
    for (size_t j = 0; j < B[0].size(); ++j) {
      int sum = 0;
      for (size_t k = 0; k < A[0].size(); ++k)
        sum += A[i][k] * B[k][j];
      row.Push_back(sum);
    }
    result.Push_back(row);
  }
  return result;
}

// ---------- 3D tensor multiplication along last two axes ----------
Tensor<Tensor<Tensor<int>>> Matmul3D(const Tensor<Tensor<Tensor<int>>> &A,
                                     const Tensor<Tensor<Tensor<int>>> &B) {
  if (A[0][0].size() != B[0].size())
    throw std::runtime_error("Dimension mismatch");
  Tensor<Tensor<Tensor<int>>> result;
  for (size_t i = 0; i < A.size(); ++i) {
    Tensor<Tensor<int>> matrix;
    for (size_t j = 0; j < B[0][0].size(); ++j) {
      Tensor<int> row;
      for (size_t k = 0; k < A[0][0].size(); ++k) {
        int sum = 0;
        for (size_t l = 0; l < A[0][0].size(); ++l)
          sum += A[i][k][l] * B[i][l][j];
        row.Push_back(sum);
      }
      matrix.Push_back(row);
    }
    result.Push_back(matrix);
  }
  return result;
}

// ---------- 4D tensor multiplication along last two axes ----------
Tensor<Tensor<Tensor<Tensor<int>>>>
Matmul4D(const Tensor<Tensor<Tensor<Tensor<int>>>> &A,
         const Tensor<Tensor<Tensor<Tensor<int>>>> &B) {
  Tensor<Tensor<Tensor<Tensor<int>>>> result;
  for (size_t n = 0; n < A.size(); ++n) {
    Tensor<Tensor<Tensor<int>>> batch;
    for (size_t i = 0; i < A[0].size(); ++i) {
      Tensor<Tensor<int>> matrix;
      for (size_t j = 0; j < B[0][0][0].size(); ++j) {
        Tensor<int> row;
        for (size_t k = 0; k < A[0][0][0].size(); ++k) {
          int sum = 0;
          for (size_t l = 0; l < A[0][0][0].size(); ++l)
            sum += A[n][i][k][l] * B[n][i][l][j];
          row.Push_back(sum);
        }
        matrix.Push_back(row);
      }
      batch.Push_back(matrix);
    }
    result.Push_back(batch);
  }
  return result;
}
