#include "utils/Tensor.h"
#include <cstddef>

int Matmul1D(const Tensor<int> &a, const Tensor<int> &b) {
  if (a.Size() != b.Size())
    throw std::runtime_error("Size mismatch");
  int sum = 0;
  for (size_t i = 0; i < a.Size(); ++i)
    sum += a[i] * b[i];
  return sum;
}

// ---------- 2D matrix multiplication ----------
Tensor<Tensor<int>> Matmul2D(const Tensor<Tensor<int>> &A,
                             const Tensor<Tensor<int>> &B) {
  if (A[0].Size() != B.Size())
    throw std::runtime_error("Matrix dimension mismatch");
  Tensor<Tensor<int>> result;
  for (size_t i = 0; i < A.Size(); ++i) {
    Tensor<int> row;
    for (size_t j = 0; j < B[0].Size(); ++j) {
      int sum = 0;
      for (size_t k = 0; k < A[0].Size(); ++k)
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
  if (A[0][0].Size() != B[0].Size())
    throw std::runtime_error("Dimension mismatch");
  Tensor<Tensor<Tensor<int>>> result;
  for (size_t i = 0; i < A.Size(); ++i) {
    Tensor<Tensor<int>> matrix;
    for (size_t j = 0; j < B[0][0].Size(); ++j) {
      Tensor<int> row;
      for (size_t k = 0; k < A[0][0].Size(); ++k) {
        int sum = 0;
        for (size_t l = 0; l < A[0][0].Size(); ++l)
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
  for (size_t n = 0; n < A.Size(); ++n) {
    Tensor<Tensor<Tensor<int>>> batch;
    for (size_t i = 0; i < A[0].Size(); ++i) {
      Tensor<Tensor<int>> matrix;
      for (size_t j = 0; j < B[0][0][0].Size(); ++j) {
        Tensor<int> row;
        for (size_t k = 0; k < A[0][0][0].Size(); ++k) {
          int sum = 0;
          for (size_t l = 0; l < A[0][0][0].Size(); ++l)
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
