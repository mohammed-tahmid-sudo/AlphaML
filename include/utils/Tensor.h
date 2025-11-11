// Tensor.h / Tensor.cpp
// Defines Tensor class for multi-dimensional arrays.
// Supports basic math operations and broadcasting.

// Tensor.cpp
// Implementation of Tensor class methods.
// tensor_print.hpp
// tensor_nd_matmul.hpp
#pragma once
#include <iostream>
#include <vector>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>

// Forward
template<typename T> class Tensor;

// trait: is this a Tensor<...>?
template<typename T> struct is_tensor : std::false_type {};
template<typename U> struct is_tensor<Tensor<U>> : std::true_type {};

// scalar_type: peel nested Tensor<> to underlying scalar
template<typename T> struct scalar_type { using type = T; };
template<typename U> struct scalar_type<Tensor<U>> { using type = typename scalar_type<U>::type; };

// Basic Tensor (nested - like you had). Minimal container helpers included.
template <typename T>
class Tensor {
public:
    static_assert(std::is_arithmetic_v<T> || is_tensor<T>::value,
                  "Tensor only supports arithmetic or nested Tensor types.");

private:
    std::vector<T> data;

public:
    Tensor() = default;
    Tensor(std::initializer_list<T> list) : data(list) {}

    // container helpers
    void push_back(const T &v) { data.push_back(v); }
    void push_back(T &&v) { data.push_back(std::move(v)); }
    void pop_back() { data.pop_back(); }
    void clear() noexcept { data.clear(); }
    size_t size() const noexcept { return data.size(); }
    bool empty() const noexcept { return data.empty(); }

    T &operator[](size_t i) { return data[i]; }
    const T &operator[](size_t i) const { return data[i]; }

    T &front() { return data.front(); }
    const T &front() const { return data.front(); }
    T &back() { return data.back(); }
    const T &back() const { return data.back(); }

    auto begin() noexcept { return data.begin(); }
    auto end() noexcept { return data.end(); }
    auto begin() const noexcept { return data.begin(); }
    auto end() const noexcept { return data.end(); }

    // print single-line nested
    void print(std::ostream &os = std::cout) const {
        os << "[";
        for (size_t i = 0; i < data.size(); ++i) {
            if constexpr (is_tensor<T>::value) data[i].print(os);
            else os << data[i];
            if (i + 1 != data.size()) os << ", ";
        }
        os << "]";
    }
    void println(std::ostream &os = std::cout) const { print(os); os << '\n'; }

    // runtime shape (outer to inner)
    std::vector<size_t> shape() const {
        std::vector<size_t> s;
        s.push_back(size());
        if constexpr (is_tensor<T>::value) {
            if (!empty()) {
                auto sub = data[0].shape();
                s.insert(s.end(), sub.begin(), sub.end());
            }
        }
        return s;
    }
};

// ---------- Matmul (N-D with broadcasting on leading dims) ----------
// Strategy:
// - handle base numeric cases explicitly (1D dot, matrix×vector, matrix×matrix)
// - for higher-rank tensors (immediate elements are tensors) perform broadcasting over outer dims:
//     out_batch = broadcast shape of outer dims
//   where broadcasting rule is: if sizes equal -> use that index; if one is 1 -> reuse index 0; otherwise error.

// Forward declarations
template<typename A, typename B>
auto Matmul(const Tensor<A> &A_t, const Tensor<B> &B_t);

// 1D · 1D -> dot -> return Tensor<scalar> with single element (keeps consistent nested type)
template<typename Scalar>
auto Matmul(const Tensor<Scalar> &A, const Tensor<Scalar> &B)
-> std::enable_if_t<std::is_arithmetic_v<Scalar>, Tensor<Scalar>>
{
    if (A.size() != B.size()) throw std::invalid_argument("dot: size mismatch");
    Scalar acc = Scalar{};
    for (size_t i = 0; i < A.size(); ++i) acc += A[i] * B[i];
    return Tensor<Scalar>{ { acc } };
}

// Matrix (2D) × Vector (1D) -> Vector (1D)
template<typename Row>
auto Matmul(const Tensor<Tensor<Row>> &A, const Tensor<Row> &B)
-> std::enable_if_t<std::is_arithmetic_v<typename scalar_type<Row>::type>, Tensor<typename scalar_type<Row>::type>>
{
    using scalar_t = typename scalar_type<Row>::type;
    if (A.empty()) throw std::invalid_argument("matmul: empty A");
    size_t m = A.size();
    size_t n = A[0].size();
    if (n != B.size()) throw std::invalid_argument("matmul: inner dim mismatch (matrix x vector)");
    Tensor<scalar_t> out;
    for (size_t i = 0; i < m; ++i) {
        scalar_t acc = scalar_t{};
        for (size_t k = 0; k < n; ++k) acc += A[i][k] * B[k];
        out.push_back(acc);
    }
    return out;
}

// Vector (1D) × Matrix (2D) -> Vector (1D) (treat vector as row vector)
template<typename Row>
auto Matmul(const Tensor<Row> &A, const Tensor<Tensor<Row>> &B)
-> std::enable_if_t<std::is_arithmetic_v<typename scalar_type<Row>::type>, Tensor<typename scalar_type<Row>::type>>
{
    using scalar_t = typename scalar_type<Row>::type;
    size_t n = A.size();
    if (B.empty()) throw std::invalid_argument("matmul: empty B");
    size_t nB = B.size();
    size_t p = B[0].size();
    if (n != nB) throw std::invalid_argument("matmul: inner dim mismatch (vector x matrix)");
    Tensor<scalar_t> out;
    // result is length p
    for (size_t j = 0; j < p; ++j) {
        scalar_t acc = scalar_t{};
        for (size_t k = 0; k < n; ++k) acc += A[k] * B[k][j];
        out.push_back(acc);
    }
    return out;
}

// Matrix (2D) × Matrix (2D)
template<typename RowA, typename RowB>
auto Matmul(const Tensor<Tensor<RowA>> &A, const Tensor<Tensor<RowB>> &B)
-> std::enable_if_t<std::is_arithmetic_v<typename scalar_type<RowA>::type> &&
                    std::is_arithmetic_v<typename scalar_type<RowB>::type>,
                    Tensor<Tensor<typename scalar_type<RowA>::type>>>
{
    using scalar_t = typename scalar_type<RowA>::type;
    if (A.empty() || B.empty()) throw std::invalid_argument("matmul: empty matrix");
    size_t m = A.size();
    size_t n = A[0].size();
    size_t nB = B.size();
    if (n != nB) throw std::invalid_argument("matmul: inner dim mismatch (matrix x matrix)");
    size_t p = B[0].size();
    Tensor<Tensor<scalar_t>> C;
    for (size_t i = 0; i < m; ++i) {
        Tensor<scalar_t> row;
        for (size_t j = 0; j < p; ++j) {
            scalar_t acc = scalar_t{};
            for (size_t k = 0; k < n; ++k) acc += A[i][k] * B[k][j];
            row.push_back(acc);
        }
        C.push_back(std::move(row));
    }
    return C;
}

// Recursive / batched cases with broadcasting:
// Case A and B have immediate elements that are themselves tensors (i.e., rank >= 2 and we are at a leading batch dim).
template<typename SubA, typename SubB>
auto Matmul(const Tensor<SubA> &A, const Tensor<SubB> &B)
-> std::enable_if_t<is_tensor<SubA>::value && is_tensor<SubB>::value,
                    Tensor<decltype(Matmul(std::declval<SubA>(), std::declval<SubB>()))>>
{
    // Determine broadcasted outer size:
    size_t sa = A.size();
    size_t sb = B.size();
    size_t s_out = 0;
    if (sa == sb) s_out = sa;
    else if (sa == 1) s_out = sb;
    else if (sb == 1) s_out = sa;
    else {
        throw std::invalid_argument("matmul: cannot broadcast outer dimensions (sizes differ and neither is 1).");
    }
    using ElemOut = decltype(Matmul(std::declval<SubA>(), std::declval<SubB>()));
    Tensor<ElemOut> out;
    for (size_t i = 0; i < s_out; ++i) {
        const SubA &subA = (sa == 1 ? A[0] : A[i]);
        const SubB &subB = (sb == 1 ? B[0] : B[i]);
        out.push_back(Matmul(subA, subB));
    }
    return out;
}

// Case: A has extra leading dims but B does not (B is lower-rank). Broadcast B as size-1 batch.
template<typename SubA, typename Bleaf>
auto Matmul(const Tensor<SubA> &A, const Tensor<Bleaf> &B)
-> std::enable_if_t<is_tensor<SubA>::value && !is_tensor<Bleaf>::value,
                    Tensor<decltype(Matmul(std::declval<SubA>(), std::declval<Tensor<Bleaf>>()))>>
{
    // treat B as batch size 1, broadcast to A.size()
    size_t sa = A.size();
    Tensor<decltype(Matmul(std::declval<SubA>(), std::declval<Tensor<Bleaf>>()))> out;
    for (size_t i = 0; i < sa; ++i) {
        out.push_back(Matmul(A[i], B));
    }
    return out;
}

// Case: B has extra leading dims but A does not. Broadcast A as size-1 batch.
template<typename Aleaf, typename SubB>
auto Matmul(const Tensor<Aleaf> &A, const Tensor<SubB> &B)
-> std::enable_if_t<!is_tensor<Aleaf>::value && is_tensor<SubB>::value,
                    Tensor<decltype(Matmul(std::declval<Tensor<Aleaf>>(), std::declval<SubB>()))>>
{
    size_t sb = B.size();
    Tensor<decltype(Matmul(std::declval<Tensor<Aleaf>>(), std::declval<SubB>()))> out;
    for (size_t i = 0; i < sb; ++i) {
        out.push_back(Matmul(A, B[i]));
    }
    return out;
}

