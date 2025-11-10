// // Tensor.cpp
// // Implementation of Tensor class methods.
//
// #include <iostream>
// #include <vector>
//
// template <typename T> class Tensor {
//   static_assert(std::is_arithmetic<T>::value,
//                 "Tensor Only supports Numeric Values not `Strings`");
//
//   std::vector<T> data;
//
// public:
//   Tensor(std::initializer_list<T> list) : data(list) {}
//
//   size_t len() { return data.size(); }
//
//   void push_back() { return data.push_back(); }
//
//   T &operator[](size_t i) { return data[i]; }
//   const T &operator[](size_t i) const { return data[i]; }
//
//   // Prints the value recursively
//   void print() const {
//     for (const auto &v : data) {
//       if constexpr (std::is_same_v<T, Tensor<typename T::value_type>>) {
//         v.print(); // recursive call for nested MyVector
//       } else {
//         std::cout << v << " ";
//       }
//     }
//     std::cout << "\n";
//   }
//
//   // This is for the GPU
//   T *raw_ptr() { return data.data(); }
// };


