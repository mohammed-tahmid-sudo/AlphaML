// Utils.h / Utils.cpp
// Helper functions: metrics, printing, etc.

#include <utils/Tensor.h>
#include <type_traits>

// base case: the type is not a container
template<typename T>
struct is_container : std::false_type {};

// specialize for Tensor<T>
template<typename U>
struct is_container<Tensor<U>> : std::true_type {};

// Generic batching for any container
template <typename Container>
auto Batching(const Container& data, size_t batch_size) 
    -> std::enable_if_t<is_container<Container>::value, Tensor<Container>> 
{
    Tensor<Container> out;
    for (size_t i = 0; i < data.size(); i += batch_size) {
        out.emplace_back(data.begin() + i,
                         data.begin() + std::min(data.size(), i + batch_size));
    }
    return out;
}

