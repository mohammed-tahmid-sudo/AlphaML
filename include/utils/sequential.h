// Sequential Function

Pragma once
#pragma once

#include "layers/Layer.h"
#include "utils/Tensor.h"
#include <initializer_list>
#include <vector>

// template <typename T> class Sequential : public Layer<T> {
// public:
//   Sequential() = default;
//
//   std::vector<Layer<T> *> layers;
//
//   Sequential(std::initializer_list<Layer<T> *> list) {
//     for (auto l : list)
//       layers.push_back(l);
//   }
//
//   Tensor<T> Forward(const Tensor<T> &x) override {
//
//     Tensor<T> output = x;
//
//     for (auto *l : layers) {
//
//       output = l->Forward(output);
//     }
//
//     return output;
//   }
//
//   std::vector<std::vector<std::vector<std::vector<T>>>>
//   create_batches(const std::vector<std::vector<std::vector<T>>> &data,
//                  size_t batch_size) {
//     std::vector<std::vector<std::vector<std::vector<T>>>> batches;
//     size_t n = data.size();
//
//     for (size_t i = 0; i < n; i += batch_size) {
//       size_t end = std::min(i + batch_size, n);
//       std::vector<std::vector<std::vector<T>>> batch(data.begin() + i,
//                                                      data.begin() + end);
//       batches.push_back(batch);
//     }
//
//     void train(Layer<T> & Model, Tensor<Tensor<Tensor<Tensor<T>>>> & train_x,
//                Tensor<Tensor<Tensor<Tensor<T>>>> & test_x, Tensor<T> & train_y,
//                Tensor<T> & test_y, size_t epoch, int batch_size = 32,
//                float Learning_rate = 0.1, std::string loss = "crossentropy",
//                std::string optimizer = "adam",
//                const std::string device = "CPU") {
//
//       auto batches_x = create_batches(train_x, batch_size);
//       auto batches_y = create_batches(train_y, batch_size);
//
//     }
//   }
// };



template <typename T>
class Sequential : public Layer<T> {
public:
    Sequential() = default;
    Tensor<Layer<T>*> layers;  // store layers in Tensor

    Sequential(std::initializer_list<Layer<T>*> list) {
        for (auto l : list)
            layers.Push_back(l);
    }

    Tensor<T> Forward(const Tensor<T>& x) override {
        Tensor<T> output = x;
        for (size_t i = 0; i < layers.Size(); ++i)
            output = layers[i]->Forward(output);
        return output;
    }

    Tensor<T> Backward(const Tensor<T>& grad) override {
        Tensor<T> grad_out = grad;
        for (int i = layers.Size() - 1; i >= 0; --i)
            grad_out = layers[i]->Backward(grad_out);
        return grad_out;
    }

    Tensor<Tensor<T>> create_batches(const Tensor<Tensor<T>>& data, size_t batch_size) {
        Tensor<Tensor<T>> batches;
        size_t n = data.Size();
        for (size_t i = 0; i < n; i += batch_size) {
            size_t end = std::min(i + batch_size, n);
            Tensor<Tensor<T>> batch;
            for (size_t j = i; j < end; ++j)
                batch.Push_back(data[j]);
            batches.Push_back(batch);
        }
        return batches;
    }

    void train(Tensor<Tensor<T>>& train_x, Tensor<Tensor<T>>& train_y,
               size_t epochs, size_t batch_size = 32, float learning_rate = 0.1) {

        Tensor<Tensor<Tensor<T>>> batches_x = create_batches(train_x, batch_size);
        Tensor<Tensor<Tensor<T>>> batches_y = create_batches(train_y, batch_size);

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            T epoch_loss = 0;
            for (size_t i = 0; i < batches_x.Size(); ++i) {
                // Forward
                Tensor<T> output = Forward(batches_x[i]);

                // Loss (MSE example)
                Tensor<T> loss = output - batches_y[i];
                epoch_loss += loss.Sum();

                // Backward
                Tensor<T> grad = loss * (2.0 / batch_size);
                Backward(grad);

                // Update parameters
                for (size_t j = 0; j < layers.Size(); ++j)
                    layers[j]->UpdateParams(learning_rate);
            }
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " Loss: " 
                      << epoch_loss / batches_x.Size() << "\n";
        }
    }

    void print_output(const Tensor<T>& x) {
        x.print();
    }
};

