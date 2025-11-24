// Layer.h
// Abstract base class for all neural network layers.
// Requires forward() and backward() methods.

template <typename T> struct Layer {

  // TODO: Add Backwards
  virtual T Forward(T X);
  virtual ~Layer();
};
