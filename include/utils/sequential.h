// Sequential Function

#include "layers/Layer.h"
#include "utils/Tensor.h"

#pragma once

template <typename T> class Sequential : public Layer<T> {
	public:
	Tensor<T> Forward() {
		return 0;
	}
};
