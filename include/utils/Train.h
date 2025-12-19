
#include <layers/Layer.h>
#include <utils/color_text.h>
#include <utils/

template <typename DataTypeInput, typename DataTypeOutput, typename ModelType>
class Trainer {

public:
  DataTypeInput dataInput;
  DataTypeOutput dataOutput;
  Layer<ModelType> &model;
  float lr = 0.1f;
  int Batch = 32;

  Trainer(DataTypeInput input, DataTypeOutput output,
          Layer<ModelType> &MachingLearningModel, float LearningRate = 0.01f,
          int btch = 32)
      : dataInput(input), dataOutput(output), model(MachingLearningModel),
        lr(LearningRate), Batch(btch) {}

  void Train() {
	auto batched_input = Batching(&dataInput, Batch);
	auto batched_output = Batching(&dataOutput, Batch);


  };
};
