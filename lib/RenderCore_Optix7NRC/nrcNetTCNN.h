#include "core_settings.h"

#include <memory>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/encodings/oneblob.h>

#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <tiny-cuda-nn/trainer.h>
#include "shared_host_code/cudatools.h"

using network_precision_t = tcnn::network_precision_t;

class NRCTinyCudaNN {
public:
  void Init();
  void Train(
    CoreBuffer<NRCTraceBuf>* trainTraceBuffer,
    uint numTrainingRays,
    uint maxPathLength,  // 1, 2, 3.., NRC_MAX_TRAIN_PATHLENGTH
    uint batchSize,
    uint numTrainingSteps
  );
  void Inference(
    CoreBuffer<NRCNetInferenceInput>* infInputBuffer,
    uint numInfRays,
    CoreBuffer<NRCNetInferenceOutput>* infOutputBuffer
  );

  //using TCNNTrainInput = NRCNetInferenceInput;
  // Added two dummy segs
  struct alignas(sizeof(float) * 4) TCNNTrainInput{
    float3 rayIsect;
    float roughness;
    float2 rayDir;
    float2 normalDir;
    float3 diffuseRefl;
    float3 specularRefl;
    float dummies[2];
  };
  void Destroy();

private:
  bool initialized = false;

  tcnn::TrainableModel model;
	cudaStream_t inference_stream;
	cudaStream_t training_stream;

  // Column major by default, alter might cause tiny-cuda-nn deficiency
  std::unique_ptr<tcnn::GPUMatrix<float>> trainBatchInputCM;
  std::unique_ptr<tcnn::GPUMatrix<float>> trainBatchTargetCM;
};