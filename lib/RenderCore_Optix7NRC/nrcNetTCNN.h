#include "core_settings.h"

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
  float Train(
    CoreBuffer<NRCTraceBuf>* trainTraceBuffer,
    uint numTrainingRays,
    uint maxPathLength,
    uint batchSize
  );
  void Inference(
    CoreBuffer<NRCNetInferenceInput>* infInputBuffer,
    uint numInfRays,
    CoreBuffer<NRCNetInferenceOutput>* infOutputBuffer
  );

private:
  std::shared_ptr<tcnn::Loss<network_precision_t>> loss;
	std::shared_ptr<tcnn::Optimizer<network_precision_t>> optimizer;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<network_precision_t>> network;
	std::shared_ptr<tcnn::Trainer<float, network_precision_t, network_precision_t>> trainer;

	cudaStream_t inference_stream;
	cudaStream_t training_stream;

};