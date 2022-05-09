#include "core_settings.h"

#include <memory>
#include <vector>

#include "shared_host_code/cudatools.h"

// TODO: implement me
class NRCTorch {
public:
  void Init();
  float Train(
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
  void Destroy();

};