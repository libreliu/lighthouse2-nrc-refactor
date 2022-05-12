#include "core_settings.h"

#include <memory>
#include <vector>

#include "shared_host_code/cudatools.h"

class NRCANN {
public:
  void Init();
  uint Preprocess(
    CoreBuffer<NRCTraceBuf>* trainTraceBuffer,
    uint numTrainingRays,
    uint maxPathLength  // 1, 2, 3.., NRC_MAX_TRAIN_PATHLENGTH
  );
  float Train(
    uint batchSize,
    uint numTrainingSteps
  );
  void Inference(
    CoreBuffer<NRCNetInferenceInput>* infInputBuffer,
    uint numInfRays,
    CoreBuffer<NRCNetInferenceOutput>* infOutputBuffer
  );

  // Reset network weight
  // NRCNET_RESETMODE_UNIFORM
  void Reset(int mode);
  void Destroy();
};