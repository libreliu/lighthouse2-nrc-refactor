#pragma once

#define NRC_SKIP_RENDERCORE_INCLUDE
#include "core_settings.h"
#undef NRC_SKIP_RENDERCORE_INCLUDE

#include <memory>
#include <vector>

#include "shared_host_code/cudatools.h"

class ANNkd_tree;

class NRCKNN {
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

private:
  // 16-dimentional float
  struct alignas(sizeof(float) * 4) NRCKNNInput {
    float3 rayIsect;
    float roughness;
    float2 rayDir;
    float2 normalDir;
    float3 diffuseRefl;
    float3 specularRefl;
    float dummies[2];
  };

  struct NRCKNNOutput {
    float3 lumOutput;
  };

  ANNkd_tree *kdTree;
  std::vector<NRCKNNInput> trainedInputs;
  std::vector<NRCKNNOutput> trainedTargets;
  std::vector<float*> annDataPts;
  uint offset = 0;
};