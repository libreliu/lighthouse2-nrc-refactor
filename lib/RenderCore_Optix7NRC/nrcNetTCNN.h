#include "core_settings.h"

#include <memory>
#include <vector>

#include "shared_host_code/cudatools.h"

#ifdef __CUDACC__
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#else

namespace tcnn {
class TrainableModel;
enum class MatrixLayout {
	RowMajor = 0,
	SoA = 0, // For data matrices TCNN's convention is RowMajor == SoA (struct of arrays)
	ColumnMajor = 1,
	AoS = 1,
};

template<typename T>
class GPUMatrixDynamic;

template<typename T, MatrixLayout _layout = MatrixLayout::ColumnMajor>
class GPUMatrix;

}
#endif

class NRCTinyCudaNN {
public:
  void Init(int maxTrainHistCount);
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
  tcnn::TrainableModel* model;

  // Column major by default, alter might cause tiny-cuda-nn deficiency
  // Reason for not using unique_ptr: this class shall be called from hostcc compiled host code
  // and on the hostcc side, including <tiny-cuda-nn/gpu_matrix.h> will break since it requires
  // nvcc to compile. Forward declaration is tried, but std::unique_ptr requires complete type
  // to perform delete, hence will not work.
  // All the creation and destroy belongs to the coda impl.h as a final design decision.
  tcnn::GPUMatrix<float> *trainBatchInputCM;
  tcnn::GPUMatrix<float> *trainBatchTargetCM;
};