#include "../nrcNetTCNN.h"

__global__ void NRCTCNN_TraceBufToTrainBatch(
    const NRCTraceBuf* trainTraceBuffer,
    uint uniIdxStart,
    uint uniIdxEnd,
    uint maxPathLength,
    float* trainInputCM,
    float* trainTargetCM
) {
    static_assert(sizeof(NRCTinyCudaNN::TCNNTrainInput) == sizeof(float) * 4 * 4, "size unexpected");

    uint jobIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (jobIndex >= uniIdxEnd - uniIdxStart) {
        return;
    }
    uint pLenIdx = (jobIndex + uniIdxStart) % maxPathLength;
    uint elemIdx = (jobIndex + uniIdxStart) / maxPathLength;

    const NRCTraceBufComponent &comp = trainTraceBuffer[elemIdx].traceComponent[pLenIdx];
    NRCTinyCudaNN::TCNNTrainInput &tInput = *(NRCTinyCudaNN::TCNNTrainInput*)&trainInputCM[64 * jobIndex];
    tInput.rayIsect = comp.rayIsect;

    // TODO: add preprocess
    tInput.roughness = comp.roughness;
    tInput.rayDir = comp.rayDir;
    tInput.normalDir = comp.normalDir;
    tInput.diffuseRefl = comp.diffuseRefl;
    tInput.specularRefl = comp.specularRefl;
    tInput.dummies[0] = tInput.dummies[1] = 0.0f;

    float* tTarget = &trainTargetCM[3 * jobIndex];
    tTarget[0] = comp.lumOutput.x;
    tTarget[1] = comp.lumOutput.y;
    tTarget[2] = comp.lumOutput.z;
}

void NRCTinyCudaNN::Init(int maxTrainHistCount) {
    static_assert(sizeof(TCNNTrainInput) == sizeof(float) * 4 * 4, "size unexpected");

    nlohmann::json config = {
        {"loss", {
            {"otype", "RelativeL2Luminance"}
        }},
        {"optimizer", {
            {"otype", "Adam"},
            {"learning_rate", 1e-3},
        }},
        {"encoding", {
            {"otype", "Composite"},
            {"nested", {
                // Position, rayIsect
                // TriangleWave is way faster than OneBlob
                // (As is described in the paper)
                {
                    {"otype", "TriangleWave"}, 
                    {"n_frequencies", 12u},
                    {"n_dims_to_encode", 3u}
                },
                // Roughness & RayDir & NormalDir
                {
                    {"otype", "OneBlob"},
                    {"n_bins", 4u},
                    {"n_dims_to_encode", 5u}
                },
                // DiffuseRefl & SpecularRefl
                // (6u expected, and 2 dummy)
                {
                    {"otype", "Identity"}
                }
            }}
        }},
        {"network", {
            {"otype", "FullyFusedMLP"},
            {"activation", "ReLU"},
            {"output_activation", "None"},
            {"n_neurons", 64},
            {"n_hidden_layers", 2},
        }},
    };

    model = new tcnn::TrainableModel;
    *model = tcnn::create_from_config(16, 3, config);
}

float NRCTinyCudaNN::Train(
    CoreBuffer<NRCTraceBuf>* trainTraceBuffer,
    uint numTrainingRays,
    uint maxPathLength,  // 1, 2, 3.., NRC_MAX_TRAIN_PATHLENGTH
    uint batchSize,
    uint numTrainingSteps
) {
    if (numTrainingRays == 0 || maxPathLength == 0) {
        printf("Warning: NRCTinyCudaNN got zero numTrainingRays or zero maxPathLength\n");
        return 0.0;
    }

    uint totalElements = numTrainingRays * maxPathLength;

    if (trainBatchInputCM == nullptr ||
        trainBatchInputCM->cols() < batchSize) {
        if (trainBatchInputCM) delete trainBatchInputCM;
        trainBatchInputCM = new tcnn::GPUMatrix<float>(64, batchSize);
    }

    if (trainBatchTargetCM == nullptr ||
        trainBatchTargetCM->cols() < batchSize) {
        if (trainBatchTargetCM) delete trainBatchTargetCM;
        trainBatchTargetCM = new tcnn::GPUMatrix<float>(3, batchSize);
    }

    int numBatches = totalElements / batchSize;
    if (numBatches * batchSize < totalElements) {
        numBatches++;
    }

    float avgLoss = 0.0f;

    for (uint i = 0; i < numTrainingSteps; i++) {
        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            const dim3 gridDim( NEXTMULTIPLEOF( totalElements, 32 ) / 32, 1 );
            NRCTCNN_TraceBufToTrainBatch<<<gridDim.x, 32>>>(
                trainTraceBuffer->DevPtr(),
                batchIdx * batchSize,
                batchIdx == numBatches - 1 ? totalElements : (batchIdx + 1) * batchSize,
                maxPathLength,
                trainBatchInputCM->data(),
                trainBatchTargetCM->data()
            );

            auto ctx = model->trainer->training_step(*trainBatchInputCM, *trainBatchTargetCM);
            avgLoss += model->trainer->loss(*ctx) * (1.0f / numTrainingSteps);
        }
    }

    return avgLoss;
}

void NRCTinyCudaNN::Inference(
    CoreBuffer<NRCNetInferenceInput>* infInputBuffer,
    uint numInfRays,
    CoreBuffer<NRCNetInferenceOutput>* infOutputBuffer
) {
    // NOTE: THIS ASSERTS DUMMY ENTRY PRESENT, OR conversion shall be applied
    tcnn::GPUMatrix<float> infInputCM((float*)infInputBuffer->DevPtr(), 64u, numInfRays);
    tcnn::GPUMatrix<float> infOutputCM((float*)infOutputBuffer->DevPtr(), 3u, numInfRays);

    model->network->inference(infInputCM, infOutputCM);
}

void NRCTinyCudaNN::Destroy() {
    // TODO: perform stream sync
    if (trainBatchInputCM) delete trainBatchInputCM;
    if (trainBatchTargetCM) delete trainBatchTargetCM;
    if (model) delete model;
}