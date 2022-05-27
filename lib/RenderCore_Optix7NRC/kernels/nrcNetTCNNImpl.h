#include "../nrcNetTCNN.h"
#include <cassert>
#include <random>

// Sif: Size in float
const uint tcnnInputSif = sizeof(NRCTinyCudaNN::TCNNTrainInput) / sizeof(float);
const uint tcnnTargetSif = 3;

__global__ void NRCTCNN_TraceBufToTrainBuffer(
    const NRCTraceBuf* trainTraceBuffer,
    uint uniIdxEnd,
    uint maxPathLength,
    float* trainInputCM,
    float* trainTargetCM,
    uint *numPreparedRays
) {
    static_assert(sizeof(NRCTinyCudaNN::TCNNTrainInput) == sizeof(float) * 4 * 4, "size unexpected");

    uint jobIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (jobIndex >= uniIdxEnd) {
        return;
    }
    uint pLenIdx = jobIndex % maxPathLength;
    uint elemIdx = jobIndex / maxPathLength;

    const NRCTraceBufComponent &comp = trainTraceBuffer[elemIdx].traceComponent[pLenIdx];
    
    // Valid component have non-zero traceFlag
    if (comp.traceFlags == 0) {
        return;
    }

    const uint slotIdx = atomicAdd(numPreparedRays, 1);
    NRCTinyCudaNN::TCNNTrainInput &tInput = *(NRCTinyCudaNN::TCNNTrainInput*)(&trainInputCM[tcnnInputSif * slotIdx]);

    // -- fill train input struct --
    tInput.rayIsect = comp.rayIsect;

    // TODO: add preprocess & discard non-hit
    tInput.roughness = comp.roughness;
    tInput.rayDir = comp.rayDir;
    tInput.normalDir = comp.normalDir;
    tInput.diffuseRefl = comp.diffuseRefl;
    tInput.specularRefl = comp.specularRefl;
    tInput.dummies[0] = tInput.dummies[1] = 0.0f;

    // tInput.roughness = 0.0f;
    // tInput.rayDir = make_float2(0.0f);
    // tInput.normalDir = make_float2(0.0f);
    // tInput.diffuseRefl = make_float3(0.0f);
    // tInput.specularRefl = make_float3(0.0f);
    // tInput.dummies[0] = tInput.dummies[1] = 0.0f;

    float* tTarget = &trainTargetCM[3 * slotIdx];
    tTarget[0] = comp.lumOutput.x;
    tTarget[1] = comp.lumOutput.y;
    tTarget[2] = comp.lumOutput.z;
}

__global__ void NRCTCNN_GenTrainBatchFromTrainInput(
    float* trainInputCM,
    float* trainTargetCM,
    uint startIdx,
    uint batchSize,
    float* trainInputBatchCM,
    float* trainTargetBatchCM
) {
    uint jobIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (jobIndex >= batchSize) {
        return;
    }

    for (int i = 0; i < tcnnInputSif; i++) {
        trainInputBatchCM[jobIndex * tcnnInputSif + i] = trainInputCM[(startIdx + jobIndex) * tcnnInputSif + i];
    }

    for (int i = 0; i < tcnnTargetSif; i++) {
        trainTargetBatchCM[jobIndex * tcnnTargetSif + i] = trainTargetCM[(startIdx + jobIndex) * tcnnTargetSif + i];
    }
}

void NRCTinyCudaNN::Init() {
    static_assert(sizeof(TCNNTrainInput) == sizeof(float) * 4 * 4, "size unexpected");
    static_assert(sizeof(NRCNetInferenceOutput) == sizeof(float) * 3, "size unexpected");

    assert(!initialized);

    nlohmann::json config = {
        {"loss", {
            //{"otype", "L2"},
            //{"otype", "RelativeL2"}
            {"otype", "RelativeL2Luminance"}
        }},
        {"optimizer", {
            // Moments might be nan, difficult to recover
            {"otype", "Adam"},
            // {"otype", "SGD"},
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
        //{"encoding", {
        //    {"otype", "Identity"}
        //}},
        {"network", {
            // {"otype", "FullyFusedMLP"},
            {"otype", "CutlassMLP"},
            {"activation", "ReLU"},
            {"output_activation", "None"},
            {"n_neurons", numNeurons},
            {"n_hidden_layers", numHiddenLayers},
        }},
    };

    model = new tcnn::TrainableModel;
    *model = tcnn::create_from_config(16, 3, config);
    
    numPreparedRays = new CoreBuffer<uint>(1, ON_DEVICE | ON_HOST);
    initialized = true;
}

uint NRCTinyCudaNN::Preprocess(
    CoreBuffer<NRCTraceBuf>* trainTraceBuffer,
    uint numTrainingRays,
    uint maxPathLength
) {
    assert(initialized && !prepared);
    numPreparedRays->Clear(ON_DEVICE | ON_HOST);

    if (numTrainingRays == 0 || maxPathLength == 0) {
        printf("Warning: NRCTinyCudaNN got zero numTrainingRays or zero maxPathLength\n");
        return 0.0;
    }

    uint totalElements = numTrainingRays * maxPathLength;
    if (trainInputCM == nullptr ||
        trainInputCM->cols() < totalElements) {
        if (trainInputCM) delete trainInputCM;
        trainInputCM = new tcnn::GPUMatrix<float>(
            tcnnInputSif,
            totalElements
        );
    }

    if (trainTargetCM == nullptr ||
        trainTargetCM->cols() < totalElements) {
        if (trainTargetCM) delete trainTargetCM;
        trainTargetCM = new tcnn::GPUMatrix<float>(
            tcnnTargetSif,
            totalElements
        );
    }

    const dim3 gridDim( NEXTMULTIPLEOF( totalElements, 32 ) / 32, 1 );
    NRCTCNN_TraceBufToTrainBuffer<<<gridDim.x, 32>>>(
        trainTraceBuffer->DevPtr(),
        numTrainingRays,
        maxPathLength,
        trainInputCM->data(),
        trainTargetCM->data(),
        numPreparedRays->DevPtr()
    );

    numPreparedRays->CopyToHost();
    prepared = true;

    return *numPreparedRays->HostPtr();
}

float NRCTinyCudaNN::Train(
    uint batchSize,
    uint numTrainingSteps
) {
    assert(initialized && prepared);
    assert(batchSize >= 256 && batchSize % 16 == 0);

    if (trainBatchInputCM == nullptr ||
        trainBatchInputCM->cols() < batchSize) {
        if (trainBatchInputCM) delete trainBatchInputCM;
        trainBatchInputCM = new tcnn::GPUMatrix<float>(
            tcnnInputSif,
            batchSize
        );
    }

    if (trainBatchTargetCM == nullptr ||
        trainBatchTargetCM->cols() < batchSize) {
        if (trainBatchTargetCM) delete trainBatchTargetCM;
        trainBatchTargetCM = new tcnn::GPUMatrix<float>(tcnnTargetSif, batchSize);
    }

    int numRaysToTrain = *numPreparedRays->HostPtr();
    int numBatches = numRaysToTrain / batchSize;
    if (numBatches == 0) {
        printf("[NRC_WARN] no batches are being trained\n");
        prepared = false;
        return 0.0f;
    }

    float avgLoss = 0.0f;

    for (uint i = 0; i < numTrainingSteps; i++) {
        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            const dim3 gridDim( NEXTMULTIPLEOF( batchSize, 32 ) / 32, 1 );
            NRCTCNN_GenTrainBatchFromTrainInput<<<gridDim.x, 32>>>(
                trainInputCM->data(),
                trainTargetCM->data(),
                batchIdx * batchSize,
                batchSize,
                trainBatchInputCM->data(),
                trainBatchTargetCM->data()
            );

            // visualize
            /*std::vector<float> vVis(batchSize * tcnnInputSif);
            CHK_CUDA(cudaMemcpy(vVis.data(), trainBatchInputCM->data(), batchSize * tcnnInputSif, cudaMemcpyDeviceToHost));

            for (int row = 0; row < tcnnInputSif; row++) {
                for (int col = 0; col < batchSize; col++) {
                    printf("%f ", vVis[col * tcnnInputSif + row]);
                }
                printf("\n");
            }

            printf("==============\n");
            std::vector<float> vVis2(batchSize * tcnnTargetSif);
            CHK_CUDA(cudaMemcpy(vVis2.data(), trainBatchInputCM->data(), batchSize * tcnnTargetSif, cudaMemcpyDeviceToHost));

            for (int row = 0; row < tcnnTargetSif; row++) {
                for (int col = 0; col < batchSize; col++) {
                    printf("%f ", vVis2[col * tcnnTargetSif + row]);
                }
                printf("\n");
            }*/

            auto ctx = model->trainer->training_step(*trainBatchInputCM, *trainBatchTargetCM);
            avgLoss += model->trainer->loss(*ctx) * (1.0f / numTrainingSteps);
        }
    }

    CHK_CUDA(cudaDeviceSynchronize());

    prepared = false;
    return avgLoss;
}

void NRCTinyCudaNN::Inference(
    CoreBuffer<NRCNetInferenceInput>* infInputBuffer,
    uint numInfRays,
    CoreBuffer<NRCNetInferenceOutput>* infOutputBuffer
) {
    assert(initialized);

    // NOTE: THIS ASSERTS DUMMY ENTRY PRESENT, OR conversion shall be applied
    tcnn::GPUMatrix<float> infInputCM((float*)infInputBuffer->DevPtr(), 16u, numInfRays);
    tcnn::GPUMatrix<float> infOutputCM((float*)infOutputBuffer->DevPtr(), 3u, numInfRays);

    model->network->inference(infInputCM, infOutputCM);
}

void NRCTinyCudaNN::Reset(int mode) {
    // output 3 is padded to 16
    size_t numParams = numNeurons * numNeurons * numHiddenLayers + numNeurons * 16;
    std::vector<float> paramsCPU(numParams);

    if (mode == NRCNET_RESETMODE_UNIFORM) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(-1, 1);

        for (int i = 0; i < numParams; i++) {
            paramsCPU[i] = distrib(gen);
        }
    }

    CHK_CUDA(cudaDeviceSynchronize());

    model->trainer->set_params_full_precision(paramsCPU.data(), numParams);
    
    // TODO: reset optimizer
    //model->optimizer->
}

void NRCTinyCudaNN::Destroy() {
    // TODO: perform stream sync
    if (trainBatchInputCM) delete trainBatchInputCM;
    if (trainBatchTargetCM) delete trainBatchTargetCM;
    if (model) delete model;
    if (numPreparedRays) delete numPreparedRays;
    tcnn::cpp::free_temporary_memory();
}