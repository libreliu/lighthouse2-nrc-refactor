#include "nrcNetKNN.h"
#include <ANN/ANN.h>

static const int nrcKNN_TrainSampleSize = 100;
static const int nrcKNN_K = 1;

void NRCKNN::Init() {
    static_assert(sizeof(NRCKNNInput) == sizeof(float) * 16);

    // Expect ANNcoord to be float, patch this if not
    static_assert(sizeof(ANNcoord) == sizeof(float));

    // *VERY* important, prevents reallocation (and hence the invalidation of annDatapts)
    trainedInputs.reserve(nrcKNN_TrainSampleSize);
    trainedTargets.reserve(nrcKNN_TrainSampleSize);
    annDataPts.reserve(nrcKNN_TrainSampleSize);
}

// Only primary is implemented
uint NRCKNN::Preprocess(
    CoreBuffer<NRCTraceBuf>* trainTraceBuffer,
    uint numTrainingRays,
    uint maxPathLength  // 1, 2, 3.., NRC_MAX_TRAIN_PATHLENGTH
) {
    assert(maxPathLength == 1);
    trainTraceBuffer->CopyToHost();
    NRCTraceBuf *tBuf = trainTraceBuffer->HostPtr();

    for (uint i = 0; i < numTrainingRays; i++) {
        NRCKNNInput kInput;
        kInput.rayIsect = tBuf[i].traceComponent[0].rayIsect;
        kInput.roughness = tBuf[i].traceComponent[0].roughness;
        kInput.rayDir = tBuf[i].traceComponent[0].rayDir;
        kInput.normalDir = tBuf[i].traceComponent[0].normalDir;
        kInput.diffuseRefl = tBuf[i].traceComponent[0].diffuseRefl;
        kInput.specularRefl = tBuf[i].traceComponent[0].specularRefl;
        kInput.dummies[0] = kInput.dummies[1] = 0;

        NRCKNNOutput kOutput;
        kOutput.lumOutput = tBuf[i].traceComponent[0].lumOutput;

        if (trainedInputs.size() == nrcKNN_TrainSampleSize) {
            trainedInputs[offset] = kInput;
            trainedTargets[offset] = kOutput;
            annDataPts[offset] = (float*)&trainedInputs[offset];
            offset += 1;
            if (offset == nrcKNN_TrainSampleSize) {
                offset = 0;
            }
        } else {
            trainedInputs.push_back(kInput);
            trainedTargets.push_back(kOutput);
            annDataPts.push_back((float*)&trainedInputs.back());
        }
    }

    if (kdTree) {
        delete kdTree;
        kdTree = nullptr;
    }

    return numTrainingRays;
}

float NRCKNN::Train(
    uint batchSize,
    uint numTrainingSteps
) {
    // no-op
    return 1.0f;
}

void NRCKNN::Inference(
    CoreBuffer<NRCNetInferenceInput>* infInputBuffer,
    uint numInfRays,
    CoreBuffer<NRCNetInferenceOutput>* infOutputBuffer
) {
    if (trainedInputs.size() == 0) {
        printf("[NRC WARN] KNN have no trained samples to inference\n");
        return;
    }

    infInputBuffer->CopyToHost();
    if (!kdTree) {
        kdTree = new ANNkd_tree(
            annDataPts.data(),
            annDataPts.size(),
            16
        );
    }

    if (infOutputBuffer->HostPtr() == nullptr) {
        infOutputBuffer->CopyToHost();
    }

    NRCNetInferenceOutput *iOut = infOutputBuffer->HostPtr();
    NRCNetInferenceInput *iIn = infInputBuffer->HostPtr();

    ANNidx nnIdx[nrcKNN_K];
    ANNdist nnDist[nrcKNN_K];

    for (uint i = 0; i < numInfRays; i++) {
        kdTree->annkSearch((float *)&iIn[i], 1, nnIdx, nnDist, 0);
        iOut[i].lumOutput = trainedTargets[nnIdx[0]].lumOutput;
    }

    infOutputBuffer->CopyToDevice();
}

void NRCKNN::Reset(int mode) {
    offset = 0;
    if (kdTree) {
        delete kdTree;
        kdTree = nullptr;
    }

    trainedInputs.clear();
    trainedTargets.clear();
    annDataPts.clear();
    trainedInputs.reserve(nrcKNN_TrainSampleSize);
    trainedTargets.reserve(nrcKNN_TrainSampleSize);
    annDataPts.reserve(nrcKNN_TrainSampleSize);
}

void NRCKNN::Destroy() {
    offset = 0;
    if (kdTree) {
        delete kdTree;
        kdTree = nullptr;
    }
    trainedInputs.clear();
    trainedTargets.clear();
    annDataPts.clear();
}