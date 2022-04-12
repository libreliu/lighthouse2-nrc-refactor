__global__ void pathStateBufferVisualizeKernel(
    const TrainPathState* trainPathStates, const uint numElements, const uint stride,
    const float4* hitData, float4* debugRT, const uint w, const uint h
) {
    const uint jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= numElements) {
        return;
    }

    // visualize pixelIdx
    const uint pixelIdx = trainPathStates[jobIndex].pixelIdx;
    const int pixelX = pixelIdx % w;
    const int pixelY = pixelIdx / w;
    const float4 hitEntry = hitData[jobIndex];
    const uint primIdx = __float_as_uint(hitEntry.z);
    const uint instIdx = __float_as_uint(hitEntry.y);
    const float tmin = hitEntry.w;

    // Draw rect
    int halfSpan = 1;
    for (int i = pixelX - halfSpan; i <= pixelX + halfSpan; i++) {
        if (i < 0 || i >= w) continue;
        for (int j = pixelY - halfSpan; j <= pixelY + halfSpan; j++) {
            if (j < 0 || j >= h) continue;
            debugRT[i + j * w] = make_float4(tmin - std::floor(tmin));
        }
    }
}

__host__ void pathStateBufferVisualize(
    const TrainPathState* trainPathStates, const uint numElements, const uint stride,
    const float4* hitData, float4* debugRT, const uint w, const uint h
) {
 	const dim3 gridDim( NEXTMULTIPLEOF( numElements, 128 ) / 128, 1 );
 	pathStateBufferVisualizeKernel <<< gridDim, 128 >>> (
        trainPathStates, numElements, stride, hitData, debugRT, w, h
    );
}

__global__ void debugRTVisualizeKernel(
    float4* debugRT, const uint w, const uint h
) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= w) || (y >= h)) return;

    debugRT[x + y * w] = make_float4(((x + y) & 0xFF) / 256.0f);
}

__host__ void debugRTVisualize(
    float4* debugRT, const uint w, const uint h
) {
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	debugRTVisualizeKernel <<< gridDim, blockDim >>> (
        debugRT, w, h
    );
}

// InputObjBasePtr, Seqno, WorldPos, Color
template <typename ...BaseArgPtrs>
using WorldPointIterator = bool (*)(uint, float3&, float3&, const BaseArgPtrs...);

template<auto iterOp, int extraSpan = 0, typename ...BaseArgPtrs>
__global__ void worldPosVisualizeKernel(
    uint numElements, float4* debugRT, const uint w, const uint h,
    // -- camera properties --
    // TODO: inv distortion
    const float3 viewP1, const float3 viewP2, const float3 viewP3,
    const float3 viewPos, const float distortion,
    const BaseArgPtrs... argPtrs
) {
    const uint jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= numElements) {
        return;
    }

    float3 worldPos, color;
    bool valid = iterOp(jobIndex, worldPos, color, argPtrs...);
    if (!valid) return;

    const float3 p1p2 = viewP2 - viewP1, p3p1 = viewP1 - viewP3;
    const float3 f = ((viewP3 - viewPos) + (viewP2 - viewPos)) * 0.5f;
    const float rl12 = 1.0f / length(p1p2), rl31 = 1.0f / length(p3p1), rlf = 1.0f / length(f);
    const float3 x = p1p2 * rl12, y = p3p1 * rl31, z = f * rlf;
    float3 dir = worldPos - viewPos;
    dir = make_float3( dot( dir, x ), dot( dir, y ), dot( dir, z ) );
    if (dir.z < 0) return;
    dir /= dir.z * rlf;
    float tx = dir.x * rl12, ty = -dir.y * rl31;
    
    // TODO: check distortion
    const int2 pixelPos = make_int2( (tx + 0.5f) * w, (ty + 0.5f) * h );
    if (pixelPos.x >= 0 && pixelPos.x < w && pixelPos.y >= 0 && pixelPos.y < h)
        debugRT[pixelPos.x + pixelPos.y * w] = make_float4(color, 0.0f);
}

template<auto iterOp, int extraSpan = 0, typename ...BaseArgPtrs>
__host__ void worldPosVisualize(
    uint numElements, float4* debugRT, const uint w, const uint h,
    const float3 viewP1, const float3 viewP2, const float3 viewP3,
    const float3 viewPos, const float distortion,
    const BaseArgPtrs... argPtrs
) {
    const dim3 gridDim( NEXTMULTIPLEOF( numElements, 128 ) / 128, 1 );
    worldPosVisualizeKernel<iterOp, extraSpan, BaseArgPtrs...> <<< gridDim, 128 >>> (
        numElements, debugRT, w, h, viewP1, viewP2, viewP3, viewPos, distortion, argPtrs...
    );
}

__device__ bool pathStateIntersectionIterator(
    uint jobIndex, float3& worldPos, float3& color,
    const TrainPathState* trainPathStates, const float4* hitData, const uint stride
) {
    const uint pixelIdx = trainPathStates[jobIndex].pixelIdx;
    const float3 rayOrigin = trainPathStates[jobIndex].O;
    const float3 rayDirection =  trainPathStates[jobIndex].D;

    const float4 hitEntry = hitData[jobIndex];
    const uint primIdx = __float_as_uint(hitEntry.z);
    const uint instIdx = __float_as_uint(hitEntry.y);
    const float tmin = hitEntry.w;

    if (!isfinite(tmin)) {
        return false;
    }

    worldPos = rayOrigin + rayDirection * tmin;
    color = make_float3(tmin - std::floor(tmin));
    return true;
}

__host__ void pathStateIntersectionVisualize(
    const TrainPathState* trainPathStates, const uint numElements, const uint stride,
    const float4* hitData, float4* debugRT, const uint w, const uint h,
    const float3 viewP1, const float3 viewP2, const float3 viewP3,
    const float3 viewPos, const float distortion
) {
    worldPosVisualize<pathStateIntersectionIterator, 0>(
        numElements, debugRT, w, h,
        viewP1, viewP2, viewP3, viewPos, distortion,
        trainPathStates, hitData, stride
    );
}