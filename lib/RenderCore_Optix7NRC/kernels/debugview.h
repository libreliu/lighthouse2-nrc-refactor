__global__ void pathStateBufferVisualizeKernel(
    const float4* pathStates, const uint numElements, const uint stride,
    float4* debugRT, const uint w, const uint h
) {
    const uint jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= numElements) {
        return;
    }

    // visualize pixelIdx
    const uint pixelIdx = __float_as_uint(pathStates[jobIndex].w) >> 6;
    const int pixelX = pixelIdx % w;
    const int pixelY = pixelIdx / w;

    // Draw rect
    int halfSpan = 2;
    for (int i = pixelX - halfSpan; i <= pixelX + halfSpan; i++) {
        if (i < 0 || i >= w) continue;
        for (int j = pixelY - halfSpan; j <= pixelY + halfSpan; j++) {
            if (j < 0 || j >= h) continue;
            debugRT[i + j * w] = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
        }
    }
}

__host__ void pathStateBufferVisualize(
    const float4* pathStates, const uint numElements, const uint stride,
    float4* debugRT, const uint w, const uint h
) {
	const dim3 gridDim( NEXTMULTIPLEOF( numElements, 128 ) / 128, 1 );
	pathStateBufferVisualizeKernel <<< gridDim, 128 >>> (
        pathStates, numElements, stride, debugRT, w, h
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