__global__ void pathStateBufferVisualizeKernel(
    const float4* pathStates, const uint numElements, const uint stride,
    float4* debugRT, const uint w, const uint h
) {
    const uint jobIndex = blockIdx.x + blockIdx.y * blockDim.x;
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
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	pathStateBufferVisualizeKernel <<< gridDim, blockDim >>> (
        pathStates, numElements, stride, debugRT, w, h
    );
}