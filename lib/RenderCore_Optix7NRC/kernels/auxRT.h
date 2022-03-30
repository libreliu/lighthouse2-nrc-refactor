__global__ void writeToRenderTargetKernel(
	const float4* accumulator, const int scrwidth, const int scrheight, cudaSurfaceObject_t RTSurface
) {
	// get x and y for pixel
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= scrwidth) || (y >= scrheight)) return;
	// plot scaled pixel
	float4 value = accumulator[x + y * scrwidth];
	surf2Dwrite<float4>( value, renderTarget, x * sizeof( float4 ), y, cudaBoundaryModeClamp );
}

__host__ void writeToRenderTarget(
	const float4* accumulator, const int w, const int h, cudaSurfaceObject_t surfObj
) {
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	writeToRenderTargetKernel << < gridDim, blockDim >> > (accumulator, w, h, surfObj);
}