#include "noerrors.h"

// path state flags
#define S_SPECULAR		1	// previous path vertex was specular
#define S_BOUNCED		2	// path encountered a diffuse vertex
#define S_VIASPECULAR	4	// path has seen at least one specular vertex
#define S_BOUNCEDTWICE	8	// this core will stop after two diffuse bounces
#define ENOUGH_BOUNCES	S_BOUNCED // or S_BOUNCEDTWICE

// readability defines; data layout is optimized for 128-bit accesses
#define PRIMIDX __float_as_int( hitData.z )
#define INSTANCEIDX __float_as_int( hitData.y )
#define HIT_U ((__float_as_uint( hitData.x ) & 65535) * (1.0f / 65535.0f))
#define HIT_V ((__float_as_uint( hitData.x ) >> 16) * (1.0f / 65535.0f))
#define HIT_T hitData.w

// Full sphere instead of half, different from Spherical* utility functions
LH2_DEVFUNC float2 toSphericalCoord(const float3& v)
{
  /* -PI ~ PI */
  const float theta = std::atan2(v.y, v.x);

  /* -PI/2 ~ PI/2 */
  const float phi = std::asin(clamp(v.z, -1.f, 1.f));
  return make_float2(theta, phi);
}

// Uses the naive ray tracing
__global__ void shadeTrainKernel(
    TrainPathState* trainPathStates,
	float4* hits, const uint hitsStride,
    float4* connections, const uint connectionStride,
    NRCTraceBuf* traceBuf,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int pathLength, const int w, const int h, const float spreadAngle,
	const uint pathCount
) {
    static_assert(sizeof(NRCTraceBuf) == NRC_MAX_TRAIN_PATHLENGTH * 6 * 4 * sizeof(float));

    // respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

    const float4 hitData = hits[jobIndex];
    const float3 D = trainPathStates[jobIndex].D;
    const uint pathIdx = trainPathStates[jobIndex].pathIdx;
    const uint pixelIdx = trainPathStates[jobIndex].pixelIdx;
    const uint flags = trainPathStates[jobIndex].flags;
    const uint sampleIdx = pass;

    if (PRIMIDX == NOHIT) {
        float3 tD = -worldToSky.TransformVector( D );
		float3 skyPixel = flags & S_BOUNCED ? SampleSmallSkydome( tD ) : SampleSkydome( tD );
		float3 contribution = skyPixel;
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
		FIXNAN_FLOAT3( contribution );
		
        NRCTraceBufComponent comp;
        
		return;
    }
}

#undef S_SPECULAR
#undef S_BOUNCED
#undef S_VIASPECULAR
#undef S_BOUNCEDTWICE
#undef ENOUGH_BOUNCES

#undef PRIMIDX 
#undef INSTANCEIDX
#undef HIT_U
#undef HIT_V
#undef HIT_T
