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
// NOTE: our pathLength starts from 0, as is opposed to shadeKernel
template <bool trainSkybox>
__global__ void shadeTrainKernel(
    TrainPathState* trainPathStates, const uint pathCount,
    TrainPathState* nextTrainPathStates,
	float4* hits,
    TrainConnectionState* connections,
    NRCTraceBuf* traceBuf,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int pathLength, const int w, const int h, const float spreadAngle
) {
    static_assert(sizeof(NRCTraceBuf) == NRC_MAX_TRAIN_PATHLENGTH * 6 * 4 * sizeof(float));

    // respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

    const float4 hitData = hits[jobIndex];
    hits[jobIndex].z = __int_as_float(-1);  // reset for next query
    const TrainPathState& tpState = trainPathStates[jobIndex];
    const float3 O = tpState.O;
    uint flags = tpState.flags;
    const float3 D = tpState.D;
    const uint pathIdx = tpState.pathIdx;
    const float3 throughput = tpState.throughput;
    const uint pixelIdx = tpState.pixelIdx;
    const uint sampleIdx = pass;

    if (PRIMIDX == NOHIT) {
        float3 tD = -worldToSky.TransformVector( D );
		float3 skyPixel = flags & S_BOUNCED ? SampleSmallSkydome( tD ) : SampleSkydome( tD );
		
        if (trainSkybox) {
            NRCTraceBufComponent comp;
            comp.rayIsect = NRC_INVALID_FLOAT3;
            comp.roughness = NRC_INVALID_FLOAT;
            comp.rayDir = toSphericalCoord(D);
            comp.normalDir = NRC_INVALID_FLOAT2;
            comp.diffuseRefl = NRC_INVALID_FLOAT3;
            comp.specularRefl = NRC_INVALID_FLOAT3;
            comp.lumOutput = skyPixel;
            comp.throughput = make_float3(1.0f);
            comp.pixelIdx = pixelIdx;
            comp.pathIdx = pathIdx;
            comp.traceFlags = NRC_TRACEFLAG_HIT_SKYBOX;
            comp.rrProb = 1;

            traceBuf[pathIdx].traceComponent[pathLength] = comp;
        }
		return;
    }

    const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

    // get shadingData and normals
	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = O + HIT_T * D;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( D, HIT_U, HIT_V, coneWidth, instanceTriangles[PRIMIDX], INSTANCEIDX, shadingData, N, iN, fN, T );
	uint seed = WangHash( pathIdx * 17 + R0 /* well-seeded xor32 is all you need */ );

    // Initialize basic information
    // Remain to be filled: comp.lumOutput comp.throughput comp.traceFlags (initialized)
    NRCTraceBufComponent comp;
    comp.rayIsect = I;
    comp.roughness = ROUGHNESS;
    comp.rayDir = toSphericalCoord(D);
    comp.normalDir = toSphericalCoord(fN);
    comp.diffuseRefl = shadingData.color;
    comp.specularRefl = shadingData.color;  // TODO: figure out
    comp.pixelIdx = pixelIdx;
    comp.pathIdx = pathIdx;
    comp.traceFlags = 0;                    // default value
    comp.lumOutput = make_float3(0.0f);     // default value, useful for secondary ray hit program
    comp.rrProb = 1;                        // default value

    // TODO: handle translucent material

    if (shadingData.IsEmissive()) {
        const float DdotNL = -dot(D, N);
        comp.throughput = make_float3(1.0f);
        comp.lumOutput = make_float3(0);

        float3 contribution = make_float3( 0 ); // initialization required.
		if (DdotNL > 0 /* lights are not double sided */) {
            if (pathLength == 0 || (flags & S_SPECULAR) > 0) {
                comp.lumOutput = throughput * shadingData.color;
            }
            comp.traceFlags |= NRC_TRACEFLAG_HIT_LIGHT_FRONT;
		} else {
            comp.traceFlags |= NRC_TRACEFLAG_HIT_LIGHT_BACK;
        }

        traceBuf[pathIdx].traceComponent[pathLength] = comp;
		return;
    }

    // detect specular surfaces
	if (ROUGHNESS <= 0.001f || TRANSMISSION > 0.5f) {
        /* detect pure speculars; skip NEE for these */
        flags |= S_SPECULAR;
    } else {
        flags &= ~S_SPECULAR;
    }

    
	// normal alignment for backfacing polygons
	const float faceDir = (dot( D, N ) > 0) ? -1 : 1;
	if (faceDir == 1) shadingData.transmittance = make_float3( 0 );

	// prepare random numbers
	float4 r4;
	if (sampleIdx < 64) {
		const uint x = ((pathIdx % w) + (shift & 127)) & 127;
		const uint y = ((pathIdx / w) + (shift >> 24)) & 127;
		r4 = blueNoiseSampler4( blueNoise, x, y, sampleIdx, 4 * pathLength );
	} else {
		r4.x = RandomFloat( seed ), r4.y = RandomFloat( seed );
		r4.z = RandomFloat( seed ), r4.w = RandomFloat( seed );
	}

	// next event estimation: connect eye path to light
    // NOTE: lumOutput (if hit) are updated by optix codes
	if ((flags & S_SPECULAR) == 0 && connections != 0) {
        float pickProb, lightPdf = 0;
		float3 lightColor, L = RandomPointOnLight( r4.x, r4.y, I, fN * faceDir, pickProb, lightPdf, lightColor ) - I;
		const float dist = length( L );
		L *= 1.0f / dist;
		const float NdotL = dot( L, fN * faceDir );
		if (NdotL > 0 && lightPdf > 0)
		{
            comp.traceFlags |= NRC_TRACEFLAG_NEE_EMIT;

			float bsdfPdf;
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN /* * faceDir */, T, D * -1.0f, L, bsdfPdf );
			{
				// add fire-and-forget shadow ray to the connections buffer
				const uint shadowRayIdx = atomicAdd( &counters->shadowRays, 1 ); // compaction
				TrainConnectionState tcState;
                tcState.O = SafeOrigin( I, L, N, geometryEpsilon );
                tcState.pathIdx = pathIdx;
                tcState.D = L;
                tcState.dist = dist - 2 * geometryEpsilon;
                tcState.directLum = sampledBSDF * lightColor * (NdotL / (pickProb * lightPdf));
                tcState.pixelIdx = pixelIdx;
                
                connections[shadowRayIdx] = tcState;
			}
		}
    }

    // cap at maxium path length
	if (pathLength == NRC_MAX_TRAIN_PATHLENGTH - 1) {
        comp.traceFlags |= NRC_TRACEFLAG_PATHLEN_TRUNCTUATE;
        traceBuf[pathIdx].traceComponent[pathLength] = comp;
        return;
    }

    // evaluate bsdf to obtain direction for next path segment
	float3 R;
	float newBsdfPdf;
	bool specular = false;
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, D * -1.0f, HIT_T, r4.z, r4.w, RandomFloat( seed ), R, newBsdfPdf, specular );
    if (specular) flags |= S_SPECULAR;
    
    // premature ending conditions
	if (newBsdfPdf < EPSILON || isnan( newBsdfPdf )) {
        comp.traceFlags |= NRC_TRACEFLAG_BSDF_TRUNCTUATE;
        traceBuf[pathIdx].traceComponent[pathLength] = comp;
        return;
    }
	

	// russian roulette
    const float p = ((flags & S_SPECULAR) || ((flags & S_BOUNCED) == 0)) ? 1 : SurvivalProbability( bsdf );
	if (p < RandomFloat( seed )) {
        comp.traceFlags |= NRC_TRACEFLAG_RR_TRUNCTUATE;
        comp.rrProb = p;
        traceBuf[pathIdx].traceComponent[pathLength] = comp;
        return;
    }

    const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
	if (!(flags & S_SPECULAR)) {
        flags |= flags & S_BOUNCED ? S_BOUNCEDTWICE : S_BOUNCED;
    } else {
        flags |= S_VIASPECULAR;
    }

    TrainPathState nextTp;
    nextTp.O = SafeOrigin(I, R, N, geometryEpsilon);
    nextTp.flags = flags;
    nextTp.D = R;
    nextTp.pathIdx = pathIdx;
    nextTp.throughput = (1 / p) * bsdf * abs(dot(fN, R)) / newBsdfPdf;
    nextTp.pixelIdx = pixelIdx;

    nextTrainPathStates[extensionRayIdx] = nextTp;

    comp.traceFlags |= NRC_TRACEFLAG_NEXT_BOUNCE;
    comp.throughput = nextTp.throughput;
    traceBuf[pathIdx].traceComponent[pathLength] = comp;
}

__host__ void shadeTrain(
    TrainPathState* trainPathStates, const uint pathCount,
    TrainPathState* nextTrainPathStates,
	float4* hits,
    TrainConnectionState* connections,
    NRCTraceBuf* traceBuf,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int pathLength, const int w, const int h, const float spreadAngle
) {
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 );
	shadeTrainKernel<true> <<<gridDim.x, 128 >>> (
        trainPathStates, pathCount,
        nextTrainPathStates,
        hits,
        connections,
        traceBuf,
        R0, shift, blueNoise, pass,
        pathLength, w, h, spreadAngle
    );
}

__global__ void shadeNRCOnlyKernel(
    float4* accumulator, 
    InferencePathState* pathStates, const uint pathCount,
    InferencePathState* nextPathStates,
    float4* hits, 
    InferenceConnState* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
    const int pathLength, const int w, const int h, const float spreadAngle,
    int* numRaysToBeInferenced,
    NRCNetInferenceInput* inferenceInput,
    uint* inferencePixelIndices,
    float3* inferencePixelContribs
) {
    // respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

    const float4 hitData = hits[jobIndex];
    hits[jobIndex].z = __int_as_float(-1);  // reset for next query

    const InferencePathState& ipState = pathStates[jobIndex];
    const float3 O = ipState.O;
    uint flags = ipState.flags;
    const float3 D = ipState.D;
    const uint pathIdx = ipState.pathIdx;
    const float3 throughput = pathLength == 0 ? make_float3(1.0f) : ipState.throughput;
    const uint pixelIdx = ipState.pixelIdx;
    const uint sampleIdx = pass;

    if (PRIMIDX == NOHIT) {
        float3 tD = -worldToSky.TransformVector( D );
		float3 skyPixel = flags & S_BOUNCED ? SampleSmallSkydome( tD ) : SampleSkydome( tD );
		float3 contribution = throughput * skyPixel;
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
        FIXNAN_FLOAT3( contribution );

        accumulator[pixelIdx] += make_float4( contribution, 0 );
		return;
    }

    const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

    // Look up the net
	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = O + HIT_T * D;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( D, HIT_U, HIT_V, coneWidth, instanceTriangles[PRIMIDX], INSTANCEIDX, shadingData, N, iN, fN, T );
	uint seed = WangHash( pathIdx * 17 + R0 /* well-seeded xor32 is all you need */ );

    const uint infIdx = atomicAdd(numRaysToBeInferenced, 1);

    NRCNetInferenceInput &iInput = inferenceInput[infIdx];
    
    iInput.rayIsect = I;
    iInput.roughness = ROUGHNESS;
    iInput.rayDir = toSphericalCoord(D);
    iInput.normalDir = toSphericalCoord(fN);
    iInput.diffuseRefl = shadingData.color;
    iInput.specularRefl = shadingData.color;  // TODO: figure out
    iInput.dummies[0] = iInput.dummies[1] = 0.0f;

    // iInput.roughness = 0.0f;
    // iInput.rayDir = make_float2(0.0f);
    // iInput.normalDir = make_float2(0.0f);
    // iInput.diffuseRefl = make_float3(0.0f);
    // iInput.specularRefl = make_float3(0.0f);
    // iInput.dummies[0] = iInput.dummies[1] = 0.0f;

    inferencePixelIndices[infIdx] = pixelIdx;
    inferencePixelContribs[infIdx] = make_float3(1.0f, 1.0f, 1.0f);
}

__host__ void shadeNRCOnly(
    float4* accumulator, 
    InferencePathState* pathStates, const uint pathCount,
    InferencePathState* nextPathStates,
    float4* hits, 
    InferenceConnState* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
    const int pathLength, const int w, const int h, const float spreadAngle,
    int* numRaysToBeInferenced,
    NRCNetInferenceInput* inferenceInput,
    uint* inferencePixelIndices,
    float3* inferencePixelContribs
) {
    const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 );
    shadeNRCOnlyKernel <<<gridDim.x, 128>>> (
        accumulator,
        pathStates, pathCount,
        nextPathStates,
        hits,
        connections,
        R0, shift, blueNoise, pass,
        pathLength, w, h, spreadAngle,
        numRaysToBeInferenced,
        inferenceInput,
        inferencePixelIndices,
        inferencePixelContribs
    );
}

__global__ void shadeNRCKernel(
    float4* accumulator, 
    InferencePathState* pathStates, const uint pathCount,
    InferencePathState* nextPathStates,
    float4* hits, 
    InferenceConnState* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
    const int pathLength, const int w, const int h, const float spreadAngle,
    int* numRaysToBeInferenced,
    NRCNetInferenceInput* inferenceInput,
    uint* inferencePixelIndices,
    float3* inferencePixelContribs
) {
    // respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

    const float4 hitData = hits[jobIndex];
    hits[jobIndex].z = __int_as_float(-1);  // reset for next query

    const InferencePathState& ipState = pathStates[jobIndex];
    const float3 O = ipState.O;
    uint flags = ipState.flags;
    const float3 D = ipState.D;
    const uint pathIdx = ipState.pathIdx;
    const float3 throughput = pathLength == 0 ? make_float3(1.0f) : ipState.throughput;
    const uint pixelIdx = ipState.pixelIdx;
    const uint sampleIdx = pass;

    if (PRIMIDX == NOHIT) {
        float3 tD = -worldToSky.TransformVector( D );
		float3 skyPixel = flags & S_BOUNCED ? SampleSmallSkydome( tD ) : SampleSkydome( tD );
		float3 contribution = throughput * skyPixel;
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
        FIXNAN_FLOAT3( contribution );

        accumulator[pixelIdx] += make_float4( contribution, 0 );
		return;
    }

    const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

    // get shadingData and normals
	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = O + HIT_T * D;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( D, HIT_U, HIT_V, coneWidth, instanceTriangles[PRIMIDX], INSTANCEIDX, shadingData, N, iN, fN, T );
	uint seed = WangHash( pathIdx * 17 + R0 /* well-seeded xor32 is all you need */ );

    // TODO: handle translucent material
    if (shadingData.IsEmissive()) {
        const float DdotNL = -dot(D, N);
        float3 contribution = make_float3(0);
        if (DdotNL > 0) {
            if (pathLength == 0 || (ipState.flags & S_SPECULAR) > 0) {
                contribution = throughput * shadingData.color;
            }
        }

        CLAMPINTENSITY;
		FIXNAN_FLOAT3( contribution );
		accumulator[pixelIdx] += make_float4( contribution, 0 );
		return;
    }

    // detect specular surfaces
	if (ROUGHNESS <= 0.001f || TRANSMISSION > 0.5f) {
        /* detect pure speculars; skip NEE for these */
        flags |= S_SPECULAR;
    } else {
        flags &= ~S_SPECULAR;
    }

    const float faceDir = (dot( D, N ) > 0) ? -1 : 1;
	if (faceDir == 1) shadingData.transmittance = make_float3( 0 );

    // prepare random numbers
	float4 r4;
	if (sampleIdx < 64) {
		const uint x = ((pathIdx % w) + (shift & 127)) & 127;
		const uint y = ((pathIdx / w) + (shift >> 24)) & 127;
		r4 = blueNoiseSampler4( blueNoise, x, y, sampleIdx, 4 * pathLength );
	} else {
		r4.x = RandomFloat( seed ), r4.y = RandomFloat( seed );
		r4.z = RandomFloat( seed ), r4.w = RandomFloat( seed );
	}

	// next event estimation: connect eye path to light
    // NOTE: lumOutput (if hit) are updated by optix codes
	if ((flags & S_SPECULAR) == 0 && connections != 0) {
        float pickProb, lightPdf = 0;
		float3 lightColor, L = RandomPointOnLight( r4.x, r4.y, I, fN * faceDir, pickProb, lightPdf, lightColor ) - I;
		const float dist = length( L );
		L *= 1.0f / dist;
		const float NdotL = dot( L, fN * faceDir );
		if (NdotL > 0 && lightPdf > 0)
		{
			float bsdfPdf;
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN /* * faceDir */, T, D * -1.0f, L, bsdfPdf );
			{
				// add fire-and-forget shadow ray to the connections buffer
                float3 contribution = throughput * sampledBSDF * lightColor * (NdotL / (pickProb * lightPdf));
                FIXNAN_FLOAT3(contribution);
                CLAMPINTENSITY;

				const uint shadowRayIdx = atomicAdd( &counters->shadowRays, 1 ); // compaction
				InferenceConnState icState;
                icState.O = SafeOrigin( I, L, N, geometryEpsilon );
                icState.pathIdx = pathIdx;
                icState.D = L;
                icState.dist = dist - 2 * geometryEpsilon;
                icState.directLum = contribution;
                icState.pixelIdx = pixelIdx;
                
                connections[shadowRayIdx] = icState;
			}
		}
    }

    // cap at maxium path length
	// if (pathLength == NRC_FULL_MAX_PATHLENGTH - 1) {
    //     // TODO: figure out how to lookup the net, since
    //     // the contribution factor is puzzling
    //     return;
    // }

    // evaluate bsdf to obtain direction for next path segment
	float3 R;
	float newBsdfPdf;
	bool specular = false;
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, D * -1.0f, HIT_T, r4.z, r4.w, RandomFloat( seed ), R, newBsdfPdf, specular );
    if (specular) flags |= S_SPECULAR;

    const float p = ((flags & S_SPECULAR) || ((flags & S_BOUNCED) == 0)) ? 1 : SurvivalProbability( bsdf );

    if (newBsdfPdf < EPSILON || isnan( newBsdfPdf )) {
        // TODO: better estimation?

        return;
    }

    if (pathLength == NRC_FULL_MAX_PATHLENGTH - 1 || p < RandomFloat(seed)) {

        const uint infIdx = atomicAdd(numRaysToBeInferenced, 1);
        NRCNetInferenceInput &iInput = inferenceInput[infIdx];
        
        iInput.rayIsect = I;
        iInput.roughness = ROUGHNESS;
        iInput.rayDir = toSphericalCoord(D);
        iInput.normalDir = toSphericalCoord(fN);
        iInput.diffuseRefl = shadingData.color;
        iInput.specularRefl = shadingData.color;  // TODO: figure out
        iInput.dummies[0] = iInput.dummies[1] = 0.0f;

        // iInput.roughness = 0.0f;
        // iInput.rayDir = make_float2(0.0f);
        // iInput.normalDir = make_float2(0.0f);
        // iInput.diffuseRefl = make_float3(0.0f);
        // iInput.specularRefl = make_float3(0.0f);
        // iInput.dummies[0] = iInput.dummies[1] = 0.0f;

        inferencePixelIndices[infIdx] = pixelIdx;
        inferencePixelContribs[infIdx] = throughput * bsdf * abs(dot(fN, R)) / newBsdfPdf;

        return;
    }

    const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
	if (!(flags & S_SPECULAR)) {
        flags |= flags & S_BOUNCED ? S_BOUNCEDTWICE : S_BOUNCED;
    } else {
        flags |= S_VIASPECULAR;
    }

    InferencePathState nextIp;
    nextIp.O = SafeOrigin(I, R, N, geometryEpsilon);
    nextIp.flags = flags;
    nextIp.D = R;
    nextIp.pathIdx = pathIdx;
    nextIp.throughput = throughput * bsdf * abs(dot(fN, R)) / newBsdfPdf;
    nextIp.pixelIdx = pixelIdx;

    nextPathStates[extensionRayIdx] = nextIp;
}

__host__ void shadeNRC(
    float4* accumulator, 
    InferencePathState* pathStates, const uint pathCount,
    InferencePathState* nextPathStates,
    float4* hits, 
    InferenceConnState* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
    const int pathLength, const int w, const int h, const float spreadAngle,
    int* numRaysToBeInferenced,
    NRCNetInferenceInput* inferenceInput,
    uint* inferencePixelIndices,
    float3* inferencePixelContribs
) {
    const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 );
    shadeNRCKernel <<<gridDim.x, 128>>> (
        accumulator,
        pathStates, pathCount,
        nextPathStates,
        hits,
        connections,
        R0, shift, blueNoise, pass,
        pathLength, w, h, spreadAngle,
        numRaysToBeInferenced,
        inferenceInput,
        inferencePixelIndices,
        inferencePixelContribs
    );
}

__global__ void nrcTraceBufPostprocess_Kernel(
    NRCTraceBuf* traceBuf,
    uint numTrainRays
) {
    uint jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (jobIndex >= numTrainRays) {
        return;
    }
    
    float3 prevLumOut = make_float3(0.0f);
    for (int pLen = NRC_MAX_TRAIN_PATHLENGTH - 1; pLen >= 0; pLen--) {
        NRCTraceBufComponent &comp = traceBuf[jobIndex].traceComponent[pLen];
        if (comp.traceFlags == 0) {
            continue;
        }

        comp.lumOutput += comp.throughput * prevLumOut;
        prevLumOut = comp.lumOutput;
    }
}

// TODO: implement me
__host__ void nrcTraceBufPostprocess(
    NRCTraceBuf* traceBuf,
    uint numTrainRays
) {
    const dim3 gridDim( NEXTMULTIPLEOF( numTrainRays, 128 ) / 128, 1 );
    nrcTraceBufPostprocess_Kernel <<<gridDim.x, 128>>>(
        traceBuf,
        numTrainRays
    );
}


__global__ void nrcContribAdd_Kernel(
    float4* accumulator,
    const uint numInferenceRays,
    const NRCNetInferenceOutput* infOutput,
    const uint* infPixelIndices,
    const float3* infPixelContribs
) {
    uint jobIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (jobIndex >= numInferenceRays) {
        return;
    }

    accumulator[infPixelIndices[jobIndex]] += make_float4(
        infOutput[jobIndex].lumOutput * infPixelContribs[jobIndex],
        0.0f
    );
}

__host__ void nrcContribAdd(
    float4* accumulator,
    const uint numInferenceRays,
    const NRCNetInferenceOutput *infOutput,
    const uint* infPixelIndices,
    const float3* infPixelContribs
) {
    nrcContribAdd_Kernel <<< NEXTMULTIPLEOF(numInferenceRays, 32) / 32, 32 >>> (
        accumulator,
        numInferenceRays,
        infOutput,
        infPixelIndices,
        infPixelContribs
    );
}

#include "nrcEnhancedPathTracer.h"

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
