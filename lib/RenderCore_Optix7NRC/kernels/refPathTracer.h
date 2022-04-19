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
#define RAY_O make_float3( O4 )
#define FLAGS data
#define PATHIDX (data >> 6)

#if __CUDA_ARCH__ > 700 // Volta deliberately excluded
__global__  __launch_bounds__( 128 /* max block size */, 2 /* min blocks per sm TURING */ )
#else
__global__  __launch_bounds__( 256 /* max block size */, 2 /* min blocks per sm, PASCAL, VOLTA */ )
#endif
void shadeRefKernel( float4* accumulator, const uint stride,
	float4* pathStates, float4* hits, float4* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int w, const int h, const float spreadAngle,
	const uint pathCount )
{
	// respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

	// gather data by reading sets of four floats for optimal throughput
	const float4 O4 = pathStates[jobIndex];				// ray origin xyz, w can be ignored
	const float4 D4 = pathStates[jobIndex + stride];	// ray direction xyz
	float4 T4 = pathLength == 1 ? make_float4( 1 ) /* faster */ : pathStates[jobIndex + stride * 2]; // path thoughput rgb
	const float4 hitData = hits[jobIndex];
	hits[jobIndex].z = __int_as_float( -1 ); // reset for next query
	const float bsdfPdf = T4.w;

	// derived data
	uint data = __float_as_uint( O4.w ); // prob.density of the last sampled dir, postponed because of MIS
	const float3 D = make_float3( D4 );
	float3 throughput = make_float3( T4 );
	const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;
	const uint pathIdx = PATHIDX;
	const uint pixelIdx = pathIdx % (w * h);
	const uint sampleIdx = pathIdx / (w * h) + pass;

	// initialize depth in accumulator for DOF shader
	// if (pathLength == 1) accumulator[pixelIdx].w += PRIMIDX == NOHIT ? 10000 : HIT_T;

	// use skydome if we didn't hit any geometry
	if (PRIMIDX == NOHIT)
	{
		float3 tD = -worldToSky.TransformVector( D );
		float3 skyPixel = FLAGS & S_BOUNCED ? SampleSmallSkydome( tD ) : SampleSkydome( tD );
		float3 contribution = throughput * skyPixel * (1.0f / bsdfPdf);
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
		FIXNAN_FLOAT3( contribution );
		accumulator[pixelIdx] += make_float4( contribution, 0 );
		return;
	}

	// object picking
	if (pixelIdx == probePixelIdx && pathLength == 1)
	{
		counters->probedInstid = INSTANCEIDX;	// record instace id at the selected pixel
		counters->probedTriid = PRIMIDX;		// record primitive id at the selected pixel
		counters->probedDist = HIT_T;			// record primary ray hit distance
	}

	// get shadingData and normals
	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = RAY_O + HIT_T * D;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( D, HIT_U, HIT_V, coneWidth, instanceTriangles[PRIMIDX], INSTANCEIDX, shadingData, N, iN, fN, T );
	uint seed = WangHash( pathIdx * 17 + R0 /* well-seeded xor32 is all you need */ );

	// we need to detect alpha in the shading code.
	if (shadingData.flags & 1)
	{
		if (pathLength < MAXPATHLENGTH)
		{
			const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
			pathStates[extensionRayIdx] = make_float4( I + D * geometryEpsilon, O4.w );
			pathStates[extensionRayIdx + stride] = D4;
			if (!(isfinite( T4.x + T4.y + T4.z ))) T4 = make_float4( 0, 0, 0, T4.w );
			pathStates[extensionRayIdx + stride * 2] = T4;
		}
		return;
	}

	// stop on light
	if (shadingData.IsEmissive() /* r, g or b exceeds 1 */)
	{
		const float DdotNL = -dot( D, N );
		float3 contribution = make_float3( 0 ); // initialization required.
		if (DdotNL > 0 /* lights are not double sided */)
		{
			if (pathLength == 1 || (FLAGS & S_SPECULAR) > 0 || connections == 0)
			{
				// accept light contribution if previous vertex was specular
				contribution = shadingData.color;
			}
			else
			{
				contribution = throughput * shadingData.color;
			}
			CLAMPINTENSITY;
			FIXNAN_FLOAT3( contribution );
			accumulator[pixelIdx] += make_float4( contribution, 0 );
		}
		return;
	}

	// path regularization
	if (FLAGS & S_BOUNCED) shadingData.parameters.x |= 255u << 24; // set roughness to 1 after a bounce

	// detect specular surfaces
	if (ROUGHNESS <= 0.001f || TRANSMISSION > 0.5f) FLAGS |= S_SPECULAR; /* detect pure speculars; skip NEE for these */ else FLAGS &= ~S_SPECULAR;

	// normal alignment for backfacing polygons
	const float faceDir = (dot( D, N ) > 0) ? -1 : 1;
	if (faceDir == 1) shadingData.transmittance = make_float3( 0 );

	// prepare random numbers
	float4 r4;
	if (sampleIdx < 64)
	{
		const uint x = ((pathIdx % w) + (shift & 127)) & 127;
		const uint y = ((pathIdx / w) + (shift >> 24)) & 127;
		r4 = blueNoiseSampler4( blueNoise, x, y, sampleIdx, 4 * pathLength - 4 );
	}
	else
	{
		r4.x = RandomFloat( seed ), r4.y = RandomFloat( seed );
		r4.z = RandomFloat( seed ), r4.w = RandomFloat( seed );
	}

	// next event estimation: connect eye path to light
	if ((FLAGS & S_SPECULAR) == 0 && connections != 0) // skip for specular vertices
	{
		float pickProb, lightPdf = 0;
		float3 lightColor, L = RandomPointOnLight( r4.x, r4.y, I, fN * faceDir, pickProb, lightPdf, lightColor ) - I;
		// float3 lightColor, L = RandomPointOnLightLTree( r4.x, r4.y, seed, I, fN * faceDir, pickProb, lightPdf, lightColor, false ) - I;
		const float dist = length( L );
		L *= 1.0f / dist;
		const float NdotL = dot( L, fN * faceDir );
		if (NdotL > 0 && lightPdf > 0)
		{
			float bsdfPdf;
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN /* * faceDir */, T, D * -1.0f, L, bsdfPdf );
			{
				// calculate potential contribution
				float3 contribution = throughput * sampledBSDF * lightColor * (NdotL / (pickProb * lightPdf));
				FIXNAN_FLOAT3( contribution );
				CLAMPINTENSITY;
				// add fire-and-forget shadow ray to the connections buffer
				const uint shadowRayIdx = atomicAdd( &counters->shadowRays, 1 ); // compaction
				connections[shadowRayIdx] = make_float4( SafeOrigin( I, L, N, geometryEpsilon ), 0 ); // O4
				connections[shadowRayIdx + stride * 2] = make_float4( L, dist - 2 * geometryEpsilon ); // D4
				connections[shadowRayIdx + stride * 2 * 2] = make_float4( contribution, __int_as_float( pixelIdx ) ); // E4
			}
		}
	}

	// cap at two diffuse bounces, or a maxium path length
	if (FLAGS & ENOUGH_BOUNCES || pathLength == MAXPATHLENGTH) return;

	// evaluate bsdf to obtain direction for next path segment
	float3 R;
	float newBsdfPdf;
	bool specular = false;
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, D * -1.0f, HIT_T, r4.z, r4.w, RandomFloat( seed ), R, newBsdfPdf, specular );
	if (newBsdfPdf < EPSILON || isnan( newBsdfPdf )) return;
	if (specular) FLAGS |= S_SPECULAR;

	// russian roulette
	const float p = ((FLAGS & S_SPECULAR) || ((FLAGS & S_BOUNCED) == 0)) ? 1 : SurvivalProbability( bsdf );
	if (p < RandomFloat( seed )) return; else throughput *= 1 / p;

	// write extension ray, with compaction. Note: nvcc will aggregate automatically, 
	// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics 
	const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
	const uint packedNormal = PackNormal( fN * faceDir );
	if (!(FLAGS & S_SPECULAR)) FLAGS |= FLAGS & S_BOUNCED ? S_BOUNCEDTWICE : S_BOUNCED; else FLAGS |= S_VIASPECULAR;
	pathStates[extensionRayIdx] = make_float4( SafeOrigin( I, R, N, geometryEpsilon ), __uint_as_float( FLAGS ) );
	pathStates[extensionRayIdx + stride] = make_float4( R, __uint_as_float( packedNormal ) );
	FIXNAN_FLOAT3( throughput );
	pathStates[extensionRayIdx + stride * 2] = make_float4( throughput * bsdf * abs( dot( fN, R ) ) / newBsdfPdf, 1.0f);
}

__host__ void shadeRef( const int pathCount, float4* accumulator, const uint stride,
	float4* pathStates, float4* hits, float4* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int scrwidth, const int scrheight, const float spreadAngle )
{
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 );
	shadeRefKernel << <gridDim.x, 128 >> > (accumulator, stride, pathStates, hits, connections, R0, shift, blueNoise,
		pass, probePixelIdx, pathLength, scrwidth, scrheight, spreadAngle, pathCount);
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
#undef RAY_O
#undef FLAGS
#undef PATHIDX
