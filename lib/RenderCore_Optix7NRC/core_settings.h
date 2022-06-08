/* core_settings.h - Copyright 2019/2021 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   The settings and classes in this file are core-specific:
   - available in host and device code
   - specific to this particular core.
   Global settings can be configured shared.h.
*/

#pragma once

// core-specific settings
#define CLAMPFIREFLIES		// suppress fireflies by clamping
#define MAXPATHLENGTH		3
// #define CONSISTENTNORMALS	// consistent normal interpolation

// nrc settings
#define NRC_MAX_TRAIN_PATHLENGTH 3
#define NRC_FULL_MAX_PATHLENGTH 3

#define NRCNET_RESETMODE_UNIFORM 0

// low-level settings
#define BLUENOISE			// use blue noise instead of uniform random numbers
#define BILINEAR			// enable bilinear interpolation
// #define NOTEXTURES		// all texture reads will be white

#define APPLYSAFENORMALS	if (dot( N, wi ) <= 0) pdf = 0;
#define NOHIT				-1

#ifndef __CUDACC__

#define CUDABUILD			// signal system.h to include full CUDA headers
#include "helper_math.h"	// for vector types
#include "platform.h"
#undef APIENTRY				// get rid of an anoying warning
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "nvrtc.h"
#include "FreeImage.h"		// for loading blue noise
#include "shared_host_code/cudatools.h"
#include "shared_host_code/interoptexture.h"

#include <optix.h>
#include <optix_stubs.h>

const char* ParseOptixError( OptixResult r );

#define CHK_OPTIX( stmt ) FATALERROR_IN_CALL( ( stmt ), ParseOptixError, "" )
#define CHK_OPTIX_LOG( stmt ) FATALERROR_IN_CALL( ( stmt ), ParseOptixError, "\n%s", log )

using namespace lighthouse2;

#include "core_mesh.h"

using namespace lh2core;

#else

#include <optix.h>

#endif

#define NRC_INVALID_FLOAT -5e3
#define NRC_INVALID_FLOAT2 make_float2(NRC_INVALID_FLOAT)
#define NRC_INVALID_FLOAT3 make_float3(NRC_INVALID_FLOAT)

// lower bits are used to do this
#define NRC_TRACEFLAG_HIT_SKYBOX 1
#define NRC_TRACEFLAG_HIT_LIGHT_FRONT (1 << 1)
#define NRC_TRACEFLAG_HIT_LIGHT_BACK (1 << 2)
#define NRC_TRACEFLAG_NEE_EMIT (1 << 3)
#define NRC_TRACEFLAG_PATHLEN_TRUNCTUATE (1 << 4)
#define NRC_TRACEFLAG_BSDF_TRUNCTUATE (1 << 5)
#define NRC_TRACEFLAG_RR_TRUNCTUATE (1 << 6)
#define NRC_TRACEFLAG_NEXT_BOUNCE (1 << 7)

// NRC structures
struct alignas(sizeof(float) * 4) NRCTraceBufComponent {
    float3 rayIsect;
    float roughness;
    float2 rayDir;
    float2 normalDir;
    float3 diffuseRefl;
    uint traceFlags;
    float3 specularRefl;
    uint pathIdx;
    float3 lumOutput;
	uint pixelIdx;
    float3 throughput;
    float rrProb;
};

struct NRCTraceBuf {
	NRCTraceBufComponent traceComponent[NRC_MAX_TRAIN_PATHLENGTH];
};

// the one that conforms to refPathTracer
struct alignas(sizeof(float) * 4) TrainPathState
{
	float3 O;
	uint flags;
	float3 D;
	uint pathIdx;
	float3 throughput;
	uint pixelIdx;
	// float bsdfPdf;
	// int N;
	// float2 dummy;
};

using InferencePathState = TrainPathState;

struct alignas(sizeof(float) * 4) TrainConnectionState
{
	float3 O;
	int pathIdx;
	float3 D;
	float dist;
	float3 directLum;
	int pixelIdx;
};

using InferenceConnState = TrainConnectionState;

struct alignas(sizeof(float) * 4) NRCNetInferenceInput
{
	float3 rayIsect;
	float roughness;
	float2 rayDir;
	float2 normalDir;
	float3 diffuseRefl;
	float3 specularRefl;
	float dummies[2];    // TODO: remove this once we fix tiny-cuda-nn; set to 0 before
};

// SHOULD BE ALIGNED AS 3 FLOAT
struct NRCNetInferenceOutput
{
	float3 lumOutput;
};

#ifdef __CUDACC__
__global__ void nrc_check_align_cudacc(float* dummy) {
	// do some random things to avoid being optimized out, if any
	dummy[0] = 1;

	// the working part
	static_assert(sizeof(NRCTraceBufComponent) == 6 * 4 * sizeof(float), "size unexpected");
	static_assert(
		sizeof(NRCTraceBuf) == sizeof(NRCTraceBufComponent) * NRC_MAX_TRAIN_PATHLENGTH,
		"size unexpected"
	);
	static_assert(sizeof(TrainPathState) == 3 * 4 * sizeof(float), "size unexpected");

	static_assert(sizeof(NRCNetInferenceInput) == 4 * 4 * sizeof(float), "size unexpected");
	static_assert(sizeof(NRCNetInferenceOutput) == 3 * sizeof(float), "size unexpected");
}

#else
inline void nrc_check_align_hostcc(float* dummy) {
	// do some random things to avoid being optimized out, if any
	dummy[0] = 1;

	// the working part
	static_assert(sizeof(NRCTraceBufComponent) == 6 * 4 * sizeof(float), "size unexpected");
	static_assert(
		sizeof(NRCTraceBuf) == sizeof(NRCTraceBufComponent) * NRC_MAX_TRAIN_PATHLENGTH,
		"size unexpected"
	);
	static_assert(sizeof(TrainPathState) == 3 * 4 * sizeof(float), "size unexpected");
	static_assert(sizeof(NRCNetInferenceInput) == 4 * 4 * sizeof(float), "size unexpected");
	static_assert(sizeof(NRCNetInferenceOutput) == 3 * sizeof(float), "size unexpected");
}
#endif

// for a full path state, we need a Ray, an Intersection, and
// the data specified in PathState.
struct PathState
{
	float3 O;				// normal at last path vertex
	uint data;				// holds pixel index << 8, path length in bits 0..3, flags in bits 4..7
	float3 D; int N;		// ray direction of the current path segment, encoded normal in w
	float3 throughput;		// path transport
	float bsdfPdf;			// postponed pdf
};
struct PathState4 { float4 O4, D4, T4; };

// PotentialContribution: besides a shadow ray, a connection needs
// a pixel index and the energy that will be depositied to that pixel
// if there is no occlusion.
struct PotentialContribution
{
	float3 O;
	uint dummy1;
	float3 D;
	uint dummy2;
	float3 E;
	uint pixelIdx;
};
struct PotentialContribution4 { float4 O4, D4, E4; };

// counters and other global data, in device memory
struct Counters
{
	uint activePaths;
	uint shaded;
	uint generated;
	uint connected;
	uint extended;
	uint extensionRays;
	uint shadowRays;
	uint totalExtensionRays;
	uint totalShadowRays;
	int probedInstid;
	int probedTriid;
	float probedDist;
};

// path tracer parameters
struct Params
{
	enum
	{
		SPAWN_PRIMARY = 0,	// optix code will spawn and trace primary rays
		SPAWN_SHADOW,		// optix code will spawn and trace shadow rays
		SPAWN_SECONDARY,		// optix code will spawn and trace extension rays
		SPAWN_NRC_PRIMARY_UNIFORM,	// Use uniform sampler
		SPAWN_NRC_PRIMARY_HILTON,	// Use Hilton sampler
		SPAWN_NRC_SECONDARY,
		SPAWN_NRC_SHADOW,
		SPAWN_INF_PRIMARY,
		SPAWN_INF_SECONDARY,
		SPAWN_INF_SHADOW
	};
	float4 posLensSize;
	float3 right, up, p1;
	float geometryEpsilon;
	float distortion;
	int3 scrsize;
	int pass, phase, shift;
	float4* accumulator;
	float4* connectData;
	float4* hitData;
	float4* pathStates;
	uint* blueNoise;
	OptixTraversableHandle bvhRoot;

	// -- NRC added --
	TrainPathState* trainPathStates;
	TrainConnectionState* trainConnStates;
	InferencePathState* infPathStates;
	InferenceConnState* infConnStates;
	NRCTraceBuf* trainTraces;
	uint pathLength;
};

// internal material representation
struct CUDAMaterial
{
#ifndef OPTIX_CU
	void SetDiffuse( float3 d ) { diffuse_r = d.x, diffuse_g = d.y, diffuse_b = d.z; }
	void SetTransmittance( float3 t ) { transmittance_r = t.x, transmittance_g = t.y, transmittance_b = t.z; }
	struct Map { short width, height; half uscale, vscale, uoffs, voffs; uint addr; };
	// data to be read unconditionally
	half diffuse_r, diffuse_g, diffuse_b, transmittance_r, transmittance_g, transmittance_b; uint flags;
	uint4 parameters; // 16 Disney principled BRDF parameters, 0.8 fixed point
	// texture / normal map descriptors; exactly 128-bit each
	Map tex0, tex1, nmap0, nmap1, smap, rmap;
#endif
};

struct CUDAMaterial4
{
	uint4 baseData4;
	uint4 parameters;
	uint4 t0data4;
	uint4 t1data4;
	uint4 n0data4;
	uint4 n1data4;
	uint4 sdata4;
	uint4 rdata4;
	// flag query macros
#define ISDIELECTRIC				(1 << 0)
#define DIFFUSEMAPISHDR				(1 << 1)
#define HASDIFFUSEMAP				(1 << 2)
#define HASNORMALMAP				(1 << 3)
#define HASSPECULARITYMAP			(1 << 4)
#define HASROUGHNESSMAP				(1 << 5)
#define ISANISOTROPIC				(1 << 6)
#define HAS2NDNORMALMAP				(1 << 7)
#define HAS2NDDIFFUSEMAP			(1 << 9)
#define HASSMOOTHNORMALS			(1 << 11)
#define HASALPHA					(1 << 12)
#define HASMETALNESSMAP				(1 << 13)
#define MAT_ISDIELECTRIC			(flags & ISDIELECTRIC)
#define MAT_DIFFUSEMAPISHDR			(flags & DIFFUSEMAPISHDR)
#define MAT_HASDIFFUSEMAP			(flags & HASDIFFUSEMAP)
#define MAT_HASNORMALMAP			(flags & HASNORMALMAP)
#define MAT_HASSPECULARITYMAP		(flags & HASSPECULARITYMAP)
#define MAT_HASROUGHNESSMAP			(flags & HASROUGHNESSMAP)
#define MAT_ISANISOTROPIC			(flags & ISANISOTROPIC)
#define MAT_HAS2NDNORMALMAP			(flags & HAS2NDNORMALMAP)
#define MAT_HAS2NDDIFFUSEMAP		(flags & HAS2NDDIFFUSEMAP)
#define MAT_HASSMOOTHNORMALS		(flags & HASSMOOTHNORMALS)
#define MAT_HASALPHA				(flags & HASALPHA)
#define MAT_HASMETALNESSMAP			(flags & HASMETALNESSMAP)
};

// ------------------------------------------------------------------------------
// Below this line: derived, low-level and internal.

// clamping
#ifdef CLAMPFIREFLIES
#define CLAMPINTENSITY		const float v=max(contribution.x,max(contribution.y,contribution.z)); \
							if(v>clampValue){const float m=clampValue/v;contribution.x*=m; \
							contribution.y*=m;contribution.z*=m; /* don't touch w */ }
#else
#define CLAMPINTENSITY
#endif

#if !defined(__CUDACC__) && !defined(NRC_SKIP_RENDERCORE_INCLUDE)
#define OPTIXU_MATH_DEFINE_IN_NAMESPACE
#define _USE_MATH_DEFINES
#include "core_api_base.h"
#include "rendercore.h"
#include <cstdint>

namespace lh2core
{

// setters / getters
void stageInstanceDescriptors( CoreInstanceDesc* p );
void stageMaterialList( CUDAMaterial* p );
void stageTriLights( CoreLightTri* p );
void stagePointLights( CorePointLight* p );
void stageSpotLights( CoreSpotLight* p );
void stageDirectionalLights( CoreDirectionalLight* p );
void stageLightCounts( int tris, int point, int spot, int directional );
void stageARGB32Pixels( uint* p );
void stageARGB128Pixels( float4* p );
void stageNRM32Pixels( uint* p );
void stageSkyPixels( float4* p );
void stageSkySize( int w, int h );
void stageWorldToSky( const mat4& worldToLight );
void stageDebugData( float4* p );
void stageGeometryEpsilon( float e );
void stageClampValue( float c );
void stageMemcpy( void* d, void* s, int n );
void stageLightTree( LightCluster* t );
void pushStagedCopies();
void SetCounters( Counters* p );

} // namespace lh2core

#include "../RenderSystem/common_bluenoise.h"

#endif

// EOF