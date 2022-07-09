/* tools_shared.h - Copyright 2019/2021 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   THIS IS A SHARED FILE: used in
   - RenderCore_OptixPrime_b
   - RenderCore_Optix7
   - RenderCore_Optix7Filter
   - RenderCore_PrimeRef
*/

#include "noerrors.h"

#define CHAR2FLT(a,s) (((float)(((a)>>s)&255))*(1.0f/255.0f))

struct ShadingData
{
	// This structure is filled for an intersection point. It will contain the spatially varying material properties.
	float3 color; int flags;
	float3 transmittance; int matID;
	float4 tint;
	uint4 parameters;
	/* 16 uchars:   x: 0..7 = metallic, 8..15 = subsurface, 16..23 = specular, 24..31 = roughness;
					y: 0..7 = specTint, 8..15 = anisotropic, 16..23 = sheen, 24..31 = sheenTint;
					z: 0..7 = clearcoat, 8..15 = clearcoatGloss, 16..23 = transmission, 24..31 = dummy;
					w: eta (32-bit float). */
	__device__ int IsSpecular( const int layer ) const { return 0; /* for now. */ }
	__device__ bool IsEmissive() const { return color.x > 1.0f || color.y > 1.0f || color.z > 1.0f; }
	__device__ void InvertETA() { parameters.w = __float_as_uint( 1.0f / __uint_as_float( parameters.w ) ); }
#define METALLIC CHAR2FLT( shadingData.parameters.x, 0 )
#define SUBSURFACE CHAR2FLT( shadingData.parameters.x, 8 )
#define SPECULAR CHAR2FLT( shadingData.parameters.x, 16 )
#define ROUGHNESS (max( 0.001f, CHAR2FLT( shadingData.parameters.x, 24 ) ))
#define SPECTINT CHAR2FLT( shadingData.parameters.y, 0 )
#define ANISOTROPIC CHAR2FLT( shadingData.parameters.y, 8 )
#define SHEEN CHAR2FLT( shadingData.parameters.y, 16 )
#define SHEENTINT CHAR2FLT( shadingData.parameters.y, 24 )
#define CLEARCOAT CHAR2FLT( shadingData.parameters.z, 0 )
#define CLEARCOATGLOSS CHAR2FLT( shadingData.parameters.z, 8 )
#define TRANSMISSION CHAR2FLT( shadingData.parameters.z, 16 )
#define TINT make_float3( shadingData.tint )
#define LUMINANCE shadingData.tint.w
#define DUMMY0 CHAR2FLT( shadingData.parameters.z, 24 )
#define ETA __uint_as_float( shadingData.parameters.w )
};
struct ShadingData4 { float4 data0, data1, tint4; uint4 data2; /* for fast 128-bit access */ };

// random numbers

LH2_DEVFUNC uint WangHash( uint s ) { s = (s ^ 61) ^ (s >> 16), s *= 9, s = s ^ (s >> 4), s *= 0x27d4eb2d, s = s ^ (s >> 15); return s; }
LH2_DEVFUNC uint RandomInt( uint& s ) { s ^= s << 13, s ^= s >> 17, s ^= s << 5; return s; }
LH2_DEVFUNC float RandomFloat( uint& s ) { return RandomInt( s ) * 2.3283064365387e-10f; }

// math helpers

LH2_DEVFUNC float2 min2( const float2& a, const float2& b ) { return make_float2( min( a.x, b.x ), min( a.y, b.y ) ); }
LH2_DEVFUNC float2 min2( const float2& a, const float b ) { return make_float2( min( a.x, b ), min( a.y, b ) ); }
LH2_DEVFUNC float2 min2( const float a, const float2& b ) { return make_float2( min( a, b.x ), min( a, b.y ) ); }
LH2_DEVFUNC float2 max2( const float2& a, const float2& b ) { return make_float2( max( a.x, b.x ), max( a.y, b.y ) ); }
LH2_DEVFUNC float2 max2( const float2& a, const float b ) { return make_float2( max( a.x, b ), max( a.y, b ) ); }
LH2_DEVFUNC float2 max2( const float a, const float2& b ) { return make_float2( max( a, b.x ), max( a, b.y ) ); }
LH2_DEVFUNC float3 min3( const float3& a, const float3& b ) { return make_float3( min( a.x, b.x ), min( a.y, b.y ), min( a.z, b.z ) ); }
LH2_DEVFUNC float3 min3( const float3& a, const float b ) { return make_float3( min( a.x, b ), min( a.y, b ), min( a.z, b ) ); }
LH2_DEVFUNC float3 min3( const float a, const float3& b ) { return make_float3( min( a, b.x ), min( a, b.y ), min( a, b.z ) ); }
LH2_DEVFUNC float3 max3( const float3& a, const float3& b ) { return make_float3( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ) ); }
LH2_DEVFUNC float3 max3( const float3& a, const float b ) { return make_float3( max( a.x, b ), max( a.y, b ), max( a.z, b ) ); }
LH2_DEVFUNC float3 max3( const float a, const float3& b ) { return make_float3( max( a, b.x ), max( a, b.y ), max( a, b.z ) ); }
LH2_DEVFUNC float4 min4( const float4& a, const float4& b ) { return make_float4( min( a.x, b.x ), min( a.y, b.y ), min( a.z, b.z ), min( a.w, b.w ) ); }
LH2_DEVFUNC float4 min4( const float4& a, const float b ) { return make_float4( min( a.x, b ), min( a.y, b ), min( a.z, b ), min( a.w, b ) ); }
LH2_DEVFUNC float4 min4( const float a, const float4& b ) { return make_float4( min( a, b.x ), min( a, b.y ), min( a, b.z ), min( a, b.w ) ); }
LH2_DEVFUNC float4 max4( const float4& a, const float4& b ) { return make_float4( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ), max( a.w, b.w ) ); }
LH2_DEVFUNC float4 max4( const float4& a, const float b ) { return make_float4( max( a.x, b ), max( a.y, b ), max( a.z, b ), max( a.w, b ) ); }
LH2_DEVFUNC float4 max4( const float a, const float4& b ) { return make_float4( max( a, b.x ), max( a, b.y ), max( a, b.z ), max( a, b.w ) ); }
LH2_DEVFUNC float sqrLength( const float3& a ) { return dot( a, a ); }
LH2_DEVFUNC float sqr( const float x ) { return x * x; }
LH2_DEVFUNC float oneoverpow2( const int p ) { return __uint_as_float( (127 - p) << 23 ); }
LH2_DEVFUNC float fastexpf( const float x )
{	// https://codingforspeed.com/using-faster-exponential-approximation
	float y = 1.0f + x / 256.0f;
	y *= y; y *= y; y *= y; y *= y;
	y *= y; y *= y; y *= y; y *= y;
	return y;
}
LH2_DEVFUNC float saturate( const float x ) { return max( 0.0f, min( 1.0f, x ) ); }
LH2_DEVFUNC float2 saturate( const float2 x ) { return max2( make_float2( 0 ), min2( make_float2( 1 ), x ) ); }
LH2_DEVFUNC float3 saturate( const float3 x ) { return max3( make_float3( 0 ), min3( make_float3( 1 ), x ) ); }
LH2_DEVFUNC float4 saturate( const float4 x ) { return max4( make_float4( 0 ), min4( make_float4( 1 ), x ) ); }
LH2_DEVFUNC float mix( const float a, const float b, const float x ) { return x <= 0 ? a : x >= 1 ? b : lerp( a, b, x ); }

// from: https://aras-p.info/texts/CompactNormalStorage.html
LH2_DEVFUNC uint PackNormal( const float3 N )
{
#if 1
	// more efficient
	const float f = 65535.0f / fmaxf( sqrtf( 8.0f * N.z + 8.0f ), 0.0001f ); // Thanks Robbin Marcus
	return (uint)(N.x * f + 32767.0f) + ((uint)(N.y * f + 32767.0f) << 16);
#else
	float2 enc = normalize( make_float2( N ) ) * (sqrtf( -N.z * 0.5f + 0.5f ));
	enc = enc * 0.5f + 0.5f;
	return (uint)(enc.x * 65535.0f) + ((uint)(enc.y * 65535.0f) << 16);
#endif
}
LH2_DEVFUNC float3 UnpackNormal( const uint p )
{
	float4 nn = make_float4( (float)(p & 65535) * (2.0f / 65535.0f), (float)(p >> 16) * (2.0f / 65535.0f), 0, 0 );
	nn += make_float4( -1, -1, 1, -1 );
	float l = dot( make_float3( nn.x, nn.y, nn.z ), make_float3( -nn.x, -nn.y, -nn.w ) );
	nn.z = l, l = sqrtf( l ), nn.x *= l, nn.y *= l;
	return make_float3( nn ) * 2.0f + make_float3( 0, 0, -1 );
}
// alternative method
LH2_DEVFUNC uint PackNormal2( const float3 N )
{
	// simple, and good enough discrimination of normals for filtering.
	const uint x = clamp( (uint)((N.x + 1) * 511), 0u, 1023u );
	const uint y = clamp( (uint)((N.y + 1) * 511), 0u, 1023u );
	const uint z = clamp( (uint)((N.z + 1) * 511), 0u, 1023u );
	return (x << 2u) + (y << 12u) + (z << 22u);
}
LH2_DEVFUNC float3 UnpackNormal2( const uint pi )
{
	const uint x = (pi >> 2u) & 1023u;
	const uint y = (pi >> 12u) & 1023u;
	const uint z = pi >> 22u;
	return make_float3( x * (1.0f / 511.0f) - 1, y * (1.0f / 511.0f) - 1, z * (1.0f / 511.0f) - 1 );
}

// color conversions

LH2_DEVFUNC float3 RGBToYCoCg( const float3 RGB )
{
	const float3 rgb = min3( make_float3( 4 ), RGB ); // clamp helps AA for strong HDR
	const float Y = dot( rgb, make_float3( 1, 2, 1 ) ) * 0.25f;
	const float Co = dot( rgb, make_float3( 2, 0, -2 ) ) * 0.25f + (0.5f * 256.0f / 255.0f);
	const float Cg = dot( rgb, make_float3( -1, 2, -1 ) ) * 0.25f + (0.5f * 256.0f / 255.0f);
	return make_float3( Y, Co, Cg );
}

LH2_DEVFUNC float3 YCoCgToRGB( const float3 YCoCg )
{
	const float Y = YCoCg.x;
	const float Co = YCoCg.y - (0.5f * 256.0f / 255.0f);
	const float Cg = YCoCg.z - (0.5f * 256.0f / 255.0f);
	return make_float3( Y + Co - Cg, Y + Cg, Y - Co - Cg );
}

LH2_DEVFUNC float Luminance( const float3 rgb )
{
	return 0.299f * min( rgb.x, 10.0f ) + 0.587f * min( rgb.y, 10.0f ) + 0.114f * min( rgb.z, 10.0f );
}

LH2_DEVFUNC uint HDRtoRGB32( const float3& c )
{
	const uint r = (uint)(1023.0f * min( 1.0f, c.x ));
	const uint g = (uint)(2047.0f * min( 1.0f, c.y ));
	const uint b = (uint)(2047.0f * min( 1.0f, c.z ));
	return (r << 22) + (g << 11) + b;
}
LH2_DEVFUNC float3 RGB32toHDR( const uint c )
{
	return make_float3(
		(float)(c >> 22) * (1.0f / 1023.0f),
		(float)((c >> 11) & 2047) * (1.0f / 2047.0f),
		(float)(c & 2047) * (1.0f / 2047.0f)
	);
}
LH2_DEVFUNC float3 RGB32toHDRmin1( const uint c )
{
	return make_float3(
		(float)max( 1u, c >> 22 ) * (1.0f / 1023.0f),
		(float)max( 1u, (c >> 11) & 2047 ) * (1.0f / 2047.0f),
		(float)max( 1u, c & 2047 ) * (1.0f / 2047.0f) );
}

LH2_DEVFUNC float SphericalTheta( const float3& v )
{
	return std::acos( clamp( v.z, -1.f, 1.f ) );
}

LH2_DEVFUNC float SphericalPhi( const float3& v )
{
	const float p = std::atan2( v.y, v.x );
	return (p < 0) ? (p + 2 * PI) : p;
}

LH2_DEVFUNC float3 SampleSkydome( const float3& D )
{
	const uint u = (uint)(skywidth * SphericalPhi( D ) * INV2PI - 0.5f);
	const uint v = (uint)(skyheight * SphericalTheta( D ) * INVPI - 0.5f);
	const uint idx = u + v * skywidth;
	return idx < skywidth* skyheight ? make_float3( skyPixels[idx] ) : make_float3( 0 );
}

LH2_DEVFUNC float3 SampleSmallSkydome( const float3& D )
{
	const uint w = skywidth >> 6, h = skyheight >> 6;
	const uint u = (uint)(w * SphericalPhi( D ) * INV2PI - 0.5f);
	const uint v = (uint)(h * SphericalTheta( D ) * INVPI - 0.5f);
	const uint idx = u + v * w;
	return idx < w* h ? make_float3( skyPixels[idx + skywidth * skyheight] ) : make_float3( 0 );
}

LH2_DEVFUNC float SurvivalProbability( const float3& albedo )
{
	// return min( 1.0f, max( max( albedo.x, albedo.y ), albedo.z ) );
	float albedoMax = max( max( albedo.x, albedo.y ), albedo.z );
	float passMax = 0.8f;
	return min( 1.0f, max(albedoMax, passMax) );
}

LH2_DEVFUNC float FresnelDielectricExact( const float3& wo, const float3& N, float eta )
{
	if (eta <= 1.0f) return 0.0f;
	const float cosThetaI = max( 0.0f, dot( wo, N ) );
	float scale = 1 / eta, cosThetaTSqr = 1 - (1 - cosThetaI * cosThetaI) * (scale * scale);
	if (cosThetaTSqr <= 0.0f) return 1.0f;
	float cosThetaT = sqrtf( cosThetaTSqr );
	float Rs = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
	float Rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
	return 0.5f * (Rs * Rs + Rp * Rp);
}

// origin offset
LH2_DEVFUNC float3 SafeOrigin( const float3& O, const float3& R, const float3& N, const float geoEpsilon )
{
#if 1
	// simply offset along the normal
	return O + N * (dot( N, R ) > 0 ? geoEpsilon : -geoEpsilon);
#else
#if 0
	// from Ray Tracing Gems 1, chapter 6: does not use geoEpsilon nor ray direction.
	const float3 _N = dot( N, R ) > 0 ? N : (-N);
	int3 of_i = make_int3( 256.0f * _N.x, 256.0f * _N.y, 256.0f * _N.z );
	float3 p_i = make_float3(
		__int_as_float( __float_as_int( O.x ) + ((O.x < 0) ? -of_i.x : of_i.x) ),
		__int_as_float( __float_as_int( O.y ) + ((O.y < 0) ? -of_i.y : of_i.y) ),
		__int_as_float( __float_as_int( O.z ) + ((O.z < 0) ? -of_i.z : of_i.z) ) );
	return make_float3( fabsf( O.x ) < (1.0f / 32.0f) ? O.x + (1.0f / 65536.0f) * _N.x : p_i.x,
		fabsf( O.y ) < (1.0f / 32.0f) ? O.y + (1.0f / 65536.0f) * _N.y : p_i.y,
		fabsf( O.z ) < (1.0f / 32.0f) ? O.z + (1.0f / 65536.0f) * _N.z : p_i.z );
#else
	// offset outgoing ray direction along R and / or N: along N when strongly parallel to the origin surface; mostly along R otherwise
	const float parallel = 1 - fabs( dot( N, R ) );
	const float v = parallel * parallel;
#if 1
	// we can go slightly into the surface when iN != N; negate the offset along N in that case
	const float side = dot( N, R ) < 0 ? -1 : 1;
#else
	// negating offset along N only makes sense once we backface cull
	const float side = 1.0f;
#endif
	return O + R * geoEpsilon * (1 - v) + N * side * geoEpsilon * v;
#endif
#endif
}

// consistent normal interpolation

LH2_DEVFUNC float3 ConsistentNormal( const float3& D, const float3& iN, const float alpha )
{
	// part of the implementation of "Consistent Normal Interpolation", Reshetov et al., 2010
	// calculates a safe normal given an incoming direction, phong normal and alpha
#if 0
	// Eq. 1, exact
	const float q = (1 - sinf( alpha )) / (1 + sinf( alpha ));
#else
	// Eq. 1 approximation, as in Figure 6 (not the wrong one in Table 8)
	const float t = PI - 2 * alpha, q = (t * t) / (PI * (PI + (2 * PI - 4) * alpha));
#endif
	const float b = dot( D, iN ), g = 1 + q * (b - 1), rho = sqrtf( q * (1 + g) / (1 + b) );
	const float3 Rc = (g + rho * b) * iN - (rho * D);
	return normalize( D + Rc );
}

LH2_DEVFUNC float4 CombineToFloat4( const float3& A, const float3& B )
{
	// convert two float3's to a single uint4, where each int stores two components of the input vectors.
	// assumptions:
	// - the input is positive
	// - the input can be safely clamped to 31.999
	// with this in mind, the data is stored as 5:11 unsigned fixed point, which should be plenty.
	const uint Ar = (uint)(min( A.x, 31.999f ) * 2048.0f), Ag = (uint)(min( A.y, 31.999f ) * 2048.0f), Ab = (uint)(min( A.z, 31.999f ) * 2048.0f);
	const uint Br = (uint)(min( B.x, 31.999f ) * 2048.0f), Bg = (uint)(min( B.y, 31.999f ) * 2048.0f), Bb = (uint)(min( B.z, 31.999f ) * 2048.0f);
	return make_float4( __uint_as_float( (Ar << 16) + Ag ), __uint_as_float( Ab ), __uint_as_float( (Br << 16) + Bg ), __uint_as_float( Bb ) );
}

LH2_DEVFUNC float3 GetDirectFromFloat4( const float4& X )
{
	const uint v0 = __float_as_uint( X.x ), v1 = __float_as_uint( X.y );
	return make_float3( (float)(v0 >> 16) * (1.0f / 2048.0f), (float)(v0 & 65535) * (1.0f / 2048.0f), (float)v1 * (1.0f / 2048.0f) );
}

LH2_DEVFUNC float3 GetIndirectFromFloat4( const float4& X )
{
	const uint v2 = __float_as_uint( X.z ), v3 = __float_as_uint( X.w );
	return make_float3( (float)(v2 >> 16) * (1.0f / 2048.0f), (float)(v2 & 65535) * (1.0f / 2048.0f), (float)v3 * (1.0f / 2048.0f) );
}

LH2_DEVFUNC float blueNoiseSampler( const uint* blueNoise, int x, int y, int sampleIndex, int sampleDimension, const float noiseShift = 0 )
{
	// Adapated from E. Heitz. Arguments:
	// sampleIndex: 0..255
	// sampleDimension: 0..255
	x &= 127, y &= 127, sampleIndex &= 255, sampleDimension &= 255;
	// xor index based on optimized ranking
	int rankedSampleIndex = (sampleIndex ^ blueNoise[sampleDimension + (x + y * 128) * 8 + 65536 * 3]) & 255;
	// fetch value in sequence
	int value = blueNoise[sampleDimension + rankedSampleIndex * 256];
	// if the dimension is optimized, xor sequence value based on optimized scrambling
	value ^= blueNoise[(sampleDimension & 7) + (x + y * 128) * 8 + 65536];
	// convert to float and return
	float retVal = (0.5f + value) * (1.0f / 256.0f) + noiseShift;
	if (retVal >= 1) retVal -= 1;
	return retVal;
}

LH2_DEVFUNC float4 blueNoiseSampler4( const uint* blueNoise, int x, int y, int sampleIndex, int sampleDimension )
{
	// Optimized retrieval of 4 blue noise samples.
	const uint4 bn4 = *((uint4*)(blueNoise + sampleDimension + (x + y * 128) * 8 + 65536 * 3));
	const int rsi1 = (sampleIndex ^ bn4.x) & 255, rsi2 = (sampleIndex ^ bn4.y) & 255;
	const int rsi3 = (sampleIndex ^ bn4.z) & 255, rsi4 = (sampleIndex ^ bn4.w) & 255;
	const int v1 = blueNoise[sampleDimension + 0 + rsi1 * 256];
	const int v2 = blueNoise[sampleDimension + 1 + rsi2 * 256];
	const int v3 = blueNoise[sampleDimension + 2 + rsi3 * 256];
	const int v4 = blueNoise[sampleDimension + 3 + rsi4 * 256];
	const uint4 bx4 = *((uint4*)(blueNoise + (sampleDimension & 7) + (x + y * 128) * 8 + 65536));
	return make_float4( (0.5f + (v1 ^ bx4.x)) * (1.0f / 256.0f), (0.5f + (v2 ^ bx4.y)) * (1.0f / 256.0f),
		(0.5f + (v3 ^ bx4.z)) * (1.0f / 256.0f), (0.5f + (v4 ^ bx4.w)) * (1.0f / 256.0f) );
}

// EOF