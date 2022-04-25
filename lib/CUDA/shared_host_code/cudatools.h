/* cudatools.h - Copyright 2019/2021 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

// Avoid circular dependency
#ifndef FATALERROR
// forward definition from system.h
void FatalError( const char* fmt, ... );

#define FATALERROR( f, ... ) FatalError( "Error on line %d of %s: " f "\n", __LINE__, __FILE__, ##__VA_ARGS__ )
#define FATALERROR_IF( c, f, ... ) do { if (c) FATALERROR( f, ##__VA_ARGS__ ); } while ( 0 )
#define FATALERROR_IN( p, e, f, ... ) FatalError( p " returned error '%s' at %s:%d" f "\n", e, __FILE__, __LINE__, ##__VA_ARGS__ );
#define FATALERROR_IN_CALL( s, e, f, ... ) do { auto r = (s); if (r) FATALERROR_IN( #s, e( r ), f, ##__VA_ARGS__ ) } while (0)
#endif

#define CHK_CUDA( stmt ) do { auto ret = ( stmt ); if ( ret ) {                                      \
if ( !strncmp( #stmt, "cudaGraphicsGLRegisterImage", sizeof( "cudaGraphicsGLRegisterImage" ) - 1 ) ) \
FATALERROR_IN( #stmt, CUDATools::decodeError( ret ), "\n\t(Are you running using the IGP?\n"         \
"Use NVIDIA control panel to enable the high performance GPU.)" ) else                               \
FATALERROR_IN( #stmt, CUDATools::decodeError( ret ), "" ) } } while ( 0 )
#define CHK_NVRTC( stmt ) FATALERROR_IN_CALL( ( stmt ), nvrtcGetErrorString, "" )

enum { NOT_ALLOCATED = 0, ON_HOST = 1, ON_DEVICE = 2, STAGED = 4 };
enum { POLICY_DEFAULT = 0, POLICY_COPY_SOURCE };

class CUDATools
{
public:
	static float Elapsed( cudaEvent_t start, cudaEvent_t end );
	static int _ConvertSMVer2Cores( int major, int minor );
	static int FastestDevice(); // from the CUDA 10.0 examples
	static const char* decodeError( cudaError_t res );
	static void compileToPTX( std::string& ptx, const char* cuSource, const char* sourceDir, const int cc, const int optixVer );
};


template <class T> class CoreBuffer
{
public:
	CoreBuffer() = default;
	CoreBuffer( uint64_t elements, uint64_t loc, const void* source = 0, const int policy = POLICY_DEFAULT ) : location( loc )
	{
		numElements = elements;
		sizeInBytes = elements * sizeof( T );
		if (elements > 0)
		{
			if (location & ON_DEVICE)
			{
				// location is ON_DEVICE; allocate room on device
				CHK_CUDA( cudaMalloc( &devPtr, sizeInBytes ) );
				owner |= ON_DEVICE;
			}
			if (location & ON_HOST)
			{
				// location is ON_HOST; use supplied pointer or allocate room if no source was specified
				if (source)
				{
					if (policy == POLICY_DEFAULT) hostPtr = (T*)source; else
					{
						// POLICY_COPY_SOURCE: pointer was supplied, and we are supposed to copy it
						hostPtr = (T*)MALLOC64( sizeInBytes ), owner |= ON_HOST;
						memcpy( hostPtr, source, sizeInBytes );
					}
					if (location & ON_DEVICE) CopyToDevice();
				}
				else
				{
					hostPtr = (T*)MALLOC64( sizeInBytes ), owner |= ON_HOST;
				}
			}
			else if (source && (location & ON_DEVICE))
			{
				// location is ON_DEVICE only, and we have data, so send the data over
				hostPtr = (T*)source;
				if ((location & STAGED) == 0) CopyToDevice();
				hostPtr = 0;
			}
		}
	}
	~CoreBuffer()
	{
		if (sizeInBytes > 0)
		{
			if (owner & ON_HOST)
			{
				FREE64( hostPtr );
				hostPtr = 0;
				owner &= ~ON_HOST;
			}
			if (owner & ON_DEVICE)
			{
				CHK_CUDA( cudaFree( devPtr ) );
				owner &= ~ON_DEVICE;
			}
		}
	}
	T* CopyToDevice()
	{
		if (sizeInBytes > 0)
		{
			if (!(location & ON_DEVICE))
			{
				CHK_CUDA( cudaMalloc( &devPtr, sizeInBytes ) );
				location |= ON_DEVICE;
				owner |= ON_DEVICE;
			}
			CHK_CUDA( cudaMemcpy( devPtr, hostPtr, sizeInBytes, cudaMemcpyHostToDevice ) );
		}
		return devPtr;
	}
	T* StageCopyToDevice()
	{
		if (sizeInBytes > 0)
		{
			if (!(location & ON_DEVICE))
			{
				CHK_CUDA( cudaMalloc( &devPtr, sizeInBytes ) );
				location |= ON_DEVICE;
				owner |= ON_DEVICE;
			}
			stageMemcpy( devPtr, hostPtr, sizeInBytes );
		}
		return devPtr;
	}
	T* MoveToDevice()
	{
		CopyToDevice();
		if (sizeInBytes > 0) FREE64( hostPtr );
		hostPtr = 0;
		owner &= ~ON_HOST;
		location &= ~ON_HOST;
		return devPtr;
	}
	T* CopyToHost()
	{
		if (sizeInBytes > 0)
		{
			if (!(location & ON_HOST))
			{
				hostPtr = (T*)MALLOC64( sizeInBytes );
				location |= ON_HOST;
				owner |= ON_HOST;
			}
			CHK_CUDA( cudaMemcpy( hostPtr, devPtr, sizeInBytes, cudaMemcpyDeviceToHost ) );
		}
		return hostPtr;
	}
	void Clear( int location, int overrideSize = -1 )
	{
		if (sizeInBytes > 0)
		{
			int bytesToClear = overrideSize == -1 ? sizeInBytes : overrideSize;
			if (location & ON_HOST) memset( hostPtr, 0, bytesToClear );
			if (location & ON_DEVICE) CHK_CUDA( cudaMemset( devPtr, 0, bytesToClear ) );
		}
	}
	uint64_t GetSizeInBytes() const { return sizeInBytes; }
	uint64_t GetSize() const { return numElements; }
	T* DevPtr() { return devPtr; }
	T** DevPtrPtr() { return &devPtr; /* Optix7 wants an array of pointers; this returns an array of 1 pointers. */ }
	T* HostPtr() { return hostPtr; }
	void SetHostData( T* hostData ) { hostPtr = hostData; }
	// member data
private:
	uint64_t location = NOT_ALLOCATED, owner = 0, sizeInBytes = 0, numElements = 0;
	T* devPtr = 0;
	T* hostPtr = 0;
};