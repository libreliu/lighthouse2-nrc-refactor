#include "cudatools.h"
#include <nvrtc.h>

using namespace std;

float CUDATools::Elapsed(cudaEvent_t start, cudaEvent_t end) {
    float elapsed;
	cudaEventElapsedTime( &elapsed, start, end );
    return max( 1e-20f, elapsed ) * 0.001f; // report in seconds
}

int CUDATools::_ConvertSMVer2Cores( int major, int minor ) {
	typedef struct { int SM, Cores; } sSMtoCores;
	sSMtoCores nGpuArchCoresPerSM[] = { { 0x30, 192 }, { 0x32, 192 }, { 0x35, 192 }, { 0x37, 192 },
	{ 0x50, 128 }, { 0x52, 128 }, { 0x53, 128 }, { 0x60, 64 }, { 0x61, 128 }, { 0x62, 128 },
	{ 0x70, 64 }, { 0x72, 64 }, { 0x75, 64 }, { -1, -1 } };
	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) return nGpuArchCoresPerSM[index].Cores;
		index++;
	}
	return nGpuArchCoresPerSM[index - 1].Cores;
}

// from the CUDA 10.0 examples
int CUDATools::FastestDevice() {
	int curdev = 0, smperproc = 0, fastest = 0, count = 0, prohibited = 0;
	uint64_t max_perf = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount( &count );
	if (count == 0) FatalError( "No CUDA devices found.\nIf you do have an NVIDIA GPU, consider updating your drivers." );
	while (curdev < count)
	{
		cudaGetDeviceProperties( &deviceProp, curdev );
		if (deviceProp.computeMode != cudaComputeModeProhibited)
		{
			if (deviceProp.major == 9999 && deviceProp.minor == 9999) smperproc = 1; else
				smperproc = _ConvertSMVer2Cores( deviceProp.major, deviceProp.minor );
			uint64_t compute_perf = (uint64_t)deviceProp.multiProcessorCount * smperproc * deviceProp.clockRate;
			if (compute_perf > max_perf)
			{
				max_perf = compute_perf;
				fastest = curdev;
			}
		}
		else prohibited++;
		++curdev;
	}
	if (prohibited == count) FatalError( "All CUDA devices are prohibited from use." );
	return fastest;
}

const char* CUDATools::decodeError( cudaError_t res ) {
	switch (res) {
	default:                                        return "Unknown cudaError_t";
	case CUDA_SUCCESS:                              return "No error";
	case CUDA_ERROR_INVALID_VALUE:                  return "Invalid value";
	case CUDA_ERROR_OUT_OF_MEMORY:                  return "Out of memory";
	case CUDA_ERROR_NOT_INITIALIZED:                return "Not initialized";
	case CUDA_ERROR_DEINITIALIZED:                  return "Deinitialized";
	case CUDA_ERROR_NO_DEVICE:                      return "No device";
	case CUDA_ERROR_INVALID_DEVICE:                 return "Invalid device";
	case CUDA_ERROR_INVALID_IMAGE:                  return "Invalid image";
	case CUDA_ERROR_INVALID_CONTEXT:                return "Invalid context";
	case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:        return "Context already current";
	case CUDA_ERROR_MAP_FAILED:                     return "Map failed";
	case CUDA_ERROR_UNMAP_FAILED:                   return "Unmap failed";
	case CUDA_ERROR_ARRAY_IS_MAPPED:                return "Array is mapped";
	case CUDA_ERROR_ALREADY_MAPPED:                 return "Already mapped";
	case CUDA_ERROR_NO_BINARY_FOR_GPU:              return "No binary for GPU";
	case CUDA_ERROR_ALREADY_ACQUIRED:               return "Already acquired";
	case CUDA_ERROR_NOT_MAPPED:                     return "Not mapped";
	case CUDA_ERROR_INVALID_SOURCE:                 return "Invalid source";
	case CUDA_ERROR_FILE_NOT_FOUND:                 return "File not found";
	case CUDA_ERROR_INVALID_HANDLE:                 return "Invalid handle";
	case CUDA_ERROR_NOT_FOUND:                      return "Not found";
	case CUDA_ERROR_NOT_READY:                      return "Not ready";
	case CUDA_ERROR_LAUNCH_FAILED:                  return "Launch failed";
	case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:        return "Launch out of resources";
	case CUDA_ERROR_LAUNCH_TIMEOUT:                 return "Launch timeout";
	case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:  return "Launch incompatible texturing";
	case CUDA_ERROR_UNKNOWN:                        return "Unknown error";
	case CUDA_ERROR_PROFILER_DISABLED:              return "Profiler disabled";
	case CUDA_ERROR_PROFILER_NOT_INITIALIZED:       return "Profiler not initialized";
	case CUDA_ERROR_PROFILER_ALREADY_STARTED:       return "Profiler already started";
	case CUDA_ERROR_PROFILER_ALREADY_STOPPED:       return "Profiler already stopped";
	case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:            return "Not mapped as array";
	case CUDA_ERROR_NOT_MAPPED_AS_POINTER:          return "Not mapped as pointer";
	case CUDA_ERROR_ECC_UNCORRECTABLE:              return "ECC uncorrectable";
	case CUDA_ERROR_UNSUPPORTED_LIMIT:              return "Unsupported limit";
	case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:         return "Context already in use";
	case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "Shared object symbol not found";
	case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:      return "Shared object init failed";
	case CUDA_ERROR_OPERATING_SYSTEM:               return "Operating system error";
	case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:    return "Peer access already enabled";
	case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:        return "Peer access not enabled";
	case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:         return "Primary context active";
	case CUDA_ERROR_CONTEXT_IS_DESTROYED:           return "Context is destroyed";
	case CUDA_ERROR_ILLEGAL_ADDRESS:                return "Illegal address";
	case CUDA_ERROR_MISALIGNED_ADDRESS:             return "Misaligned address";
	}
}

void CUDATools::compileToPTX( string& ptx, const char* cuSource, const char* sourceDir, const int cc, const int optixVer ) {
	// create program
	nvrtcProgram prog = 0;
	CHK_NVRTC( nvrtcCreateProgram( &prog, cuSource, 0, 0, NULL, NULL ) );
	// gather NVRTC options
	vector<const char*> options;
#if 1
	// @Marijn: this doesn't work. Optix is used in several versions, distributed with LH2.
	// TODO: Throw FatalError if no path is defined for the requested OptiX version!
	if (optixVer > 6)
	{
	#ifdef OPTIX_INCLUDE_PATH
		options.push_back( "-I" OPTIX_INCLUDE_PATH );
	#else
		options.push_back( "-I../../lib/OptiX7/include" );
	#endif
	}
	else
	{
	#ifdef OPTIX_6_INCLUDE_PATH
		options.push_back( "-I" OPTIX_6_INCLUDE_PATH );
	#else
		options.push_back( "-I../../lib/OptiX/include" );
	#endif
	}
#else
	if (optixVer > 6) options.push_back( "-I../../lib/Optix7/include/" ); else options.push_back( "-I../../lib/Optix/include/" );
#endif
	string optionString = "-I";
	optionString += string( sourceDir );
	options.push_back( optionString.c_str() );
#ifdef _MSC_VER
	options.push_back( "-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/" );
#endif
	options.push_back( "-I../../lib/CUDA/" );
#ifdef _DEBUG
	options.push_back("--device-debug");
	options.push_back("--generate-line-info");
#endif
	// collect NVRTC options
	char versionString[64];
	snprintf( versionString, sizeof( versionString ), "compute_%i", cc >= 70 ? 70 : 50 );
	const char* compiler_options[] = { "-arch", versionString, "-restrict", "-std=c++11", "-use_fast_math", "-default-device", "-rdc", "true", "-D__x86_64", 0 };
	const size_t n_compiler_options = sizeof( compiler_options ) / sizeof( compiler_options[0] );
	for (size_t i = 0; i < n_compiler_options - 1; i++) options.push_back( compiler_options[i] );
	// JIT compile CU to PTX
	int versionMinor, versionMajor;
	nvrtcVersion( &versionMajor, &versionMinor );
	printf( "compiling cuda code using nvcrt %i.%i\n", versionMajor, versionMinor );
	const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );
	// retrieve log output
	size_t log_size = 0;
	CHK_NVRTC( nvrtcGetProgramLogSize( prog, &log_size ) );
	static string nvrtcLog;
	nvrtcLog.resize( log_size );
	if (log_size > 1) CHK_NVRTC( nvrtcGetProgramLog( prog, &nvrtcLog[0] ) );
	FATALERROR_IF( compileRes != NVRTC_SUCCESS, "Compilation failed.\n%s", nvrtcLog.c_str() );
	// retrieve PTX code
	size_t ptx_size = 0;
	CHK_NVRTC( nvrtcGetPTXSize( prog, &ptx_size ) );
	ptx.resize( ptx_size );
	CHK_NVRTC( nvrtcGetPTX( prog, &ptx[0] ) );
	// cleanup
	CHK_NVRTC( nvrtcDestroyProgram( &prog ) );
}