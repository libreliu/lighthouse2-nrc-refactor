/* rendersystem.h - Copyright 2019/2021 Utrecht University

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

#include "system.h"
#ifdef RENDERSYSTEMBUILD
// we will not expose these to the host application
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"
#include "tinyxml2.h"
#include "FreeImage.h"
typedef tinygltf::AnimationSampler tinygltfAnimationSampler;
typedef tinygltf::AnimationChannel tinygltfAnimationChannel;
typedef tinygltf::Animation tinygltfAnimation;
typedef tinygltf::Model tinygltfModel;
typedef tinygltf::Mesh tinygltfMesh;
typedef tinygltf::Skin tinygltfSkin;
typedef tinygltf::Node tinygltfNode;
typedef tinygltf::Material tinygltfMaterial;
typedef tinyobj::material_t tinyobjMaterial;
#else
typedef int tinygltfAnimationSampler;
typedef int tinygltfAnimationChannel;
typedef int tinygltfAnimation;
typedef int tinygltfModel;
typedef int tinygltfMesh;
typedef int tinygltfSkin;
typedef int tinygltfNode;
typedef int tinygltfMaterial;
typedef int tinyobjMaterial;
#endif
#include "host_texture.h"
#include "host_material.h"
#include "host_mesh.h"
#include "host_light.h"
#include "host_skydome.h"
#include "camera.h"
#include "host_anim.h"
#include "host_scene.h"
#include "host_node.h"
#include "core_api_base.h"
#include "render_api.h"

#ifdef RENDERSYSTEMBUILD
using namespace tinyxml2;
#endif

namespace lighthouse2
{

struct RenderSettings
{
	float geometryEpsilon = 1.0e-3f;
	float filterDirectClamp = 15.0f;
	float filterIndirectClamp = 2.5f;
	uint filterEnabled = 1;
	uint TAAEnabled = 1;
};

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem                                                               |
//  |  High-level API.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
class RenderSystem
{
public:
	// methods
	void Init( const char* dllName );
	void SynchronizeSceneData();
	void Render( const ViewPyramid& view, Convergence converge, bool async = false );
	void WaitForRender();
	void SetTarget( GLTexture* target, const uint spp );
	void SetProbePos( int2 pos ) { if (core) core->SetProbePos( pos ); }
	void Setting( const char* name, const float value ) { if (core) core->Setting( name, value ); }
	int GetTriangleMaterial( const int coreInstId, const int coreTriId );
	int GetTriangleMesh( const int coreInstId, const int coreTriId );
	int GetTriangleNode( const int coreInstId, const int coreTriId );
	void Shutdown();
	CoreStats GetCoreStats() { return core ? core->GetCoreStats() : CoreStats(); }
	SystemStats GetSystemStats() { return stats; }
	inline bool EnableFeatureExt( const char* name ) {
		if (core) return core->EnableFeatureExt(name); else return false;
	}
	inline bool EnableAuxTargetExt( const char* name, GLTexture *target ) {
		if (core) return core->EnableAuxTargetExt(name, target); else return false;
	}
	inline bool DisableAuxTargetExt( const char* name ) {
		if (core) return core->DisableAuxTargetExt(name); else return false;
	}
	inline std::string GetSettingStringExt( const char* name ) {
		if (core) return core->GetSettingStringExt(name); else return "";
	}
	inline bool SettingStringExt( const char* name, const char* value ) {
		if (core) return core->SettingStringExt(name, value); else return false;
	}
private:
	// private methods
	void SynchronizeSky();
	void SynchronizeTextures();
	void SynchronizeMaterials();
	void SynchronizeMeshes();
	void SynchronizeLights();
	void UpdateSceneGraph();
private:
	// private data members
	CoreAPI_Base* core = nullptr;			// low-level rendering functionality
	GLTexture* renderTarget = nullptr;		// CUDA will render to this OpenGL texture
	bool meshesChanged = false;				// rebuild scene graph if a mesh was rebuilt / refit
	SystemStats stats;						// performance counters
	vector<int> instances;					// node indices that have been sent to the core as instances
public:
	// public data members
	HostScene* scene = nullptr;				// scene I/O and management module
	RenderSettings settings;				// render settings container
};

} // namespace lighthouse2

// EOF