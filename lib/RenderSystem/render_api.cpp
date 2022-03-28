/* render_api.cpp - Copyright 2019/2021 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file implements the RebderSystem API, which is the interface
   between the application and the RenderSystem.
*/

#include "rendersystem.h"

static RenderSystem* renderer = nullptr;
static RenderAPI api;

RenderAPI* RenderAPI::CreateRenderAPI( const char* dllName )
{
	if (!renderer)
	{
		renderer = new RenderSystem();
		renderer->Init( dllName );
	}
	return &api;
}

void RenderAPI::SerializeMaterials( const char* xmlFile )
{
	renderer->scene->SerializeMaterials( xmlFile );
}

void RenderAPI::DeserializeMaterials( const char* xmlFile )
{
	renderer->scene->DeserializeMaterials( xmlFile );
}

void RenderAPI::Shutdown()
{
	renderer->Shutdown();
}

void RenderAPI::DeserializeCamera( const char* xmlFile )
{
	renderer->scene->camera->Deserialize( xmlFile );
}

void RenderAPI::SerializeCamera( const char* xmlFile )
{
	renderer->scene->camera->Serialize( xmlFile );
}

int RenderAPI::AddMesh( const char* file, const char* dir, const float scale, bool flatShaded )
{
	return renderer->scene->AddMesh( file, dir, scale, flatShaded );
}

int RenderAPI::AddMesh( const char* file, const float scale, bool flatShaded )
{
	return renderer->scene->AddMesh( file, scale, flatShaded );
}

int RenderAPI::AddMesh( const int triCount )
{
	return renderer->scene->AddMesh( triCount );
}

void RenderAPI::AddTriToMesh( const int meshId, const float3& v0, const float3& v1, const float3& v2, const int matId )
{
	return renderer->scene->AddTriToMesh( meshId, v0, v1, v2, matId );
}

int RenderAPI::AddScene( const char* file, const char* dir, const mat4& transform )
{
	return renderer->scene->AddScene( file, dir, transform );
}

int RenderAPI::AddScene( const char* file, const mat4& transform )
{
	return renderer->scene->AddScene( file, transform );
}

int RenderAPI::AddQuad( const float3 N, const float3 pos, const float width, const float height, const int material, const int meshID )
{
	return renderer->scene->AddQuad( N, pos, width, height, material, meshID );
}

int RenderAPI::AddInstance( const int meshId, const mat4& transform )
{
	return renderer->scene->AddInstance( meshId, transform );
}

void RenderAPI::RemoveNode( const int nodeId )
{
	return renderer->scene->RemoveNode( nodeId );
}

void RenderAPI::SetNodeTransform( const int nodeId, const mat4& transform )
{
	renderer->scene->SetNodeTransform( nodeId, transform );
}

const mat4& RenderAPI::GetNodeTransform( const int nodeId )
{
	return renderer->scene->GetNodeTransform( nodeId );
}

void RenderAPI::ResetAnimation( const int animId )
{
	renderer->scene->ResetAnimation( animId );
}

void RenderAPI::UpdateAnimation( const int animId, const float dt )
{
	renderer->scene->UpdateAnimation( animId, dt );
}

int RenderAPI::AnimationCount()
{
	return renderer->scene->AnimationCount();
}

void RenderAPI::SynchronizeSceneData()
{
	renderer->SynchronizeSceneData();
}

void RenderAPI::Render( Convergence converge, bool async )
{
	renderer->Render( renderer->scene->camera->GetView(), converge, async );
}

void RenderAPI::WaitForRender()
{
	renderer->WaitForRender();
}

Camera* RenderAPI::GetCamera()
{
	return renderer->scene->camera;
}

RenderSettings* RenderAPI::GetSettings()
{
	return &renderer->settings;
}

void RenderAPI::Setting( const char* name, const float value )
{
	renderer->Setting( name, value );
}

int RenderAPI::GetTriangleNode( const int coreInstId, const int coreTriId )
{
	return renderer->GetTriangleNode( coreInstId, coreTriId );
}

int RenderAPI::GetTriangleMesh( const int coreInstId, const int coreTriId )
{
	return renderer->GetTriangleMesh( coreInstId, coreTriId );
}

HostScene* RenderAPI::GetScene()
{
	return renderer->scene;
}

int RenderAPI::GetTriangleMaterialID( const int coreInstId, const int coreTriId )
{
	return renderer->GetTriangleMaterial( coreInstId, coreTriId );
}

HostMaterial* RenderAPI::GetTriangleMaterial( const int coreInstId, const int coreTriId )
{
	int matId = renderer->GetTriangleMaterial( coreInstId, coreTriId );
	return GetMaterial( matId );
}

HostMaterial* RenderAPI::GetMaterial( const int matId )
{
	return renderer->scene->materials[matId];
}

const std::vector<HostMaterial *> &RenderAPI::GetMaterials()
{
	return renderer->scene->materials;
}

int RenderAPI::FindMaterialID( const char* name )
{
	return renderer->scene->FindMaterialID( name );
}

int RenderAPI::FindNode( const char* name )
{
	return renderer->scene->FindNode( name );
}

int RenderAPI::AddMaterial( const float3 color, const char* name )
{
	return renderer->scene->AddMaterial( color, name );
}

int RenderAPI::AddPointLight( const float3 pos, const float3 radiance, bool enabled )
{
	return renderer->scene->AddPointLight( pos, radiance, enabled );
}

int RenderAPI::AddSpotLight( const float3 pos, const float3 direction, const float inner, const float outer, const float3 radiance, bool enabled )
{
	return renderer->scene->AddSpotLight( pos, direction, inner, outer, radiance, enabled );
}

int RenderAPI::AddDirectionalLight( const float3 direction, const float3 radiance, bool enabled )
{
	return renderer->scene->AddDirectionalLight( direction, radiance, enabled );
}

void RenderAPI::SetTarget( GLTexture* tex, const uint spp )
{
	renderer->SetTarget( tex, spp );
}

void RenderAPI::SetProbePos( const int2 pos )
{
	renderer->SetProbePos( pos );
}

CoreStats RenderAPI::GetCoreStats() const
{
	return renderer->GetCoreStats();
}

SystemStats RenderAPI::GetSystemStats()
{
	return renderer->GetSystemStats();
}

bool RenderAPI::EnableFeatureExt( const char* name ) {
	return renderer->EnableFeatureExt(name);
}

bool RenderAPI::EnableAuxTargetExt( const char* name, GLTexture *target ) {
	return renderer->EnableAuxTargetExt(name, target);
}

bool RenderAPI::DisableAuxTargetExt( const char* name ) {
	return renderer->DisableAuxTargetExt(name);
}

std::string RenderAPI::GetSettingStringExt( const char* name ) {
	return renderer->GetSettingStringExt(name);
}

bool RenderAPI::SettingStringExt( const char* name, const char* value ) {
	return renderer->SettingStringExt(name, value);
}

// EOF