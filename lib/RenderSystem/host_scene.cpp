/* host_scene.cpp - Copyright 2019/2021 Utrecht University

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

#include "rendersystem.h"

// forward declaration of the PBRT scene loader functions
void PBRTInit();
void ParsePBRTScene( std::string filename );

//  +-----------------------------------------------------------------------------+
//  |  HostScene::HostScene                                                       |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostScene::HostScene()
{
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::~HostScene                                                      |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
HostScene::~HostScene()
{
	// clean up allocated objects
	for (auto mesh : meshPool) delete mesh;
	for (auto material : materials) delete material;
	for (auto texture : textures) delete texture;
	delete sky;
	delete camera;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::SerializeMaterials                                              |
//  |  Write the list of materials to a file.                               LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::SerializeMaterials( const char* xmlFile )
{
	XMLDocument doc;
	XMLNode* root = doc.NewElement( "materials" );
	doc.InsertFirstChild( root );
	int materialCount = (int)materials.size();
	((XMLElement*)root->InsertEndChild( doc.NewElement( "material_count" ) ))->SetText( materialCount );
	for (uint s = (uint)materials.size(), i = 0; i < s; i++)
	{
		// skip materials that were created at runtime
		if ((materials[i]->flags & HostMaterial::FROM_MTL) == 0) continue;
		// create a new entry for the material
		char entryName[128];
		snprintf( entryName, sizeof( entryName ), "material_%i", i );
		XMLNode* materialEntry = doc.NewElement( entryName );
		root->InsertEndChild( materialEntry );
		// store material properties
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "name" ) ))->SetText( materials[i]->name.c_str() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "origin" ) ))->SetText( materials[i]->origin.c_str() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "id" ) ))->SetText( materials[i]->ID );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "flags" ) ))->SetText( materials[i]->flags );
		XMLElement* diffuse = doc.NewElement( "color" );
		diffuse->SetAttribute( "b", materials[i]->color.value.z );
		diffuse->SetAttribute( "g", materials[i]->color.value.y );
		diffuse->SetAttribute( "r", materials[i]->color.value.x );
		(XMLElement*)materialEntry->InsertEndChild( diffuse );
		XMLElement* absorption = doc.NewElement( "absorption" );
		absorption->SetAttribute( "b", materials[i]->absorption.value.z );
		absorption->SetAttribute( "g", materials[i]->absorption.value.y );
		absorption->SetAttribute( "r", materials[i]->absorption.value.x );
		(XMLElement*)materialEntry->InsertEndChild( absorption );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "metallic" ) ))->SetText( materials[i]->metallic() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "subsurface" ) ))->SetText( materials[i]->subsurface() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "specular" ) ))->SetText( materials[i]->specular() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "roughness" ) ))->SetText( materials[i]->roughness() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "specularTint" ) ))->SetText( materials[i]->specularTint() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "anisotropic" ) ))->SetText( materials[i]->anisotropic() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "sheen" ) ))->SetText( materials[i]->sheen() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "sheenTint" ) ))->SetText( materials[i]->sheenTint() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "clearcoat" ) ))->SetText( materials[i]->clearcoat() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "clearcoatGloss" ) ))->SetText( materials[i]->clearcoatGloss() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "transmission" ) ))->SetText( materials[i]->transmission() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "eta" ) ))->SetText( materials[i]->eta() );
	}
	doc.SaveFile( xmlFile );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::DeserializeMaterials                                            |
//  |  Restore the materials from a file.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::DeserializeMaterials( const char* xmlFile )
{
	XMLDocument doc;
	XMLError result = doc.LoadFile( xmlFile );
	if (result != XML_SUCCESS) return;
	XMLNode* root = doc.FirstChild();
	if (root == nullptr) return;
	XMLElement* countElement = root->FirstChildElement( "material_count" );
	if (!countElement) return;
	const int materialCount = countElement->IntText();
	for (int i = 0; i < materialCount; i++)
	{
		// find the entry for the material
		char entryName[128];
		snprintf( entryName, sizeof( entryName ), "material_%i", i );
		XMLNode* entry = root->FirstChildElement( entryName );
		if (!entry) continue;
		// set the properties
		const char* materialName = entry->FirstChildElement( "name" )->GetText();
		int matID = HostScene::FindMaterialID( materialName );
		if (matID == -1) continue;
		HostMaterial* m /* for brevity */ = HostScene::materials[matID];
		if (entry->FirstChildElement( "flags" )) entry->FirstChildElement( "flags" )->QueryUnsignedText( &m->flags );
		XMLElement* color = entry->FirstChildElement( "color" );
		if (color)
			color->QueryFloatAttribute( "r", &m->color.value.x ),
			color->QueryFloatAttribute( "g", &m->color.value.y ),
			color->QueryFloatAttribute( "b", &m->color.value.z );
		XMLElement* absorption = entry->FirstChildElement( "absorption" );
		if (absorption)
			absorption->QueryFloatAttribute( "r", &m->absorption.value.x ),
			absorption->QueryFloatAttribute( "g", &m->absorption.value.y ),
			absorption->QueryFloatAttribute( "b", &m->absorption.value.z );
		if (entry->FirstChildElement( "metallic" )) entry->FirstChildElement( "metallic" )->QueryFloatText( &m->metallic() );
		if (entry->FirstChildElement( "subsurface" )) entry->FirstChildElement( "subsurface" )->QueryFloatText( &m->subsurface() );
		if (entry->FirstChildElement( "specular" )) entry->FirstChildElement( "specular" )->QueryFloatText( &m->specular() );
		if (entry->FirstChildElement( "roughness" )) entry->FirstChildElement( "roughness" )->QueryFloatText( &m->roughness() );
		if (entry->FirstChildElement( "specularTint" )) entry->FirstChildElement( "specularTint" )->QueryFloatText( &m->specularTint() );
		if (entry->FirstChildElement( "anisotropic" )) entry->FirstChildElement( "anisotropic" )->QueryFloatText( &m->anisotropic() );
		if (entry->FirstChildElement( "sheen" )) entry->FirstChildElement( "sheen" )->QueryFloatText( &m->sheen() );
		if (entry->FirstChildElement( "sheenTint" )) entry->FirstChildElement( "sheenTint" )->QueryFloatText( &m->sheenTint() );
		if (entry->FirstChildElement( "clearcoat" )) entry->FirstChildElement( "clearcoat" )->QueryFloatText( &m->clearcoat() );
		if (entry->FirstChildElement( "clearcoatGloss" )) entry->FirstChildElement( "clearcoatGloss" )->QueryFloatText( &m->clearcoatGloss() );
		if (entry->FirstChildElement( "transmission" )) entry->FirstChildElement( "transmission" )->QueryFloatText( &m->transmission() );
		if (entry->FirstChildElement( "eta" )) entry->FirstChildElement( "eta" )->QueryFloatText( &m->eta() );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::Init                                                            |
//  |  Prepare scene geometry for rendering.                                LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::Init()
{
	// initialize the camera
	camera = new Camera();
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::SetSkyDome                                                      |
//  |  Set the skydome used for this scene.                                 LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::SetSkyDome( HostSkyDome* skydome )
{
	sky = skydome;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddMesh                                                         |
//  |  Add an existing HostMesh to the list of meshes and return the mesh ID.     |
//  |                                                                       LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddMesh( HostMesh* mesh )
{
	// see if the mesh is already in the scene
	for( int s = (int)meshPool.size(), i = 0; i < s; i++ )
	{
		if (meshPool[i] == mesh)
		{
			assert( mesh->ID == i );
			return i;
		}
	}
	// add the mesh
	mesh->ID = (int)meshPool.size();
	meshPool.push_back( mesh );
	return mesh->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddMesh                                                         |
//  |  Create a mesh specified by a file name and data dir, apply a scale, add    |
//  |  the mesh to the list of meshes and return the mesh ID.               LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddMesh( const char* objFile, const float scale, const bool flatShaded )
{
	// extract directory from specified file name
	char* tmp = new char[strlen( objFile ) + 1];
	memcpy( tmp, objFile, strlen( objFile ) + 1 );
	char* lastSlash = tmp, * pos = tmp;
	while (*pos) { if (*pos == '/' || *pos == '\\') lastSlash = pos; pos++; }
	*lastSlash = 0;
	return AddMesh( lastSlash + 1, tmp, scale, flatShaded );
}
int HostScene::AddMesh( const char* objFile, const char* dir, const float scale, const bool flatShaded )
{
	HostMesh* newMesh = new HostMesh( objFile, dir, scale, flatShaded );
	return AddMesh( newMesh );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddMesh                                                         |
//  |  Create a mesh with the specified amount of triangles without actually      |
//  |  setting the triangles. These are expected to be set via the AddTriToMesh   |
//  |  function.                                                            LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddMesh( const int triCount )
{
	HostMesh* newMesh = new HostMesh( triCount );
	return AddMesh( newMesh );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddTriToMesh                                                    |
//  |  Add a single triangle to a mesh.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::AddTriToMesh( const int meshId, const float3& v0, const float3& v1, const float3& v2, const int matId )
{
	HostMesh* m = HostScene::meshPool[meshId];
	m->vertices.push_back( make_float4( v0, 1 ) );
	m->vertices.push_back( make_float4( v1, 1 ) );
	m->vertices.push_back( make_float4( v2, 1 ) );
	HostTri tri;
	tri.material = matId;
	float3 N = normalize( cross( v1 - v0, v2 - v0 ) );
	tri.vN0 = tri.vN1 = tri.vN2 = N;
	tri.Nx = N.x, tri.Ny = N.y, tri.Nz = N.z;
	tri.vertex0 = v0;
	tri.vertex1 = v1;
	tri.vertex2 = v2;
	m->triangles.push_back( tri );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddScene                                                        |
//  |  Loads a collection of meshes from a gltf file. An instance and a scene     |
//  |  graph node is created for each mesh.                                 LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddScene( const char* sceneFile, const mat4& transform )
{
	// extract directory from specified file name
	char* tmp = new char[strlen( sceneFile ) + 1];
	memcpy( tmp, sceneFile, strlen( sceneFile ) + 1 );
	char* lastSlash = tmp, * pos = tmp;
	while (*pos) { if (*pos == '/' || *pos == '\\') lastSlash = pos; pos++; }
	*lastSlash = 0;
	int retVal = 0;
	if (strstr( lastSlash + 1, ".pbrt" ))
	{
		// load a .pbrt scene
		PBRTInit();
		ParsePBRTScene( sceneFile );
	}
	else
	{
		// not a .pbrt file; must be a .gltf file
		retVal = AddScene( lastSlash + 1, tmp, transform );
	}
	delete tmp;
	return retVal;
}
int HostScene::AddScene( const char* sceneFile, const char* dir, const mat4& transform )
{
	// offsets: if we loaded an object before this one, indices should not start at 0.
	// based on https://github.com/SaschaWillems/Vulkan-glTF-PBR/blob/master/base/VulkanglTFModel.hpp
	const int meshBase = (int)meshPool.size();
	const int skinBase = (int)skins.size();
	const int retVal = (int)nodePool.size();
	const int nodeBase = (int)nodePool.size() + 1;
	// load gltf file
	string cleanFileName = string( dir ) + (dir[strlen( dir ) - 1] == '/' ? "" : "/") + string( sceneFile );
	tinygltf::Model gltfModel;
	tinygltf::TinyGLTF loader;
	string err, warn;
	bool ret = false;
	if (cleanFileName.size() > 4)
	{
		string extension4 = cleanFileName.substr( cleanFileName.size() - 5, 5 );
		string extension3 = cleanFileName.substr( cleanFileName.size() - 4, 4 );
		if (extension4.compare( ".gltf" ) == 0)
			ret = loader.LoadASCIIFromFile( &gltfModel, &err, &warn, cleanFileName.c_str() );
		else if (extension3.compare( ".bin" ) == 0 || extension3.compare( ".glb" ) == 0)
			ret = loader.LoadBinaryFromFile( &gltfModel, &err, &warn, cleanFileName.c_str() );
	}
	if (!warn.empty()) printf( "Warn: %s\n", warn.c_str() );
	if (!err.empty()) printf( "Err: %s\n", err.c_str() );
	FATALERROR_IF( !ret, "could not load glTF file:\n%s", cleanFileName.c_str() );
	// convert textures
	vector<int> texIdx;
	for (size_t s = gltfModel.textures.size(), i = 0; i < s; i++)
	{
		char t[1024];
		sprintf_s( t, "%s-%s-%03i", dir, sceneFile, (int)i );
		int textureID = FindTextureID( t );
		if (textureID != -1)
		{
			// push id of existing texture
			texIdx.push_back( textureID );
		}
		else
		{
			// create new texture
			tinygltf::Texture& gltfTexture = gltfModel.textures[i];
			HostTexture* texture = new HostTexture();
			const tinygltf::Image& image = gltfModel.images[gltfTexture.source];
			const size_t size = image.component * image.width * image.height;
			texture->name = t;
			texture->width = image.width;
			texture->height = image.height;
			texture->idata = (uchar4*)MALLOC64( texture->PixelsNeeded( image.width, image.height, MIPLEVELCOUNT ) * sizeof( uint ) );
			texture->ID = (uint)textures.size();
			texture->flags |= HostTexture::LDR;
			memcpy( texture->idata, image.image.data(), size );
			texture->ConstructMIPmaps();
			textures.push_back( texture );
			texIdx.push_back( texture->ID );
		}
	}
	// convert materials
	vector<int> matIdx;
	for (size_t s = gltfModel.materials.size(), i = 0; i < s; i++)
	{
		char t[1024];
		sprintf_s( t, "%s-%s-%03i", dir, sceneFile, (int)i );
		int matID = FindMaterialIDByOrigin( t );
		if (matID != -1)
		{
			// material already exists; reuse
			matIdx.push_back( matID );
		}
		else
		{
			// create new material
			tinygltf::Material& gltfMaterial = gltfModel.materials[i];
			HostMaterial* material = new HostMaterial();
			material->ID = (int)materials.size();
			material->origin = t;
			material->ConvertFrom( gltfMaterial, gltfModel, texIdx );
			material->flags |= HostMaterial::FROM_MTL;
			materials.push_back( material );
			matIdx.push_back( material->ID );
			// materialList.push_back( material->ID ); // can't do that, need something smarter.
		}
	}
	// convert meshes
	for (size_t s = gltfModel.meshes.size(), i = 0; i < s; i++)
	{
		tinygltf::Mesh& gltfMesh = gltfModel.meshes[i];
		HostMesh* newMesh = new HostMesh( gltfMesh, gltfModel, matIdx, gltfModel.materials.size() == 0 ? 0 : -1 );
		newMesh->ID = (int)i + meshBase;
		meshPool.push_back( newMesh );
	}
	// convert khr lights
	// khr lights array idx -> (punctualLightType, punctualLightID)
	std::map<int, std::pair<int, int>> khrLightsMap;
	auto khrLightsExt = gltfModel.extensions.find("KHR_lights_punctual");
	if (khrLightsExt != gltfModel.extensions.end()) {
		auto &khrLightsArr = khrLightsExt->second.Get("lights").Get<tinygltf::Value::Array>();

		for (int khrLightID = 0; khrLightID < khrLightsArr.size(); khrLightID++) {
			auto &light = khrLightsArr[khrLightID];
			if (light.Get("type").Get<std::string>() == "point") {
				float intensity = (float) light.Get("intensity").GetNumberAsDouble();
				auto &colorArray = light.Get("color").Get<tinygltf::Value::Array>();

				float3 color = make_float3(
					(float)colorArray[0].GetNumberAsDouble(),
					(float)colorArray[1].GetNumberAsDouble(),
					(float)colorArray[2].GetNumberAsDouble()
				);

				// pos: to be patched inside host_node.cpp
				int pointLightID = AddPointLight(make_float3(0), color * intensity);
				printf("[SceneDebug] added khr point light %d, pointLight ID = %d\n",
					khrLightID, pointLightID
				);

				khrLightsMap[khrLightID] = std::make_pair(
					1,
					pointLightID
				);
			}
		}
	}

	// push an extra node that holds a transform for the gltf scene
	HostNode* newNode = new HostNode();
	newNode->localTransform = transform;
	newNode->ID = nodeBase - 1;
	nodePool.push_back( newNode );
	// convert nodes
	for (size_t s = gltfModel.nodes.size(), i = 0; i < s; i++)
	{
		tinygltf::Node& gltfNode = gltfModel.nodes[i];
		HostNode* newNode = new HostNode( gltfNode, nodeBase, meshBase, skinBase, khrLightsMap );
		newNode->ID = (int)nodePool.size();
		nodePool.push_back( newNode );
	}
	// convert animations and skins
	for (tinygltf::Animation& gltfAnim : gltfModel.animations)
	{
		HostAnimation* anim = new HostAnimation( gltfAnim, gltfModel, nodeBase );
		animations.push_back( anim );
	}
	for (tinygltf::Skin& source : gltfModel.skins)
	{
		HostSkin* newSkin = new HostSkin( source, gltfModel, nodeBase );
		skins.push_back( newSkin );
	}
	// construct a scene graph for scene 0, assuming the GLTF file has one scene
	tinygltf::Scene& glftScene = gltfModel.scenes[0];
	// add the root nodes to the scene transform node
	for (size_t i = 0; i < glftScene.nodes.size(); i++) nodePool[nodeBase - 1]->childIdx.push_back( glftScene.nodes[i] + nodeBase );
	// add the root transform to the scene
	rootNodes.push_back( nodeBase - 1 );
	// return index of first created node
	return retVal;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddQuad                                                         |
//  |  Create a mesh that consists of two triangles, described by a normal, a     |
//  |  centroid position and a material. Typically used to add an area light      |
//  |  to a scene.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddQuad( float3 N, const float3 pos, const float width, const float height, const int matId, const int meshID )
{
	HostMesh* newMesh = meshID > -1 ? meshPool[meshID] : new HostMesh();
	N = normalize( N ); // let's not assume the normal is normalized.
#if 1
	const float3 tmp = fabs( N.x ) > 0.9f ? make_float3( 0, 1, 0 ) : make_float3( 1, 0, 0 );
	const float3 T = 0.5f * width * normalize( cross( N, tmp ) );
	const float3 B = 0.5f * height * normalize( cross( normalize( T ), N ) );
#else
	// "Building an Orthonormal Basis, Revisited"
	const float sign = copysignf( 1.0f, N.z ), a = -1.0f / (sign + N.z), b = N.x * N.y * a;
	const float3 B = 0.5f * width * make_float3( 1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x );
	const float3 T = 0.5f * height * make_float3( b, sign + N.y * N.y * a, -N.y );
#endif
	// calculate corners
	uint vertBase = (uint)newMesh->vertices.size();
	newMesh->vertices.push_back( make_float4( pos - B - T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos + B - T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos - B + T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos + B - T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos + B + T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos - B + T, 1 ) );
	// triangles
	HostTri tri1, tri2;
	tri1.material = tri2.material = matId;
	tri1.vN0 = tri1.vN1 = tri1.vN2 = N;
	tri2.vN0 = tri2.vN1 = tri2.vN2 = N;
	tri1.Nx = N.x, tri1.Ny = N.y, tri1.Nz = N.z;
	tri2.Nx = N.x, tri2.Ny = N.y, tri2.Nz = N.z;
	tri1.u0 = tri1.u1 = tri1.u2 = tri1.v0 = tri1.v1 = tri1.v2 = 0;
	tri2.u0 = tri2.u1 = tri2.u2 = tri2.v0 = tri2.v1 = tri2.v2 = 0;
	tri1.vertex0 = make_float3( newMesh->vertices[vertBase + 0] );
	tri1.vertex1 = make_float3( newMesh->vertices[vertBase + 1] );
	tri1.vertex2 = make_float3( newMesh->vertices[vertBase + 2] );
	tri2.vertex0 = make_float3( newMesh->vertices[vertBase + 3] );
	tri2.vertex1 = make_float3( newMesh->vertices[vertBase + 4] );
	tri2.vertex2 = make_float3( newMesh->vertices[vertBase + 5] );
	tri1.T = tri2.T = T / (0.5 * height);
	tri1.B = tri2.B = B / (0.5 * width);
	newMesh->triangles.push_back( tri1 );
	newMesh->triangles.push_back( tri2 );
	// if the mesh was newly created, add it to scene mesh list
	if (meshID == -1)
	{
		newMesh->ID = (int)meshPool.size();
		newMesh->materialList.push_back( matId );
		meshPool.push_back( newMesh );
	}
	return newMesh->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddInstance                                                     |
//  |  Add an instance of an existing mesh to the scene.                    LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddInstance( HostNode* newNode )
{
	if (nodeListHoles > 0)
	{
		// we have holes in the nodes vector due to instance deletions; search from the
		// end of the list to speed up frequent additions / deletions in complex scenes.
		for (int i = (int)nodePool.size() - 1; i >= 0; i--) if (nodePool[i] == 0)
		{
			// overwrite an empty slot, created by deleting an instance
			nodePool[i] = newNode;
			newNode->ID = i;
			rootNodes.push_back( i );
			nodeListHoles--; // plugged one hole.
			return i;
		}
	}
	// no empty slots available or found; make sure we don't look for them again.
	nodeListHoles = 0;
	// insert the new node at the end of the list
	newNode->ID = (int)nodePool.size();
	nodePool.push_back( newNode );
	rootNodes.push_back( newNode->ID );
	return newNode->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddInstance                                                     |
//  |  Add an instance of an existing mesh to the scene.                    LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddInstance( const int meshId, const mat4& transform )
{
	HostNode* newNode = new HostNode( meshId, transform );
	return AddInstance( newNode );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::RemoveNode                                                      |
//  |  Remove a node from the scene.                                              |
//  |  This also removes the node from the rootNodes vector. Note that will only  |
//  |  work correctly if the node is not part of a hierarchy. This assumption is  |
//  |  valid for nodes that have been created using AddInstance.                  |
//  |  See the notes at the top of host_scene.h for the relation between host     |
//  |  nodes and core instances.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::RemoveNode( const int nodeId )
{
	// remove the instance from the scene graph
	for (int s = (int)rootNodes.size(), i = 0; i < s; i++) if (rootNodes[i] == nodeId)
	{
		rootNodes[i] = rootNodes[s - 1];
		rootNodes.pop_back();
		break;
	}
	// delete the instance
	HostNode* node = nodePool[nodeId];
	nodePool[nodeId] = 0; // safe; we only access the nodes vector indirectly.
	delete node;
	nodeListHoles++; // HostScene::AddInstance will fill up holes first.
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindTextureID                                                   |
//  |  Return a texture ID if it already exists.                            LH2'20|
//  +-----------------------------------------------------------------------------+
int HostScene::FindTextureID( const char* name )
{
	for (auto texture : textures) if (strcmp( texture->name.c_str(), name ) == 0) return texture->ID;
	return -1;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindOrCreateTexture                                             |
//  |  Return a texture: if it already exists, return the existing texture (after |
//  |  increasing its refCount), otherwise, create a new texture and return its   |
//  |  ID.                                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::FindOrCreateTexture( const string& origin, const uint modFlags )
{
	// search list for existing texture
	for (auto texture : textures) if (texture->Equals( origin, modFlags ))
	{
		texture->refCount++;
		return texture->ID;
	}
	// nothing found, create a new texture
	return CreateTexture( origin, modFlags );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindOrCreateMaterial                                            |
//  |  Return a material: if it already exists, return the existing material      |
//  |  (after increasing its refCount), otherwise, create a new texture and       |
//  |  return its ID.                                                       LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::FindOrCreateMaterial( const string& name )
{
	// search list for existing texture
	for (auto material : materials) if (material->name.compare( name ) == 0)
	{
		material->refCount++;
		return material->ID;
	}
	// nothing found, create a new texture
	const int newID = AddMaterial( make_float3( 0 ) );
	materials[newID]->name = name;
	return newID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindOrCreateMaterialCopy                                        |
//  |  Create an untextured material, based on an existing material. This copy is |
//  |  to be used for a triangle that only reads a single texel from a texture;   |
//  |  using a single color is more efficient.                              LH2'20|
//  +-----------------------------------------------------------------------------+
int HostScene::FindOrCreateMaterialCopy( const int matID, const uint color )
{
	// search list for existing material copy
	const int r = (color >> 16) & 255, g = (color >> 8) & 255, b = color & 255;
	const float3 c = make_float3( b * (1.0f / 255.0f), g * (1.0f / 255.0f), r * (1.0f / 255.0f) );
	for (auto material : materials)
	{
		if (material->flags & HostMaterial::SINGLE_COLOR_COPY &&
			material->color.value.x == c.x && material->color.value.y == c.y && material->color.value.z == c.z)
		{
			material->refCount++;
			return material->ID;
		}
	}
	// nothing found, create a new material copy
	const int newID = AddMaterial( make_float3( 0 ) );
	*materials[newID] = *materials[matID];
	materials[newID]->color.textureID = -1;
	materials[newID]->color.value = c;
	materials[newID]->flags |= HostMaterial::SINGLE_COLOR_COPY;
	materials[newID]->ID = newID;
	char t[256];
	sprintf( t, "copied_mat_%i", newID );
	materials[newID]->name = t;
	return newID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindMaterialID                                                  |
//  |  Find the ID of a material with the specified name.                   LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::FindMaterialID( const char* name )
{
	for (auto material : materials) if (material->name.compare( name ) == 0) return material->ID;
	return -1;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindMaterialIDByOrigin                                          |
//  |  Find the ID of a material with the specified origin.                 LH2'20|
//  +-----------------------------------------------------------------------------+
int HostScene::FindMaterialIDByOrigin( const char* name )
{
	for (auto material : materials) if (material->origin.compare( name ) == 0) return material->ID;
	return -1;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindNextMaterialID                                              |
//  |  Find the ID of a material with the specified name, with an ID greater than |
//  |  the specified one. Used to find materials with the same name.        LH2'20|
//  +-----------------------------------------------------------------------------+
int HostScene::FindNextMaterialID( const char* name, const int matID )
{
	for (int s = (int)materials.size(), i = matID + 1; i < s; i++)
		if (materials[i]->name.compare( name ) == 0) return materials[i]->ID;
	return -1;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindNode                                                        |
//  |  Find the ID of a node with the specified name.                       LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::FindNode( const char* name )
{
	for (auto node : nodePool) if (node->name.compare( name ) == 0) return node->ID;
	return -1;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::SetNodeTransform                                                |
//  |  Set the local transform for the specified node.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::SetNodeTransform( const int nodeId, const mat4& transform )
{
	if (nodeId < 0 || nodeId >= nodePool.size()) return;
	nodePool[nodeId]->localTransform = transform;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::GetNodeTransform                                                |
//  |  Set the local transform for the specified node.                      LH2'19|
//  +-----------------------------------------------------------------------------+
const mat4& HostScene::GetNodeTransform( const int nodeId )
{
	static mat4 dummyIdentity;
	if (nodeId < 0 || nodeId >= nodePool.size()) return dummyIdentity;
	return nodePool[nodeId]->localTransform;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::ResetAnimation                                                  |
//  |  Reset the indicated animation.                                       LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::ResetAnimation( const int animId )
{
	if (animId < 0 || animId >= animations.size()) return;
	animations[animId]->Reset();
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::ResetAnimation                                                  |
//  |  Update the indicated animation.                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::UpdateAnimation( const int animId, const float dt )
{
	if (animId < 0 || animId >= animations.size()) return;
	animations[animId]->Update( dt );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::CreateTexture                                                   |
//  |  Return a texture. Create it anew, even if a texture with the same origin   |
//  |  already exists.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::CreateTexture( const string& origin, const uint modFlags )
{
	// create a new texture
	HostTexture* newTexture = new HostTexture( origin.c_str(), modFlags );
	textures.push_back( newTexture );
	return newTexture->ID = (int)textures.size() - 1;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddMaterial                                                     |
//  |  Adds an existing HostMaterial* and returns the ID. If the material         |
//  |  with that pointer is already added, it is not added again.           LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddMaterial( HostMaterial* material )
{
	auto res = std::find( materials.begin(), materials.end(), material );
	if (res != materials.end()) return std::distance( materials.begin(), res );
	int matid = (int)materials.size();
	materials.push_back( material );
	return matid;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddMaterial                                                     |
//  |  Create a material, with a limited set of parameters.                 LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddMaterial( const float3 color, const char* name )
{
	HostMaterial* material = new HostMaterial();
	material->color = color;
	if (name) material->name = name;
	return material->ID = AddMaterial( material );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddPointLight                                                   |
//  |  Create a point light and add it to the scene.                        LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddPointLight( const float3 pos, const float3 radiance, bool enabled )
{
	HostPointLight* light = new HostPointLight();
	light->position = pos;
	light->radiance = radiance;
	light->enabled = enabled;
	light->ID = (int)pointLights.size();
	pointLights.push_back( light );
	return light->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddSpotLight                                                    |
//  |  Create a spot light and add it to the scene.                         LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddSpotLight( const float3 pos, const float3 direction, const float inner, const float outer, const float3 radiance, bool enabled )
{
	HostSpotLight* light = new HostSpotLight();
	light->position = pos;
	light->direction = direction;
	light->radiance = radiance;
	light->cosInner = inner;
	light->cosOuter = outer;
	light->enabled = enabled;
	light->ID = (int)spotLights.size();
	spotLights.push_back( light );
	return light->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddDirectionalLight                                             |
//  |  Create a directional light and add it to the scene.                  LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddDirectionalLight( const float3 direction, const float3 radiance, bool enabled )
{
	HostDirectionalLight* light = new HostDirectionalLight();
	light->direction = direction;
	light->radiance = radiance;
	light->enabled = enabled;
	light->ID = (int)directionalLights.size();
	directionalLights.push_back( light );
	return light->ID;
}

// EOF