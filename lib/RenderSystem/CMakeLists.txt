add_library(RenderSystem STATIC
    "camera.cpp"
    "core_api_base.cpp"
    "host_anim.cpp"
    "host_light.cpp"
    "host_material.cpp"
    "host_mesh.cpp"
    "host_meshloaders.cpp"
    "host_node.cpp"
    "host_scene.cpp"
    "host_skydome.cpp"
    "host_texture.cpp"
    "render_api.cpp"
    "rendersystem.cpp"
    "materials/pbrt/api.cpp"
    "materials/pbrt/paramset.cpp"
    "materials/pbrt/pbrtparser.cpp"
    "materials/pbrt/plymesh.cpp"
    "materials/pbrt/spectrum.cpp"
)

target_include_directories(RenderSystem PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/materials/pbrt
)

target_compile_definitions(RenderSystem PRIVATE
    RENDERSYSTEMBUILD
    PBRT_IS_WINDOWS
    WIN32
    WIN64
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    _CONSOLE
)

set_property(TARGET RenderSystem PROPERTY CXX_STANDARD 17)
target_link_libraries(RenderSystem platform tinyxml2 tinygltf tinyobjloader)