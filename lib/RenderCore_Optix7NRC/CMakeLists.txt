add_library(RenderCore_Optix7NRC SHARED 
    "kernels/.cuda.cu"
    "core_api.cpp"
    "core_mesh.cpp"
    "rendercore.cpp"
    "../CUDA/shared_host_code/interoptexture.cpp"
)

target_link_libraries(RenderCore_Optix7NRC CUDA-Shared Optix7-Hdrs Shared-BSDF platform RenderSystem)
set_property(TARGET RenderCore_Optix7NRC PROPERTY CXX_STANDARD 17)
target_compile_definitions(RenderCore_Optix7NRC PRIVATE
    COREDLL_EXPORTS
    WIN32
    WIN64
    _CRT_SECURE_NO_WARNINGS
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    _CONSOLE
)
set_target_properties(RenderCore_Optix7NRC PROPERTIES CUDA_ARCHITECTURES "75")
target_include_directories(RenderCore_Optix7NRC PUBLIC 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}    
)
set_property(TARGET RenderCore_Optix7NRC PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/coredlls/>)

# Use this if you want runtime compilation
target_link_libraries(RenderCore_Optix7NRC nvrtc cudart_static)