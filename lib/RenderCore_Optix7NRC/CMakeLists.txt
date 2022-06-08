set(RenderCore_Optix7NRC_SRCS
    "kernels/.cuda.cu"
    "core_api.cpp"
    "core_mesh.cpp"
    "rendercore.cpp"
    "../CUDA/shared_host_code/interoptexture.cpp"
)

if (OPTIX7NRC_ENABLE_ANN_NET)
    list(APPEND RenderCore_Optix7NRC_SRCS "nrcNetKNN.cpp")
endif()

add_library(RenderCore_Optix7NRC SHARED 
    ${RenderCore_Optix7NRC_SRCS}
)

target_link_libraries(RenderCore_Optix7NRC CUDA-Shared Optix7-Hdrs Shared-BSDF platform RenderSystem)
set_property(TARGET RenderCore_Optix7NRC PROPERTY CXX_STANDARD 17)
target_compile_definitions(RenderCore_Optix7NRC PRIVATE
    COREDLL_EXPORTS
    $<$<BOOL:${WIN32}>:WIN32>
    $<$<BOOL:${WIN32}>:WIN64>
    _CRT_SECURE_NO_WARNINGS
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    _CONSOLE
)

# TODO: separate debug & release
if (OPTION_CUDA_DEBUG_FULL)
    target_compile_options(RenderCore_Optix7NRC PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>
    )
    target_compile_definitions(RenderCore_Optix7NRC PUBLIC
        NRC_OPTIX_DEBUG
    )
endif()

set_target_properties(RenderCore_Optix7NRC PROPERTIES CUDA_ARCHITECTURES "75;86")
target_include_directories(RenderCore_Optix7NRC PUBLIC 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}    
)

set_property(
    TARGET RenderCore_Optix7NRC
    PROPERTY RUNTIME_OUTPUT_DIRECTORY $<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/coredlls/debug>$<$<NOT:$<CONFIG:Debug>>:${CMAKE_SOURCE_DIR}/coredlls/release>
)

option(OPTIX7NRC_ENABLE_ANN_NET "Use ANN (LGPL) to do KNN search, helps to debug nrcNet infra" ON)

if (OPTIX7NRC_ENABLE_ANN_NET)
    target_link_libraries(RenderCore_Optix7NRC ann)
endif()

# Use nvrtc if you want runtime compilation
target_link_libraries(RenderCore_Optix7NRC CUDA::nvrtc CUDA::cudart_static tiny-cuda-nn)