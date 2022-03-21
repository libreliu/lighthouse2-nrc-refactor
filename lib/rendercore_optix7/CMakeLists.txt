add_library(RenderCore_Optix7 STATIC 
    "kernels/.cuda.cu"
    "core_api.cpp"
    "core_mesh.cpp"
    "rendercore.cpp"
)

target_link_libraries(RenderCore_Optix7 CUDA-Shared Optix7-Hdrs Shared-BSDF platform RenderSystem)
set_property(TARGET RenderCore_Optix7 PROPERTY CXX_STANDARD 17)
target_compile_definitions(RenderCore_Optix7 PRIVATE
    COREDLL_EXPORTS
    WIN32
    WIN64
    _CRT_SECURE_NO_WARNINGS
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    _CONSOLE
)
set_target_properties(RenderCore_Optix7 PROPERTIES CUDA_ARCHITECTURES "75")