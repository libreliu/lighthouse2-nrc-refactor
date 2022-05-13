add_library(Common-Libs INTERFACE)

target_include_directories(Common-Libs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# For tiny-cuda-nn
set(ENV{TCNN_CUDA_ARCHITECTURES} "75")

# Specific target libraries to build
add_subdirectory(ann)
add_subdirectory(tinyxml2)
add_subdirectory(tinygltf)
add_subdirectory(tinyobjloader)
add_subdirectory(half2.2.0)
add_subdirectory(taskflow)
add_subdirectory(zlib)
add_subdirectory(FreeImage)
add_subdirectory(FreeType2)
add_subdirectory(glad)
add_subdirectory(GLFW)
add_subdirectory(imgui)
add_subdirectory(implot)
add_subdirectory(OptiX7)
add_subdirectory(sharedBSDFs)
add_subdirectory(CUDA)
add_subdirectory(tiny-cuda-nn)
add_subdirectory(platform)
add_subdirectory(RenderSystem)
add_subdirectory(rendercore_optix7)
add_subdirectory(RenderCore_Optix7NRC)