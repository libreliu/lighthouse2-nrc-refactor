add_library(CUDA-Shared STATIC
    shared_host_code/cudatools.cpp
)
target_include_directories(CUDA-Shared INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_kernel_code
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_host_code
)
target_link_libraries(CUDA-Shared platform CUDA::cudart_static)