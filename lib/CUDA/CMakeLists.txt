add_library(CUDA-Shared INTERFACE)
target_include_directories(CUDA-Shared INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_kernel_code
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_host_code
)