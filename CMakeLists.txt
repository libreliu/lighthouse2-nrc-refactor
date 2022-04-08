cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lighthouse2 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit)

add_subdirectory(lib)
add_subdirectory(apps)