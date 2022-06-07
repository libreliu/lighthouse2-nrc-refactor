cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(lighthouse2 LANGUAGES C CXX CUDA)

option(OPTION_CUDA_DEBUG_FULL "Enable -G for all cuda codes" ON)
find_package(CUDAToolkit REQUIRED CONFIG HINTS "etc/")

# Used for zlib to disable cmake installs 
set(SKIP_INSTALL_ALL ON)

add_subdirectory(lib)
add_subdirectory(apps)

# -- The following only works for Windows as for now --
if (WIN32)

set(CPACK_PACKAGE_NAME "lh2-nrc-bundled")
set(CPACK_PACKAGE_VENDOR "libreliu")
set(CPACK_GENERATOR "7Z")
include(CPack)


# common
install(DIRECTORY "${CMAKE_SOURCE_DIR}/apps/_shareddata" DESTINATION "apps/")

# nrcapp
install(TARGETS nrcapp DESTINATION "apps/nrcapp")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/apps/nrcapp/data" DESTINATION "apps/nrcapp/")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/apps/nrcapp/shaders" DESTINATION "apps/nrcapp/")
install(FILES "${CMAKE_SOURCE_DIR}/apps/nrcapp/camera.xml" DESTINATION "apps/nrcapp/")
install(FILES "${CMAKE_SOURCE_DIR}/apps/nrcapp/imgui.ini" DESTINATION "apps/nrcapp/")

# optix7nrc
install(TARGETS RenderCore_Optix7NRC DESTINATION "apps/nrcapp")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/lib/RenderCore_Optix7NRC" DESTINATION "lib/")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/lib/CUDA" DESTINATION "lib/")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/lib/RenderSystem" DESTINATION "lib/")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/lib/Optix7" DESTINATION "lib/")

endif()