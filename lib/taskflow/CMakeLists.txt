add_library(taskflow INTERFACE)

# NOTE: use C++17 or later when including this library
target_include_directories(taskflow INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})