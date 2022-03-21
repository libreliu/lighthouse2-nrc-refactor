add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/rapidjson    
)