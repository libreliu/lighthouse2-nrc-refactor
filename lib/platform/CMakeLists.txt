add_library(platform STATIC
    "platform.cpp"
    "system.cpp"
)

target_include_directories(platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET platform PROPERTY CXX_STANDARD 17)
target_compile_definitions(platform PRIVATE
    $<$<BOOL:${WIN32}>:WIN32>
    $<$<BOOL:${WIN32}>:WIN64>
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    _CONSOLE
)

target_link_libraries(platform Common-Libs glad glfw zlibstatic FreeImage taskflow half2 freetype)