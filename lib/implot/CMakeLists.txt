add_library(implot STATIC
    implot.cpp
    implot_items.cpp
    implot_demo.cpp
)

target_link_libraries(implot PUBLIC imgui)
target_include_directories(implot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})