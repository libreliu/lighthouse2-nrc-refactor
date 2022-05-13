# NOTE: ANN uses GNU LGPL; TODO: use SHARED to satisfy constraints

add_library(ann STATIC
    src/ANN.cpp
    src/brute.cpp
    src/kd_tree.cpp
    src/kd_util.cpp
    src/kd_split.cpp
    src/kd_dump.cpp
    src/kd_search.cpp
    src/kd_pr_search.cpp
    src/kd_fix_rad_search.cpp
    src/bd_tree.cpp
    src/bd_search.cpp
    src/bd_pr_search.cpp
    src/bd_fix_rad_search.cpp
    src/perf.cpp
)

target_include_directories(ann PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (WIN32)
    target_compile_definitions(ann PUBLIC WIN32 DLL_EXPORTS)
endif()

add_executable(ann-sample sample/ann_sample.cpp)
target_link_libraries(ann-sample ann)


add_custom_command(TARGET ann-sample POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/sample/data.pts $<TARGET_FILE_DIR:ann-sample>/
)
add_custom_command(TARGET ann-sample POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/sample/query.pts $<TARGET_FILE_DIR:ann-sample>/
)