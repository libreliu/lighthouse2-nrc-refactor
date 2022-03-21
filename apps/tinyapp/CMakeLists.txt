add_executable(tinyapp
    "main.cpp"
)

target_link_libraries(tinyapp Common-Libs platform RenderSystem RenderCore_Optix7)
set_property(TARGET tinyapp PROPERTY CXX_STANDARD 17)

# Use generator expression to surpress the "Debug/" & "Release/" behaviour on multi-target build on VS
set_property(TARGET tinyapp PROPERTY RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_SOURCE_DIR}>)
set_property(TARGET tinyapp PROPERTY DEBUG_POSTFIX "d")