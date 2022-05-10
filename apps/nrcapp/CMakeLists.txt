add_executable(nrcapp
    "main.cpp"
)

target_link_libraries(nrcapp Common-Libs platform RenderSystem RenderCore_Optix7NRC imgui implot)
set_property(TARGET nrcapp PROPERTY CXX_STANDARD 17)

# Use generator expression to surpress the "Debug/" & "Release/" behaviour on multi-target build on VS
set_property(TARGET nrcapp PROPERTY RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_SOURCE_DIR}>)
set_property(TARGET nrcapp PROPERTY DEBUG_POSTFIX "d")