add_executable(tinyapp
    "main.cpp"
)

target_link_libraries(tinyapp Common-Libs platform RenderSystem RenderCore_Optix7)
set_property(TARGET tinyapp PROPERTY CXX_STANDARD 17)