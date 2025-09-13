cmake_minimum_required(VERSION 3.18)
project(HeatSim LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)

# GLAD is now included in the main executable sources

# Download GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0.9.9.8
)
FetchContent_MakeAvailable(glm)

# Download and configure ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
)

FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
    
    set(IMGUI_SOURCES
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    
    add_library(imgui STATIC ${IMGUI_SOURCES})
    target_include_directories(imgui PUBLIC 
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
endif()

set(SOURCES
    src/main.cpp
    src/core/application.cpp
    src/graphics/renderer.cpp
    src/graphics/shader.cpp
    src/ui/ui_controller.cpp
    src/utils/logger.cpp
    src/io/glad.c
    src/simulation/simulation_engine.cpp
    src/simulation/cuda_simulation_engine.cpp
    src/cuda/cuda_manager.cpp
    src/cuda/cuda_gl_interop.cpp
)

set(CUDA_SOURCES
    src/cuda/heat_kernels.cu
)

set(HEADERS
    src/core/application.h
    src/graphics/renderer.h
    src/graphics/shader.h
    src/ui/ui_controller.h
    src/utils/logger.h
    src/simulation/simulation_engine.h
    src/simulation/cuda_simulation_engine.h
    src/cuda/cuda_manager.h
    src/cuda/cuda_gl_interop.h
    src/cuda/heat_kernels.cuh
)

add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${PROJECT_NAME}
    OpenGL::GL
    glfw
    imgui
    glm::glm
    ${CUDA_LIBRARIES}
    cudart
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)