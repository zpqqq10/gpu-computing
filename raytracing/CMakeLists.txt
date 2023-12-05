cmake_minimum_required(VERSION 3.25)
project(raytracing LANGUAGES CXX CUDA)
set(GPU_TARGET_NAME main)
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -O0 -Wall -g -ggdb ")
set(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -O3 -Wall ")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} )

# 添加include文件夹路径
include_directories(include)

# 添加源文件
file(GLOB GPU_SOURCES src/rt.cu)
# set(GPU_SOURCES
#     src/rt.cu
# )

# 添加头文件
file(GLOB GPU_HEADERS
    include/gl_helper.h
    include/CPUBitmap.hpp
    include/book.h
    include/cpu_anim.h
    include/gpu_anim.h
    # include/rt.cuh
    )

# 添加可执行文件
cuda_add_executable(${GPU_TARGET_NAME} ${GPU_SOURCES} ${GPU_HEADERS})

# nvcc src/rt.cu -Iinclude -lGL -lGLU -lglut -lGLEW

# 添加链接库
target_link_libraries(${GPU_TARGET_NAME} GL GLU glut GLEW)
