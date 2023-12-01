cmake_minimum_required(VERSION 3.25)
project(julia LANGUAGES CXX CUDA)
set(CPU_TARGET_NAME cmain)
set(GPU_TARGET_NAME gmain)
find_package(CUDA REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -O0 -Wall -g -ggdb ")
set(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -O3 -Wall ")

# 添加include文件夹路径
include_directories(include)

# 添加源文件
file(GLOB CPU_SOURCES src/cpu_julia.cpp)
file(GLOB GPU_SOURCES src/gpu_julia.cu)

# 添加头文件
file(GLOB CPU_HEADERS include/cpu_julia.hpp include/CPUBitmap.hpp include/gl_helper.h)
file(GLOB GPU_HEADERS include/gpu_julia.cuh include/CPUBitmap.hpp include/gl_helper.h include/book.h)

# nvcc -o julia src/gpu_julia.cu -Iinclude -lGL -lGLU -lglut -lGLE

# 添加可执行文件
add_executable(${CPU_TARGET_NAME} ${CPU_SOURCES} ${CPU_HEADERS})
add_executable(${GPU_TARGET_NAME} ${GPU_SOURCES} ${GPU_HEADERS})

# 添加链接库
target_link_libraries(${CPU_TARGET_NAME} PRIVATE GL GLU glut GLEW)
target_link_libraries(${GPU_TARGET_NAME} PRIVATE GL GLU glut GLEW)