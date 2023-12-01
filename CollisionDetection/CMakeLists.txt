cmake_minimum_required(VERSION 3.25)
project(CDetection LANGUAGES CXX CUDA)
set(TARGET_NAME main)

# find_package(CUDA REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -O0 -Wall -g -ggdb ")
set(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -O3 -Wall ")

find_package(OpenMP REQUIRED)

# 添加include文件夹路径
include_directories(inc)

# 添加系统库的include目录
include_directories(SYSTEM /usr/include/GL)

# 添加源文件
file(GLOB SOURCES
    src/cmodel.cpp
    src/collid.cpp
    src/crigid.cpp
    src/obj-viewer.cpp)

# file(GLOB GPU_SOURCES src/gpu_julia.cu)

# 添加头文件
file(GLOB HEADERS
    inc/GL

    inc/forceline.h
    inc/pair.h
    inc/real.h

    inc/quaternion.h
    inc/vec3f.h
    inc/mat3f.h
    inc/transf.h

    inc/aabb.h
    inc/box.h

    inc/tri.h
    src/crigid.h
)

# file(GLOB GPU_HEADERS include/gpu_julia.cuh include/CPUBitmap.hpp include/gl_helper.h include/book.h)

# 添加可执行文件
add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})

# add_executable(${GPU_TARGET_NAME} ${GPU_SOURCES} ${GPU_HEADERS})

# 添加链接库目录
link_directories(lib)

# 添加链接库
target_link_libraries(${TARGET_NAME} glut GL GLU OpenMP::OpenMP_CXX)

# target_link_libraries(${GPU_TARGET_NAME} PRIVATE GL GLU glut GLEW)