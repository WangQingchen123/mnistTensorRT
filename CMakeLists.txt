cmake_minimum_required(VERSION 3.6)

project(mnistCNN)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/include)
link_directories(C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/lib/x64)
# tensorrt
include_directories(C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/include)
link_directories(C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/lib/x64)

add_executable(mnistCNN ${PROJECT_SOURCE_DIR}/mnistCNN.cpp)
target_link_libraries(mnistCNN nvinfer)
target_link_libraries(mnistCNN cudart)

add_definitions(-O2 -pthread)

