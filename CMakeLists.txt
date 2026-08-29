cmake_minimum_required(VERSION 3.20)
project(warp-route LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --use_fast_math -Xptxas -v")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

find_package(CUDAToolkit REQUIRED)

# Common include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Core library: gate, naive_moe, fused_moe
add_library(warp_route STATIC
    src/gate.cu
    src/naive_moe.cu
    src/fused_moe.cu
)
target_include_directories(warp_route PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(warp_route PUBLIC CUDA::cudart)
set_target_properties(warp_route PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Correctness test
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE warp_route CUDA::cudart)
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Latency benchmark
add_executable(bench_latency benchmarks/bench_latency.cu)
target_link_libraries(bench_latency PRIVATE warp_route CUDA::cudart)
set_target_properties(bench_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable CTest
enable_testing()
add_test(NAME correctness COMMAND test_correctness)
