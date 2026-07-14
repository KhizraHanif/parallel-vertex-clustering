cmake_minimum_required(VERSION 3.18)
project(merge-vertices LANGUAGES C CXX CUDA)
add_definitions(-DUSE_CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)

# Minimum sm_70 required for warp intrinsics (__match_any_sync, __activemask).
# Override with -DCMAKE_CUDA_ARCHITECTURES=XX for your specific GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcompiler=-Wall -rdc=true")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native -Wall -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(basepath "${CMAKE_SOURCE_DIR}")
add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
set_source_files_properties(${CPP} PROPERTIES LANGUAGE CUDA)

include_directories("${basepath}/include")
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/external/cuNSearch/include)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)
add_subdirectory(rply)
add_subdirectory(filePLY)
add_subdirectory(external/cuNSearch)

target_include_directories(cuNSearch PUBLIC
    ${CMAKE_BINARY_DIR}/external/cuNSearch/cuNSearch)

target_link_libraries(merge-vertices PRIVATE cuNSearch)

set_target_properties(merge-vertices PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
