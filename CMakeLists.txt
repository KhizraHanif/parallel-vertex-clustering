# Adapted from https://r4nd0m6uy.ch/cmake-and-cpputest.html

cmake_minimum_required(VERSION 3.18)
project(merge-vertices LANGUAGES C CXX CUDA)
add_definitions(-DUSE_CUDA)

set(CMAKE_C_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.36.32532/bin/HostX64/x64/cl.exe")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)  # Ensure CUDA is enabled

# CUDA Flags
set(CMAKE_CUDA_ARCHITECTURES 75 86)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcompiler=-Wall -rdc=true")

if(APPLE)
SET(CMAKE_CXX_COMPILER
#"/opt/homebrew/bin/g++-11"
"/usr/local/bin/g++-13"
)
endif()

set( CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} \
    -fopenmp \
    -march=native \
    -Wall \
    -g \
    -ansi"
)

set( CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} \
    -fprofile-arcs -ftest-coverage \
     -O0" 
) 
set( CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} \
  -O3"
)
set(
  basepath
    "${CMAKE_SOURCE_DIR}"
)

# find_package( Boost 1.62.0 COMPONENTS program_options REQUIRED )

# Ensure Thrust uses CUDA backend
add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

set_source_files_properties(${CPP} PROPERTIES LANGUAGE CUDA)
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "test/*.cu" "cuda/*.cu" "cpp/*.cpp" "filePLY/*.cpp" "rply/*.c")
# Base Paths
set(basepath "${CMAKE_SOURCE_DIR}")
include_directories("${basepath}/include")
include_directories(${CMAKE_SOURCE_DIR}/include)

# include_directories( ${Boost_INCLUDE_DIR} )

include_directories(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
    "C:/Users/RBTGL/vcpkg/installed/x64-windows/include"
)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
# Vcpkg Path for Catch2
set(CMAKE_TOOLCHAIN_FILE "C:/Users/RBTGL/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
set(Catch2_DIR "C:/Users/RBTGL/vcpkg/installed/x64-windows/share/catch2")
find_package(Catch2 CONFIG REQUIRED)



# Production code
add_subdirectory(src)
add_subdirectory(rply)
add_subdirectory(filePLY)
# Add cuNSearch as a library
# Add cuNSearch as a submodule
add_subdirectory(external/cuNSearch)
target_include_directories(cuNSearch PUBLIC
    ${CMAKE_BINARY_DIR}/external/cuNSearch/cuNSearch
)

# Include its headers
include_directories(${CMAKE_SOURCE_DIR}/external/cuNSearch/include)

# Link cuNSearch to merge-vertices
target_link_libraries(merge-vertices PRIVATE cuNSearch)


# Enable Unit Tests
set_target_properties(merge-vertices PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

