enable_language(CUDA)
file(GLOB CPP "../cpp/*.cpp")
set(GPU_SRC ../cpp/PWeldCuda.cu)
add_executable(
  merge-vertices
    main.cpp
    ${CPP}
    ${GPU_SRC}
)

set_target_properties(
  merge-vertices
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "../lib/"
      LIBRARY_OUTPUT_DIRECTORY "../lib/"
      RUNTIME_OUTPUT_DIRECTORY "../"
)

target_link_libraries(
  merge-vertices
  PRIVATE
    filePLY
    rply
)

add_executable(
  epsilon-finder
    find-eps.cpp
    ${CPP}
)

set_target_properties(
  epsilon-finder
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "../lib/"
      LIBRARY_OUTPUT_DIRECTORY "../lib/"
      RUNTIME_OUTPUT_DIRECTORY "../"
)

link_directories(/usr/local/cuda-12.8/lib64)

target_link_libraries(
  epsilon-finder
    filePLY
    rply
    /usr/local/cuda-12.8/lib64/libcudart.so
)
