cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("ggml-easy" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(ggml)

# demo
set(TARGET demo)
add_executable(${TARGET} demo.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET demo-dyt-rms)
add_executable(${TARGET} demo-dyt-rms.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
