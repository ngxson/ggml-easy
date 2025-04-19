cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("ggml-easy" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(ggml)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# demo
set(DEMO_TARGETS
    basic
    dyt-rms
    svd
    kyutai-mimi
    safetensors
    whisper-encoder)

foreach(TARGET ${DEMO_TARGETS})
  add_executable(${TARGET} demo/${TARGET}.cpp)
  target_link_libraries(${TARGET} PRIVATE ggml)
  target_compile_features(${TARGET} PRIVATE cxx_std_17)
endforeach()
