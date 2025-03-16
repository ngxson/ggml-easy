# ggml-easy

A simple C++ wrapper around [GGML](https://github.com/ggml-org/ggml) to make model loading and execution easier with GPU acceleration support.

## Introduction

`ggml-easy` is a lightweight header-only C++ library that simplifies working with GGML, the tensor library used in projects like llama.cpp. It provides a clean interface for loading GGUF models, creating computation graphs, and executing them on CPU or GPU with minimal boilerplate code.

## Setup

As a header-only library, using ggml-easy is straightforward:

1. Include the headers in your project
2. Make sure you have GGML as a dependency in `CMakeLists.txt`
3. Use the `ggml_easy` namespace in your code

Example:
```cpp
#include "ggml-easy.h"

// Your code here
```

See [demo/basic.cpp](demo/basic.cpp) for a complete example of how to use `ggml-easy` in a project.

## Compile examples

To compile everything inside `demo/*`

```sh
cmake -B build
cmake --build build -j
# output: build/bin/*
```
