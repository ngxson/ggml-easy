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

## Features

### Effortless GPU support

ggml-easy abstracted out all the scheduler and buffer setup. GPU is enabled by default.

To disable it explicitly:

```cpp
ggml_easy::ctx_params params;
params.use_gpu = false; // true by default
ggml_easy::ctx ctx(params);
```

Please note that the GPU support is for convenience and is not aimed to have the best performance. Some operations will fallback to CPU if the GPU does not support them.

### Load safetensors without converting to GGUF

You can directly load `.safetensors` file to `ggml-easy` without having to convert it to GGUF! Currently, F32, F16 and BF16 types are supported.

```cpp
ggml_easy::ctx_params params;
ggml_easy::ctx ctx(params);
ctx.load_safetensors("mimi.safetensors", {
    // optionally, rename tensor to make it shorter (name length limit in ggml is 64 characters)
    {".acoustic_residual_vector_quantizer", ".acoustic_rvq"},
    {".semantic_residual_vector_quantizer", ".semantic_rvq"},
});
```

For a complete example, please have a look on [demo/safetensors.cpp](demo/safetensors.cpp) where I load both GGUF + safetensors files, then compare them.

TODO: multi-shards are not supported for now, will add it soon!

### Define input, output easily

When building computation graph, each input and output nodes can be added with single line of code:

```cpp
ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
    ggml_tensor * a = utils.new_input("a", GGML_TYPE_F32, cols_A, rows_A);
    ggml_tensor * b = utils.new_input("b", GGML_TYPE_F32, cols_B, rows_B);
    ...
    utils.mark_output(result, "result");
});
```

### Easy debugging

You can also print the intermediate results with minimal effort:

```cpp
ggml_tensor * a = utils.new_input("a", GGML_TYPE_F32, cols_A, rows_A);
ggml_tensor * b = utils.new_input("b", GGML_TYPE_F32, cols_B, rows_B);
ggml_tensor * a_mul_b = ggml_mul_mat(ctx_gf, a, b);
utils.debug_print(a_mul_b, "a_mul_b");
```

This will print the intermediate result of `A * B` upon `compute()` is called, no more manual `ggml_backend_tensor_get`!

```
a_mul_b.shape = [4, 3]
a_mul_b.data: [
     [
      [     60.0000,      55.0000,      50.0000,     110.0000],
      [     90.0000,      54.0000,      54.0000,     126.0000],
      [     42.0000,      29.0000,      28.0000,      64.0000],
     ],
    ]
```
