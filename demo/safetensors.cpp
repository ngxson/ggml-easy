#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

/**
 * This example demonstrates how to load safetensors directly to GGML without any conversions.
 * 
 * We load both the GGUF and safetensors of the same model, then compare the tensors.
 * All tensors are expected to be equal.
 * 
 * I'm using https://huggingface.co/kyutai/mimi as the model. But you can use any model.
 * 
 * To get the safetensors:
 * 1. Download the model.safetensors file
 * 2. Rename the "model.safetensors" to "mimi.safetensors"
 * 
 * To get the gguf:
 * 1. Download the model.safetensors file
 * 2. Run: python convert_safetensors_to_gguf.py --outtype f32 model.safetensors mimi.gguf
 * 
 */

int main() {
    ggml_easy::ctx_params params;
    params.use_gpu = false;
    params.log_level = GGML_LOG_LEVEL_DEBUG;

    ggml_easy::ctx ctx0(params);
    ctx0.load_safetensors("mimi.safetensors", {
        {".acoustic_residual_vector_quantizer", ".acoustic_rvq"},
        {".semantic_residual_vector_quantizer", ".semantic_rvq"},
    });

    ggml_easy::ctx ctx1(params);
    ctx1.load_gguf("mimi.gguf");

    GGML_ASSERT(ctx0.tensors.size() == ctx1.tensors.size());

    GGML_ASSERT(ggml_backend_buft_is_host(ctx0.backend_buft[0]));
    GGML_ASSERT(ggml_backend_buft_is_host(ctx1.backend_buft[0]));

    // compare the tensors
    for (auto & t : ctx0.tensors) {
        auto tensor0 = t.second;
        auto tensor1 = ctx1.get_weight(t.first.c_str());

        GGML_ASSERT(ggml_are_same_shape(tensor0, tensor1));
        GGML_ASSERT(tensor0->type == GGML_TYPE_F32);
        GGML_ASSERT(tensor1->type == GGML_TYPE_F32);

        float diff = 0.0;
        for (size_t i = 0; i < ggml_nelements(tensor0); ++i) {
            float v0 = ggml_get_f32_1d(tensor0, i);
            float v1 = ggml_get_f32_1d(tensor1, i);
            diff += fabs(v0 - v1);
        }

        printf("%-60s: diff = %f\n", t.first.c_str(), diff);
        GGML_ASSERT(diff < 1e-6);
    }

    printf("\nOK: All tensors are equal\n");

    return 0;
}
