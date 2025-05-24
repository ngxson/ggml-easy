#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

/**
 * Experiment on fastvlm implementation
 * 
 * This is non-complete code, do not ask how to use it
 */


struct layer {
    struct rep_mixer {
        ggml_tensor * convffn_bn_w;
        ggml_tensor * convffn_bn_b;
        ggml_tensor * convffn_bn_mean;
        ggml_tensor * convffn_bn_std;
        ggml_tensor * convffn_w;
        ggml_tensor * convffn_fc1;
        ggml_tensor * convffn_fc2;
        ggml_tensor * layer_scale;
        ggml_tensor * token_mixer_conv_w;
        ggml_tensor * token_mixer_conv_b;
    };
    std::array<rep_mixer, 2> mixers;
};

int main() {
    ggml_easy::ctx_params params;
    // params.log_level = GGML_LOG_LEVEL_DEBUG;
    params.safetensors_ignore_unknown_dtype = true;
    params.use_gpu = false;
    ggml_easy::ctx ctx(params);
    ctx.load_safetensors("fastvlm.safetensors", {
        {"model.vision_tower.vision_tower.model.", ""},
    });

    const int image_size = 1024;

    auto * _patch_embed_0_w = ctx.get_weight("patch_embed.0.reparam_conv.weight");
    auto * _patch_embed_0_b = ctx.get_weight("patch_embed.0.reparam_conv.bias");
    auto * _patch_embed_1_w = ctx.get_weight("patch_embed.1.reparam_conv.weight");
    auto * _patch_embed_1_b = ctx.get_weight("patch_embed.1.reparam_conv.bias");
    auto * _patch_embed_2_w = ctx.get_weight("patch_embed.2.reparam_conv.weight");
    auto * _patch_embed_2_b = ctx.get_weight("patch_embed.2.reparam_conv.bias");

    // create cgraph
    ctx.build_graph([&](ggml_context * ctx0, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * inp = utils.new_input("inp", GGML_TYPE_F32, image_size, image_size, 3);
        ggml_tensor * tmp;

        auto * patch_embed_0_w = ggml_cast(ctx0, _patch_embed_0_w, GGML_TYPE_F16);
        auto * patch_embed_0_b = ggml_cast(ctx0, _patch_embed_0_b, GGML_TYPE_F32);
        auto * patch_embed_1_w = ggml_cast(ctx0, _patch_embed_1_w, GGML_TYPE_F16);
        auto * patch_embed_1_b = ggml_cast(ctx0, _patch_embed_1_b, GGML_TYPE_F32);
        auto * patch_embed_2_w = ggml_cast(ctx0, _patch_embed_2_w, GGML_TYPE_F16);
        auto * patch_embed_2_b = ggml_cast(ctx0, _patch_embed_2_b, GGML_TYPE_F32);

        inp = ggml_conv_2d(ctx0, patch_embed_0_w, inp, 2, 2, 1, 1, 1, 1);
        tmp = ggml_reshape_3d(ctx0, patch_embed_0_b, 1, 1, ggml_nelements(patch_embed_0_b));
        inp = ggml_add(ctx0, inp, tmp);
        inp = ggml_gelu(ctx0, inp);

        inp = ggml_conv_2d_dw(ctx0, patch_embed_1_w, inp, 2, 2, 1, 1, 1, 1);
        tmp = ggml_reshape_3d(ctx0, patch_embed_1_b, 1, 1, ggml_nelements(patch_embed_1_b));
        inp = ggml_add(ctx0, inp, tmp);
        inp = ggml_gelu(ctx0, inp);

        inp = ggml_conv_2d(ctx0, patch_embed_2_w, inp, 1, 1, 0, 0, 1, 1);
        tmp = ggml_reshape_3d(ctx0, patch_embed_2_b, 1, 1, ggml_nelements(patch_embed_2_b));
        inp = ggml_add(ctx0, inp, tmp);
        inp = ggml_gelu(ctx0, inp);

        utils.debug_print(inp, "after_conv");
    });

    std::vector<float> inp(image_size * image_size * 3);
    for (int i = 0; i < image_size * image_size * 3; ++i) {
        inp[i] = (float)0.1f;
    }
    ctx.set_tensor_data("inp", inp.data());

    // compute
    ggml_status status = ctx.compute();

    return 0;
}
