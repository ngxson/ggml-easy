#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

/**
 * Random experiment, do not use it
 */

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);

    // experiment with torch unfold equivalent in GGML
    {
        const int h = 12;
        const int w = 2;
        const int hidden_size = 8;
        ctx.build_graph([&](ggml_context * ctx0, ggml_cgraph * gf, auto & utils) {
            ggml_tensor * inp = utils.new_input("inp", GGML_TYPE_F32, hidden_size, h*w);
            ggml_tensor * x = inp;
            utils.debug_print(ggml_scale(ctx0, inp, 1.0f), "inp0");

            x = ggml_reshape_3d(ctx0, x, hidden_size, w, h);
            x = ggml_permute(ctx0, x, 2, 0, 1, 3); // [x, y, hidden_size]
            x = ggml_cont(ctx0, x);
            utils.debug_print_full(x, "grid");

            ggml_tensor * kernel = ggml_view_3d(ctx0, inp, 2, 2, x->ne[2], 0, 0, 0);
            x = ggml_im2col(ctx0, kernel, x, 2, 2, 0, 0, 1, 1, true, inp->type);

            utils.debug_print_full(x, "im2col");

            x = ggml_reshape_2d(ctx0, x, x->ne[0], x->ne[1] * x->ne[2]);
            utils.debug_print(x, "result");
        });
        std::vector<float> inp_data(h * w * hidden_size);
        for (int i = 0; i < h * w * hidden_size; ++i) {
            inp_data[i] = (float)i;
        }
        ctx.set_tensor_data("inp", inp_data.data());
        ctx.compute();
    }

    printf("\n\n\nLlama4UnfoldConvolution\n\n");
    {
        ggml_easy::ctx ctx(params);
        ctx.load_safetensors("../models/llama4vit.safetensors", {});

        ggml_tensor * patch_embeddings_0 = ctx.get_weight("vision_model.patch_embedding.linear.weight");

        const int h = 336;
        const int w = 336;
        const int patch_size = 14;
        const int n_embd = 1408;
        const int n_patches = (h / patch_size) * (w / patch_size);

        ctx.build_graph([&](ggml_context * ctx0, ggml_cgraph * gf, auto & utils) {
            ggml_tensor * inp = utils.new_input("inp", GGML_TYPE_F32, h, w, 3);
            
            // Llama4UnfoldConvolution
            {
                ggml_tensor * kernel = ggml_reshape_4d(ctx0, patch_embeddings_0,
                                                        patch_size, patch_size, 3, n_embd);
                inp = ggml_im2col(ctx0, kernel, inp, patch_size, patch_size, 0, 0, 1, 1, true, inp->type);
                //inp = ggml_reshape_2d(ctx0, inp, inp->ne[0], inp->ne[1] * inp->ne[2]); // flatten to 2D
                utils.debug_print(inp, "im2col");
                utils.debug_print(ggml_sum(ctx0, inp), "im2col_sum");

                utils.debug_print(ggml_cast(ctx0, patch_embeddings_0, GGML_TYPE_F32), "patch_embeddings_0");

                inp = ggml_mul_mat(ctx0, patch_embeddings_0, inp);
                utils.debug_print(inp, "patch_conv");
                utils.debug_print(ggml_sum(ctx0, inp), "patch_conv_sum");

                inp = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);
            }

            //inp = ggml_reshape_2d(ctx0, inp, inp->ne[0], inp->ne[1] * inp->ne[2]);
            utils.debug_print(inp, "result");
        });

        std::vector<float> inp_data(h * w * 3, 0.0);
        for (int i = 0; i < h * w; ++i) {
            inp_data[i] = 1.0; //(float)i * 0.1;
        }
        ctx.set_tensor_data("inp", inp_data.data());
        ctx.compute();
    }

    return 0;
}
