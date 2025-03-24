#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>


/**
 * This is WIP, currently only for logits matching
 * 
 * To get the gguf:
 * 1. Download the model.safetensors file from https://huggingface.co/kyutai/mimi
 * 2. Run: python convert_safetensors_to_gguf.py --outtype f16 model.safetensors mimi.gguf
 * 
 * Note: do NOT upload the gguf to the internet, it is NOT compatible with llama.cpp and people will complain.
 */

std::array<int, 4> upsampling_ratio   = {8, 6, 5, 4};
std::array<int, 4> downsampling_ratio = {4, 5, 6, 8}; // reverse of upsampling_ratio

struct mimi_encoder {
    struct layer {
        bool is_elu = false;
        bool is_resnet = false;
        ggml_tensor * conv_0_w;
        ggml_tensor * conv_0_b;
        ggml_tensor * conv_1_w;
        ggml_tensor * conv_1_b;
        int stride = 1;
    };
    int dilation_growth_rate = 2;
    std::vector<layer> layers;

    mimi_encoder(ggml_easy::ctx & ctx) {
        std::array<int, 4> repeated_pattern = {1, 4, 7, 10};

        layers.push_back({
            .conv_0_w = ctx.get_weight("encoder.layers.0.conv.weight"),
            .conv_0_b = ctx.get_weight("encoder.layers.0.conv.bias"),
        });
        for (int i = 0; i < 4; ++i) {
            int i_start = repeated_pattern[i];
            // residual layers
            layers.push_back({
                .is_resnet = true,
                .conv_0_w = ctx.get_weight("encoder.layers.%d.block.1.conv.weight", i_start),
                .conv_0_b = ctx.get_weight("encoder.layers.%d.block.1.conv.bias",   i_start),
                .conv_1_w = ctx.get_weight("encoder.layers.%d.block.3.conv.weight", i_start),
                .conv_1_b = ctx.get_weight("encoder.layers.%d.block.3.conv.bias",   i_start),
            });
            // downsampling layers
            layers.push_back({
                .is_elu = true, // layer (i_start + 1)
            });
            layers.push_back({
                .conv_0_w = ctx.get_weight("encoder.layers.%d.conv.weight", i_start + 2),
                .conv_0_b = ctx.get_weight("encoder.layers.%d.conv.bias",   i_start + 2),
                .stride = downsampling_ratio[i],
            });
        }
        layers.push_back({
            .is_elu = true, // layer 13
        });
        layers.push_back({
            .conv_0_w = ctx.get_weight("encoder.layers.14.conv.weight"),
            .conv_0_b = ctx.get_weight("encoder.layers.14.conv.bias"),
        });
    }

    ggml_tensor * forward(ggml_context * ctx0, ggml_easy::ctx::build_utils & utils, ggml_tensor * input) {
        ggml_tensor * x = input;

        auto mimi_conv_1d = [&ctx0](ggml_tensor * x, ggml_tensor * kernel, ggml_tensor * bias, int stride, int dilation) {
            int64_t kernel_size = (kernel->ne[0] - 1) * dilation + 1;
            int64_t p_total = kernel_size - stride; // padding total
            int64_t p_half = p_total / 2;
            int64_t is_p_odd = p_total % 2; // is padding odd

            int64_t n_frames = (x->ne[0] - kernel_size + p_total) / stride + 1;
            int64_t ideal_len = n_frames * stride + kernel_size - p_total;
            int64_t p_extra = ideal_len - x->ne[0];
            /*printf("ideal_len: %ld, p_extra: %ld\n", ideal_len, p_extra);
            if (p_extra > 0) {
                ggml_tensor * zeros = ggml_new_tensor_2d(ctx0, x->type, p_extra, x->ne[1]);
                ggml_easy::debug::print_tensor_shape(x);
                ggml_easy::debug::print_tensor_shape(zeros);
                zeros = ggml_scale(ctx0, zeros, 0.0f);
                x = ggml_concat(ctx0, x, zeros, 0);
            }


            if (is_p_odd) {
                // if odd, we add one padding column to the left
                ggml_tensor * zeros = ggml_new_tensor_2d(ctx0, x->type, 1, x->ne[1]);
                zeros = ggml_scale(ctx0, zeros, 0.0f);
                x = ggml_concat(ctx0, zeros, x, 0);
                ggml_set_name(x, "mimi_conv_1d_pad");
                ggml_easy::debug::print_tensor_shape(x);
            }*/

            int64_t p_right = p_half + p_extra;
            int64_t p_left = p_total - p_half;
            //printf("ideal_len: %ld, p_extra: %ld\n", ideal_len, p_extra);
            printf("p_total: %ld, p_left: %ld, p_right: %ld, p_extra: %ld\n", p_total, p_left, p_right, p_extra);

            // add padding
            if (p_left > 0) {
                ggml_tensor * zeros = ggml_new_tensor_2d(ctx0, x->type, p_left, x->ne[1]);
                zeros = ggml_scale(ctx0, zeros, 0.0f);
                x = ggml_concat(ctx0, zeros, x, 0);
            }
            if (p_right > 0) {
                ggml_tensor * zeros = ggml_new_tensor_2d(ctx0, x->type, p_right, x->ne[1]);
                zeros = ggml_scale(ctx0, zeros, 0.0f);
                x = ggml_concat(ctx0, x, zeros, 0);
            }

            //printf("kernel_size: %ld, padding_total: %ld, padding: %ld\n", kernel_size, p_total, p_half);
            x = ggml_conv_1d(ctx0, kernel, x, stride, 0, dilation);
            bias = ggml_cont(ctx0, ggml_transpose(ctx0, bias)); // TODO: do this at conversion time
            x = ggml_add(ctx0, x, bias);
            ggml_set_name(x, "mimi_conv_1d");
            return x;
        };

        int i = 0;
        for (auto & layer : layers) {
            printf("\n -- layer %d -- \n", i);
            if (layer.is_elu) {
                x = ggml_elu(ctx0, x);
            } else if (layer.is_resnet) {
                ggml_tensor * residual = x;
                x = ggml_elu(ctx0, x);
                x = mimi_conv_1d(x, layer.conv_0_w, layer.conv_0_b, 1, 1);
                x = ggml_elu(ctx0, x);
                x = mimi_conv_1d(x, layer.conv_1_w, layer.conv_1_b, 1, 1);
                x = ggml_add(ctx0, x, residual);
            } else {
                x = mimi_conv_1d(x, layer.conv_0_w, layer.conv_0_b, layer.stride, 1);
            }
            utils.debug_print(x, "after_layer_%d", i);
            ggml_easy::debug::print_tensor_shape(x);
            i++;
        }

        return x;
    }
};

int main() {
    ggml_easy::ctx_params params;
    //params.log_level = GGML_LOG_LEVEL_DEBUG;
    ggml_easy::ctx ctx(params);

    ctx.load_gguf("mimi.gguf");

    // optional: print backend buffer info
    ggml_easy::debug::print_backend_buffer_info(ctx);

    mimi_encoder encoder(ctx);

    // create cgraph
    ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * input = utils.new_input("input", GGML_TYPE_F32, 2048);
        ggml_tensor * output = encoder.forward(ctx_gf, utils, input);
        utils.mark_output(output, "output");
        //ggml_graph_dump_dot(gf, nullptr, "mimi.dot");
    });

    ctx.set_tensor_data("input", [](int, int, int, int) { return 1.0f; });

    ctx.compute();

    // print result
    //ggml_easy::debug::print_tensor_data(result_tensor, result_data.data());

    return 0;
}
