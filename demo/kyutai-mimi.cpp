#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>


/**
 * This is WIP, currently only for logits matching
 * 
 * To get the gguf:
 * 1. Download the model.safetensors file from https://huggingface.co/kyutai/mimi
 * 2. Run: python convert_safetensors_to_gguf.py --outtype f32 model.safetensors mimi.gguf
 * 
 * Note: do NOT upload the gguf to the internet, it is NOT compatible with llama.cpp and people will complain.
 */

static int64_t div_ceil(int64_t a, int64_t b) {
    return a / b + (a % b ? 1 : 0);
}

struct mimi_config_t {
    int max_position_embeddings = 8000;
    int num_hidden_layers = 8;
    int n_embd = 512;
    int n_ffn = 2048;
    int n_head = 8;
    int n_head_kv = 8;
    int n_rot = 64;
    float norm_eps = 1e-5;
    float rope_theta = 10000.0f;
    int sliding_window = 250;
    std::array<int, 4> upsampling_ratio   = {8, 6, 5, 4};
    std::array<int, 4> downsampling_ratio = {4, 5, 6, 8}; // reverse of upsampling_ratio
} mimi_config;

// based on MimiEncoder
// SEANet encoder as used by Mimi.
struct mimi_encoder {
    bool causal = true;
    struct layer {
        bool is_elu = false;
        bool is_resnet = false;
        ggml_tensor * conv_0_w;
        ggml_tensor * conv_0_b;
        ggml_tensor * conv_1_w;
        ggml_tensor * conv_1_b;
        int stride = 1;
    };
    int dilation_growth_rate = 2; // TODO: unused?
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
                .stride = mimi_config.downsampling_ratio[i],
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

        // based on MimiConv1d
        auto mimi_conv_1d = [&](ggml_tensor * x, ggml_tensor * kernel, ggml_tensor * bias, int stride, int dilation) {
            int64_t kernel_size = (kernel->ne[0] - 1) * dilation + 1;
            int64_t p_total = kernel_size - stride; // padding total
            int64_t p_half = p_total / 2;
            int64_t is_p_odd = p_total % 2; // is padding odd

            int64_t n_frames = div_ceil(x->ne[0] - kernel_size + p_total, stride);
            int64_t ideal_len = n_frames * stride + kernel_size - p_total;
            int64_t p_extra = ideal_len - x->ne[0];

            int64_t p_right = (causal ? 0 : p_half) + p_extra;
            int64_t p_left = p_total - (causal ? 0 : p_half);

            // add asymmetric padding
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

            kernel = ggml_cast(ctx0, kernel, GGML_TYPE_F16); // TODO: do this at conversion time
            x = ggml_conv_1d(ctx0, kernel, x, stride, 0, dilation);
            bias = ggml_cont(ctx0, ggml_transpose(ctx0, bias)); // TODO: do this at conversion time
            x = ggml_add(ctx0, x, bias);
            ggml_set_name(x, "mimi_conv_1d");
            return x;
        };

        // int i = 0; // for debugging
        for (auto & layer : layers) {
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
            // utils.debug_print(x, "after_layer_%d", i); i++;
        }

        return x;
    }
};

struct mimi_transformer {
    struct layer {
        ggml_tensor * inp_norm_w;
        ggml_tensor * inp_norm_b;

        ggml_tensor * attn_q;
        ggml_tensor * attn_k;
        ggml_tensor * attn_v;
        ggml_tensor * attn_o;
        ggml_tensor * attn_post_norm_w;
        ggml_tensor * attn_post_norm_b;
        ggml_tensor * attn_layer_scale;

        ggml_tensor * ffn_up;
        ggml_tensor * ffn_down;
        ggml_tensor * mlp_layer_scale;
    };
    std::vector<layer> layers;

    mimi_transformer(ggml_easy::ctx & ctx, const char * prefix, int n_layers) {
        for (int il = 0; il < n_layers; il++) {
            layers.push_back({
                .inp_norm_w = ctx.get_weight("%s_transformer.layers.%d.input_layernorm.weight", prefix, il),
                .inp_norm_b = ctx.get_weight("%s_transformer.layers.%d.input_layernorm.bias",   prefix, il),

                .attn_q           = ctx.get_weight("%s_transformer.layers.%d.self_attn.q_proj.weight",         prefix, il),
                .attn_k           = ctx.get_weight("%s_transformer.layers.%d.self_attn.k_proj.weight",         prefix, il),
                .attn_v           = ctx.get_weight("%s_transformer.layers.%d.self_attn.v_proj.weight",         prefix, il),
                .attn_o           = ctx.get_weight("%s_transformer.layers.%d.self_attn.o_proj.weight",         prefix, il),
                .attn_post_norm_w = ctx.get_weight("%s_transformer.layers.%d.post_attention_layernorm.weight", prefix, il),
                .attn_post_norm_b = ctx.get_weight("%s_transformer.layers.%d.post_attention_layernorm.bias",   prefix, il),
                .attn_layer_scale = ctx.get_weight("%s_transformer.layers.%d.self_attn_layer_scale.scale",     prefix, il),

                .ffn_up          = ctx.get_weight("%s_transformer.layers.%d.mlp.fc1.weight",        prefix, il),
                .ffn_down        = ctx.get_weight("%s_transformer.layers.%d.mlp.fc2.weight",        prefix, il),
                .mlp_layer_scale = ctx.get_weight("%s_transformer.layers.%d.mlp_layer_scale.scale", prefix, il),
            });
        }
    }

    ggml_tensor * forward(ggml_context * ctx0, ggml_easy::ctx::build_utils & utils, ggml_tensor * input, ggml_tensor * inp_pos) {
        int n_tokens    = input->ne[1];
        ggml_tensor * x = input;

        auto layer_norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            x = ggml_norm(ctx0, x, mimi_config.norm_eps);
            x = ggml_mul(ctx0, x, w);
            x = ggml_add(ctx0, x, b);
            return x;
        };

        auto llama_permute = [&](ggml_tensor * w) {
            int n_head = mimi_config.n_head;
            ggml_tensor * tmp = ggml_reshape_4d(ctx0, w, w->ne[0], w->ne[1] / n_head / 2, 2, n_head);
            tmp = ggml_permute(ctx0, tmp, 0, 2, 1, 3);
            tmp = ggml_cont(ctx0, tmp);
            return ggml_reshape_2d(ctx0, tmp, w->ne[0], w->ne[1]);
        };

        ggml_easy::debug::print_tensor_shape(input);

        ggml_tensor * residual = input;

        int i = 0; // for debugging
        for (auto & layer : layers) {
            residual = x;

            // input layer norm
            x = layer_norm(x, layer.inp_norm_w, layer.inp_norm_b);
            utils.debug_print(x, "inp_normed_layer_%d", i);

            // self attention
            {
                ggml_tensor * q = ggml_mul_mat(ctx0, llama_permute(layer.attn_q), x);
                ggml_tensor * k = ggml_mul_mat(ctx0, llama_permute(layer.attn_k), x);
                ggml_tensor * v = ggml_mul_mat(ctx0, layer.attn_v, x);

                int n_embd_head = mimi_config.n_embd / mimi_config.n_head;
                q = ggml_reshape_3d(ctx0, q, n_embd_head, mimi_config.n_head,    n_tokens);
                k = ggml_reshape_3d(ctx0, k, n_embd_head, mimi_config.n_head_kv, n_tokens);
                v = ggml_reshape_3d(ctx0, v, n_embd_head, mimi_config.n_head_kv, n_tokens);

                int n_rot = n_embd_head;
                q = ggml_rope_inplace(ctx0, q, inp_pos, n_rot, 0);
                q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
                utils.debug_print(q, "q rope");

                k = ggml_rope_inplace(ctx0, k, inp_pos, n_rot, 0);
                k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
                utils.debug_print(k, "k rope");

                ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
                ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
                kq = ggml_scale_inplace(ctx0, kq, 1.0f / std::sqrt(n_embd_head));
                ggml_tensor * kq_masked = ggml_diag_mask_inf_inplace(ctx0, kq, n_tokens);
                kq = ggml_soft_max_inplace(ctx0, kq_masked);
                // utils.debug_print(kq, "kq softmax");

                v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

                ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
                ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
                kqv = ggml_reshape_3d(ctx0, kqv, n_embd_head, n_tokens, mimi_config.n_head);
                kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                kqv = ggml_cont_2d(ctx0, kqv, mimi_config.n_embd, n_tokens);
                utils.debug_print(kqv, "kqv");
                utils.debug_print(ggml_sum(ctx0, kqv), "kqv_sum");

                x = ggml_mul_mat(ctx0, layer.attn_o, kqv);
            }

            // residual
            x = ggml_mul(ctx0, x, layer.attn_layer_scale);
            x = ggml_add(ctx0, x, residual);
            utils.debug_print(x, "after_attn_%d", i);

            residual = x;
            x = layer_norm(x, layer.attn_post_norm_w, layer.attn_post_norm_b);

            // mlp
            {
                x = ggml_mul_mat(ctx0, layer.ffn_up, x);
                x = ggml_gelu(ctx0, x);
                x = ggml_mul_mat(ctx0, layer.ffn_down, x);
            }

            // residual
            x = ggml_mul(ctx0, x, layer.mlp_layer_scale);
            x = ggml_add(ctx0, x, residual);
            utils.debug_print(x, "output_layer_%d", i);
            utils.debug_print(ggml_sum(ctx0, x), "output_layer_%d_sum", i); i++;
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

    mimi_encoder     encoder(ctx);
    mimi_transformer encoder_transformer(ctx, "encoder", 8);

    // create cgraph
    int n_pos = -1;
    ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * input = utils.new_input("input", GGML_TYPE_F32, 2048);
        ggml_tensor * embeddings = encoder.forward(ctx_gf, utils, input);
        utils.debug_print(embeddings, "embeddings");

        n_pos = embeddings->ne[0];
        ggml_tensor * inp_pos = utils.new_input("positions", GGML_TYPE_I32, n_pos);

        embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
        ggml_tensor * output = encoder_transformer.forward(ctx_gf, utils, embeddings, inp_pos);
        utils.mark_output(output, "output");
    });

    ctx.set_tensor_data("input", [](int, int, int, int) { return 1.0f; });

    // position data
    std::vector<int> pos_data(n_pos);
    for (int i = 0; i < n_pos; i++) {
        pos_data[i] = i;
    }
    ctx.set_tensor_data("positions", pos_data.data());

    ctx.compute();

    // print result
    //ggml_easy::debug::print_tensor_data(result_tensor, result_data.data());

    return 0;
}
