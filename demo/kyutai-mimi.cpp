#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>
#include <float.h>
#include <cmath>


/**
 * (Stil WIP) This is my trial to reimplement the Mimi model from Kyutai using ggml, the code is based on HF transformers implementation. See "modeling_mimi.py" for the original code.
 * 
 * To get the model (we are using safetensors directly, no need to convert to GGUF):
 * 1. Download the model.safetensors file from https://huggingface.co/kyutai/mimi
 * 2. Rename the "model.safetensors" to "mimi.safetensors"
 * 
 * Note: do NOT upload the gguf to the internet, it is NOT compatible with llama.cpp and people will complain.
 * 
 * ---
 * 
 * For the ENCODER, it takes raw audio waveform as input and output audio codes. Steps are:
 * 1. Convert waveform to embeddings using mimi_encoder (SEANet encoder), basically just a bunch of Conv1d but the padding is quite tricky.
 * 2. Process the embeddings using a transformer, here we use an auto-aggressive one (causal mask). This is because Laurent told me that they only trained the model with auto-regressive setting.
 * 3. Quantize the embeddings using a residual vector quantizer (RVQ) to get the audio codes. The RVQ has 32 codebooks, one for semantic and 31 for acoustic. Doing this on ggml is a bit tricky because I need to reimplement euclidean distance from scratch.
 * 
 * In the code below, we take 2048 samples of audio waveform as input (value = 1.0f), expected output is 2 tokens (according to python implementation).
 * 
 * Python code:
 *   model = MimiModel.from_pretrained("/Users/ngxson/work/models/mimi")
 *   input_values = torch.ones((1, 1, 2048))
 *   encoder_outputs = model.encode(input_values)  # this should match the output of ggml
 * 
 * ---
 * 
 * For the DECODER, we simply do the reverse of the above steps.
 * The good thing is that this time, we don't need to care about euclidean distance.
 * 
 * Python code:
 *   model = MimiModel.from_pretrained("/Users/ngxson/work/models/mimi")
 *   input_values = torch.tensor([[ [i, i+1, i+2] for i in range(0, 3*32, 3) ]], dtype=torch.long)
 *   audio_values = model.decode(input_values)[0]  # this should match the output of ggml
 *
 *   Expected output:
 *     torch.Size([1, 1, 5760]) 
 *     tensor([[[ 0.0117,  0.0130, -0.0007,  ..., -0.1295, -0.1258, -0.1343]]])
 */

struct mimi_config_t {
    bool causal = true;
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
    // vector quantizer
    float frame_rate = 12.5;
    int audio_channels = 1;
    int codebook_size = 2048;
    int codebook_dim = 256;
    int n_semantic_components = 1;
    int n_acoustic_components = 31;
    // decode
    float trim_right_ratio = 1.0f;
} mimi_config;


///////////////////////////////////////////////////////////////////////////
// extension to ggml.h
// TODO: add these ops to the library (ofc with a more optimized kernel)


// mode: (0) constant, (1) reflect, (2) replicate, (3) circular
// value is only used in "constant"
// only "constant" with 0.0f and "replicate" are implemented here
static ggml_tensor * ggml_pad_ext(ggml_context * ctx0, ggml_tensor * x, int mode,
        int64_t pad_left, int64_t pad_right, float value = 0.0f) {
    GGML_ASSERT(value == 0.0f); // we can technically use ggml_arange, but for simplication we only support 0.0f
    GGML_ASSERT(mode == 0 || mode == 2);
    if (pad_left > 0) {
        ggml_tensor * tmp = ggml_new_tensor_2d(ctx0, x->type, pad_left, x->ne[1]);
        if (mode == 0) {
            tmp = ggml_scale(ctx0, tmp, value);
        } else if (mode == 2) {
            ggml_tensor * elem = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], 0); // get first column
            tmp = ggml_repeat(ctx0, elem, tmp);
        }
        x = ggml_concat(ctx0, tmp, x, 0);
    }
    if (pad_right > 0) {
        ggml_tensor * tmp = ggml_new_tensor_2d(ctx0, x->type, pad_right, x->ne[1]);
        if (mode == 0) {
            tmp = ggml_scale(ctx0, tmp, value);
        } else if (mode == 2) {
            int64_t last = x->ne[0] - 1;
            ggml_tensor * elem = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], last * ggml_element_size(x)); // get last column
            tmp = ggml_repeat(ctx0, elem, tmp);
        }
        x = ggml_concat(ctx0, x, tmp, 0);
    }
    return x;
}

static ggml_tensor * ggml_argmin(ggml_context * ctx0, ggml_tensor * x) {
    ggml_tensor * tmp = ggml_scale(ctx0, x, -1.0f);
    return ggml_argmax(ctx0, tmp);
}

// lookup nearest vector in codebook based on euclidean distance
// return index of the vector in codebook, single element with I32 type
static ggml_tensor * ggml_lookup_vec(ggml_context * ctx0, ggml_tensor * codebook, ggml_tensor * x) {
    ggml_tensor * tmp = ggml_add(ctx0, codebook, ggml_scale(ctx0, x, -1.0f)); // a - x
    tmp = ggml_mul(ctx0, tmp, tmp); // (a - x) ** 2
    tmp = ggml_sum_rows(ctx0, tmp);
    tmp = ggml_sqrt(ctx0, tmp);
    tmp = ggml_cont(ctx0, ggml_transpose(ctx0, tmp));
    // villain version of argmin :-)
    tmp = ggml_argmax(ctx0, ggml_scale(ctx0, tmp, -1.0f));
    GGML_ASSERT(ggml_nelements(tmp) == 1);
    return tmp;
}

// lookup vectors in codebook based on euclidean distance
// return indices of the vectors in codebook, 1D tensor with I32 type
static ggml_tensor * ggml_lookup_vectors(ggml_easy::ctx::build_utils & utils, ggml_context * ctx0, ggml_tensor * codebook, ggml_tensor * list_vec, ggml_tensor * out, size_t offset) {
    int64_t n_col = list_vec->ne[0];
    int64_t n_row = list_vec->ne[1];
    for (int64_t ir = 0; ir < n_row; ir++) {
        ggml_tensor * row = ggml_view_1d(ctx0, list_vec, n_col, ir*n_col*ggml_element_size(list_vec));
        ggml_tensor * idx = ggml_lookup_vec(ctx0, codebook, row);
        ggml_tensor * dst = ggml_view_1d(ctx0, out, 1, offset + ir*ggml_element_size(out));
        ggml_build_forward_expand(utils.gf, ggml_cpy(ctx0, idx, dst));
    }
    return out;
}


///////////////////////////////////////////////////////////////////////////


static int64_t div_ceil(int64_t a, int64_t b) {
    return a / b + (a % b ? 1 : 0);
}

static ggml_tensor * mimi_conv_1d(ggml_easy::ctx::build_utils & utils, ggml_context * ctx0, ggml_tensor * x,
        ggml_tensor * kernel, ggml_tensor * bias, int stride, int dilation, bool pad_zero = true) {
    int64_t kernel_size = (kernel->ne[0] - 1) * dilation + 1;
    int64_t p_total = kernel_size - stride; // padding total
    int64_t p_half = p_total / 2;
    int64_t is_p_odd = p_total % 2; // is padding odd

    int64_t n_frames = div_ceil(x->ne[0] - kernel_size + p_total, stride);
    int64_t ideal_len = n_frames * stride + kernel_size - p_total;
    int64_t p_extra = ideal_len - x->ne[0];

    int64_t p_right = (mimi_config.causal ? 0 : p_half) + p_extra;
    int64_t p_left = p_total - (mimi_config.causal ? 0 : p_half);

    x = ggml_pad_ext(ctx0, x, pad_zero ? 0 : 2, p_left, p_right);
    // utils.debug_print(x, "mimi_conv_1d_padded");

    kernel = ggml_cast(ctx0, kernel, GGML_TYPE_F16); // TODO: do this at conversion time
    x = ggml_conv_1d(ctx0, kernel, x, stride, 0, dilation);
    if (bias) {
        bias = ggml_cont(ctx0, ggml_transpose(ctx0, bias)); // TODO: do this at conversion time
        x = ggml_add(ctx0, x, bias);
    }
    ggml_set_name(x, "mimi_conv_1d");
    return x;
};

static ggml_tensor * mimi_conv_transpose_1d(ggml_easy::ctx::build_utils & utils, ggml_context * ctx0, ggml_tensor * x,
        ggml_tensor * kernel, ggml_tensor * bias, int stride, int dilation, bool depthwise) {
    GGML_ASSERT(x->ne[1] == kernel->ne[2]);
    int64_t n_rows = x->ne[1];
    int64_t kernel_size = kernel->ne[0];
    int64_t p_total = kernel_size - stride; // padding total

    int64_t p_right = mimi_config.causal
        ? (float)p_total / mimi_config.trim_right_ratio
        : p_total / 2;
    int64_t p_left = p_total - p_right;

    ggml_tensor * out = nullptr;

    kernel = ggml_cast(ctx0, kernel, GGML_TYPE_F16); // TODO: do this at conversion time

    if (depthwise) {
        for (int64_t ir = 0; ir < n_rows; ir++) {
            ggml_tensor * row = ggml_view_1d(ctx0, x,
                                            x->ne[0], ir*x->ne[0]*ggml_element_size(x));
            ggml_tensor * krn = ggml_view_1d(ctx0, kernel,
                                            kernel->ne[0], ir*kernel->ne[0]*ggml_element_size(kernel));
            if (ir == 0) {
                ggml_set_name(krn, "krn");
                ggml_easy::debug::print_tensor_shape(krn);
            }
            row = ggml_conv_transpose_1d(ctx0, krn, row, stride, 0, dilation);
            if (ir == 0) {
                ggml_set_name(row, "ggml_conv_transpose_1d __________");
                ggml_easy::debug::print_tensor_shape(row);
            }
            // unpad (remove p_right and p_left columns)
            row = ggml_view_1d(ctx0, row, row->ne[0] - p_total, p_left*ggml_element_size(row));
    
            // TODO: concat can be slow, we should use ggml_view_1d/ggml_cpy to avoid realloc
            out = out ? ggml_concat(ctx0, out, row, 1) : row;
        }

    } else {
        out = ggml_conv_transpose_1d(ctx0, kernel, x, stride, 0, dilation);
        // unpad
        out = ggml_view_2d(ctx0, out,
            out->ne[0] - p_total, out->ne[1],
            out->nb[1], p_left*ggml_element_size(out));
    }

    if (bias) {
        bias = ggml_cont(ctx0, ggml_transpose(ctx0, bias)); // TODO: do this at conversion time
        out = ggml_add(ctx0, out, bias);
    }

    return out;
}

// based on MimiEncoder
// SEANet encoder as used by Mimi.
struct mimi_encoder_decoder {
    ggml_easy::ctx & ctx;
    struct layer {
        bool is_elu = false;
        bool is_resnet = false;
        bool is_transposed_conv = false;
        ggml_tensor * conv_0_w;
        ggml_tensor * conv_0_b;
        ggml_tensor * conv_1_w;
        ggml_tensor * conv_1_b;
        int stride = 1;
    };
    int dilation_growth_rate = 2; // TODO: unused?
    std::vector<layer> layers;

    std::array<int, 4> repeated_pattern = {1, 4, 7, 10};

    mimi_encoder_decoder(ggml_easy::ctx & ctx) : ctx(ctx) {}

    void load_encoder() {
        layers.push_back({
            .conv_0_w = ctx.get_weight("encoder.layers.0.conv.weight"),
            .conv_0_b = ctx.get_weight("encoder.layers.0.conv.bias"),
        });
        for (int i = 0; i < (int)repeated_pattern.size(); ++i) {
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

    void load_decoder() {
        layers.push_back({
            .conv_0_w = ctx.get_weight("decoder.layers.0.conv.weight"),
            .conv_0_b = ctx.get_weight("decoder.layers.0.conv.bias"),
        });
        for (int i = 0; i < (int)repeated_pattern.size(); ++i) {
            int i_start = repeated_pattern[i];
            // upsampling layers
            layers.push_back({
                .is_elu = true, // layer (i_start)
            });
            layers.push_back({
                .conv_0_w = ctx.get_weight("decoder.layers.%d.conv.weight", i_start + 1),
                .conv_0_b = ctx.get_weight("decoder.layers.%d.conv.bias",   i_start + 1),
                .stride = mimi_config.upsampling_ratio[i],
                .is_transposed_conv = true,
            });
            // residual layers
            layers.push_back({
                .is_resnet = true,
                .conv_0_w = ctx.get_weight("decoder.layers.%d.block.1.conv.weight", i_start + 2),
                .conv_0_b = ctx.get_weight("decoder.layers.%d.block.1.conv.bias",   i_start + 2),
                .conv_1_w = ctx.get_weight("decoder.layers.%d.block.3.conv.weight", i_start + 2),
                .conv_1_b = ctx.get_weight("decoder.layers.%d.block.3.conv.bias",   i_start + 2),
            });
        }
        layers.push_back({
            .is_elu = true, // layer 13
        });
        layers.push_back({
            .conv_0_w = ctx.get_weight("decoder.layers.14.conv.weight"),
            .conv_0_b = ctx.get_weight("decoder.layers.14.conv.bias"),
        });
    }

    ggml_tensor * forward(ggml_context * ctx0, ggml_easy::ctx::build_utils & utils, ggml_tensor * input) {
        ggml_tensor * x = input;

        // int i = 0; // for debugging
        for (auto & layer : layers) {
            if (layer.is_elu) {
                x = ggml_elu(ctx0, x);
            } else if (layer.is_resnet) {
                ggml_tensor * residual = x;
                x = ggml_elu(ctx0, x);
                ggml_easy::debug::print_tensor_shape(x);
                ggml_easy::debug::print_tensor_shape(layer.conv_0_w);
                x = mimi_conv_1d(utils, ctx0, x, layer.conv_0_w, layer.conv_0_b, 1, 1);
                x = ggml_elu(ctx0, x);
                x = mimi_conv_1d(utils, ctx0, x, layer.conv_1_w, layer.conv_1_b, 1, 1);
                x = ggml_add(ctx0, x, residual);
            } else {
                x = layer.is_transposed_conv
                    ? mimi_conv_transpose_1d(utils, ctx0, x, layer.conv_0_w, layer.conv_0_b, layer.stride, 1, false)
                    : mimi_conv_1d(utils, ctx0, x, layer.conv_0_w, layer.conv_0_b, layer.stride, 1);
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

        // TODO: do this at conversion time, see LlamaModel.permute in convert_hf_to_gguf.py
        auto llama_permute = [&](ggml_tensor * w) {
            int n_head = mimi_config.n_head;
            ggml_tensor * tmp = ggml_reshape_4d(ctx0, w, w->ne[0], w->ne[1] / n_head / 2, 2, n_head);
            tmp = ggml_permute(ctx0, tmp, 0, 2, 1, 3);
            tmp = ggml_cont(ctx0, tmp);
            return ggml_reshape_2d(ctx0, tmp, w->ne[0], w->ne[1]);
        };

        ggml_tensor * residual = input;

        int i = 0; // for debugging
        for (auto & layer : layers) {
            residual = x;

            // input layer norm
            x = layer_norm(x, layer.inp_norm_w, layer.inp_norm_b);

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
                // utils.debug_print(q, "q rope");

                k = ggml_rope_inplace(ctx0, k, inp_pos, n_rot, 0);
                k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
                // utils.debug_print(k, "k rope");

                ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
                ggml_mul_mat_set_prec(kq, GGML_PREC_F32); // mimic behavior of llama.cpp
                kq = ggml_scale_inplace(ctx0, kq, 1.0f / std::sqrt(n_embd_head));
                ggml_tensor * kq_masked = ggml_diag_mask_inf_inplace(ctx0, kq, n_tokens);
                kq = ggml_soft_max_inplace(ctx0, kq_masked);
                // utils.debug_print(kq, "kq softmax");

                v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

                ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
                kqv = ggml_reshape_3d(ctx0, kqv, n_embd_head, n_tokens, mimi_config.n_head);
                kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                kqv = ggml_cont_2d(ctx0, kqv, mimi_config.n_embd, n_tokens);
                // utils.debug_print(kqv, "kqv");
                // utils.debug_print(ggml_sum(ctx0, kqv), "kqv_sum");

                x = ggml_mul_mat(ctx0, layer.attn_o, kqv);
            }

            // residual
            x = ggml_mul(ctx0, x, layer.attn_layer_scale);
            x = ggml_add(ctx0, x, residual);
            // utils.debug_print(x, "after_attn_%d", i);

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
            // utils.debug_print(x, "output_layer_%d", i);
            // utils.debug_print(ggml_sum(ctx0, x), "output_layer_%d_sum", i); i++;
        }

        return x;
    }
};

struct mimi_residual_vector_quantizer {
    struct component {
        ggml_tensor * codebook_embed_sum;
        ggml_tensor * codebook_cluster_usage;
        ggml_tensor * get_embd(ggml_context * ctx0) {
            // TODO: do this at conversion time
            ggml_tensor * tmp = ggml_cont(ctx0, ggml_transpose(ctx0, codebook_cluster_usage));
            tmp = ggml_clamp(ctx0, tmp, mimi_config.norm_eps, FLT_MAX);
            return ggml_div(ctx0, codebook_embed_sum, tmp);
        }
    };

    ggml_tensor * semantic_inp_proj;
    std::vector<component> semantic_components;
    ggml_tensor * semantic_out_proj;

    ggml_tensor * acoustic_inp_proj;
    std::vector<component> acoustic_components;
    ggml_tensor * acoustic_out_proj;

    mimi_residual_vector_quantizer(ggml_easy::ctx & ctx) {
        semantic_inp_proj = ctx.get_weight("quantizer.semantic_rvq.input_proj.weight");
        semantic_out_proj = ctx.get_weight("quantizer.semantic_rvq.output_proj.weight");
        for (int i = 0; i < mimi_config.n_semantic_components; i++) {
            semantic_components.push_back({
                .codebook_embed_sum     = ctx.get_weight("quantizer.semantic_rvq.layers.%d.codebook.embed_sum",     i),
                .codebook_cluster_usage = ctx.get_weight("quantizer.semantic_rvq.layers.%d.codebook.cluster_usage", i),
            });
        }
        acoustic_inp_proj = ctx.get_weight("quantizer.acoustic_rvq.input_proj.weight");
        acoustic_out_proj = ctx.get_weight("quantizer.acoustic_rvq.output_proj.weight");
        for (int i = 0; i < mimi_config.n_acoustic_components; i++) {
            acoustic_components.push_back({
                .codebook_embed_sum     = ctx.get_weight("quantizer.acoustic_rvq.layers.%d.codebook.embed_sum",     i),
                .codebook_cluster_usage = ctx.get_weight("quantizer.acoustic_rvq.layers.%d.codebook.cluster_usage", i),
            });
        }
    }

    // 🆘🆘🆘🆘🆘 FIXME: this does not work correcly, about 50% of the output codes are incorrect
    ggml_tensor * encode(ggml_context * ctx0, ggml_easy::ctx::build_utils & utils, ggml_tensor * input) {
        int64_t n_embd           = input->ne[1];
        int64_t n_codes_per_embd = (semantic_components.size() + acoustic_components.size());
        ggml_tensor * codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_embd, n_codes_per_embd);
        ggml_set_input(codes);
        ggml_set_name(codes, "codes");

        size_t pos = 0;
        {
            // semantic
            ggml_tensor * proj = ggml_reshape_2d(ctx0, semantic_inp_proj,
                semantic_inp_proj->ne[1], semantic_inp_proj->ne[2]); // TODO: do this at conversion time
            ggml_tensor * x = ggml_mul_mat(ctx0, proj, input);
            for (size_t i = 0; i < semantic_components.size(); i++) {
                ggml_tensor * codebook = semantic_components[i].get_embd(ctx0);
                codes = ggml_lookup_vectors(utils, ctx0, codebook, x, codes, pos);
                ggml_build_forward_expand(utils.gf, codes);
                pos += n_embd*ggml_element_size(codes);
            }
        }

        {
            // acoustic
            ggml_tensor * proj = ggml_reshape_2d(ctx0, acoustic_inp_proj,
                acoustic_inp_proj->ne[1], acoustic_inp_proj->ne[2]); // TODO: do this at conversion time
            ggml_tensor * x = ggml_mul_mat(ctx0, proj, input);
            for (size_t i = 0; i < acoustic_components.size(); i++) {
                ggml_tensor * codebook = acoustic_components[i].get_embd(ctx0);
                codes = ggml_lookup_vectors(utils, ctx0, codebook, x, codes, pos);
                ggml_build_forward_expand(utils.gf, codes);
                pos += n_embd*ggml_element_size(codes);
            }
        }

        return codes;
    }

    // the input has shape [n_codes, n_codes_per_embd]
    // first row is semantic, the rest are acoustic
    // example: [ [semantic], [acoustic1], [acoustic2], ... ]
    ggml_tensor * decode(ggml_context * ctx0, ggml_easy::ctx::build_utils & utils, ggml_tensor * input) {
        GGML_ASSERT(input->type == GGML_TYPE_I32);

        size_t  n_semantic       = semantic_components.size();
        int64_t n_codes_per_embd = (n_semantic + acoustic_components.size());
        int64_t n_codes          = input->ne[0] / n_codes_per_embd;
        
        GGML_ASSERT(input->ne[0] % n_codes_per_embd == 0);

        ggml_tensor * out_s = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mimi_config.codebook_dim, n_codes);
        ggml_tensor * out_a = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mimi_config.codebook_dim, n_codes);
        out_s = ggml_scale(ctx0, out_s, 0.0f); // clear
        out_a = ggml_scale(ctx0, out_a, 0.0f); // clear

        for (size_t ir = 0; ir < n_codes_per_embd; ir++) {
            ggml_tensor * row = ggml_view_1d(ctx0, input, n_codes, ir*n_codes*ggml_element_size(input));
            if (ir < n_semantic) {
                // semantic
                ggml_tensor * codebook = semantic_components[ir].get_embd(ctx0);
                ggml_tensor * embd = ggml_get_rows(ctx0, codebook, row);
                out_s = ggml_add(ctx0, out_s, embd);
            } else {
                // acoustic
                ggml_tensor * codebook = acoustic_components[ir-n_semantic].get_embd(ctx0);
                ggml_tensor * embd = ggml_get_rows(ctx0, codebook, row);
                out_a = ggml_add(ctx0, out_a, embd);
            }
        }

        ggml_tensor * proj_s = ggml_reshape_2d(ctx0, semantic_out_proj,
            semantic_out_proj->ne[1], semantic_out_proj->ne[2]); // TODO: do this at conversion time
        ggml_tensor * proj_a = ggml_reshape_2d(ctx0, acoustic_out_proj,
            acoustic_out_proj->ne[1], acoustic_out_proj->ne[2]); // TODO: do this at conversion time

        out_s = ggml_mul_mat(ctx0, proj_s, out_s);
        out_a = ggml_mul_mat(ctx0, proj_a, out_a);

        return ggml_add(ctx0, out_s, out_a);
    }
};

int main() {
    ggml_easy::ctx_params params;
    //params.log_level = GGML_LOG_LEVEL_DEBUG;
    params.max_nodes = 1024*16;
    params.use_gpu = false;
    ggml_easy::ctx ctx(params);

    // ctx.load_gguf("mimi.gguf");
    ctx.load_safetensors("mimi.safetensors", {
        {".acoustic_residual_vector_quantizer", ".acoustic_rvq"},
        {".semantic_residual_vector_quantizer", ".semantic_rvq"},
    });

    // optional: print backend buffer info
    ggml_easy::debug::print_backend_buffer_info(ctx);

    mimi_encoder_decoder           encoder(ctx);
    mimi_encoder_decoder           decoder(ctx);
    mimi_transformer               encoder_transformer(ctx, "encoder", 8);
    mimi_transformer               decoder_transformer(ctx, "decoder", 8);
    mimi_residual_vector_quantizer quantizer(ctx);

    encoder.load_encoder();
    decoder.load_decoder();

    // create cgraph
    ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * input = utils.new_input("input", GGML_TYPE_F32, 2048);

        // encoder
        {
            // SEANET encoder
            ggml_tensor * embeddings = encoder.forward(ctx_gf, utils, input);
            utils.debug_print(embeddings, "embeddings");

            // transformer
            int n_pos = embeddings->ne[0];
            ggml_tensor * pos_enc = utils.new_input("pos_enc", GGML_TYPE_I32, n_pos);
            embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
            embeddings = encoder_transformer.forward(ctx_gf, utils, embeddings, pos_enc);
            utils.debug_print(embeddings, "embeddings_after_transformer");

            // downsample
            embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
            embeddings = mimi_conv_1d(utils, ctx_gf, embeddings, ctx.get_weight("downsample.conv.weight"), nullptr, 2, 1, false);
            utils.debug_print(embeddings, "downsample");

            // residual vector quantizer
            embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
            embeddings = quantizer.encode(ctx_gf, utils, embeddings);

            //utils.debug_print_full(embeddings, "output_codes");
            utils.mark_output(embeddings, "output_codes");
        }

        // decoder
        {
            ggml_tensor * inp_dec = utils.new_input("inp_dec", GGML_TYPE_I32, 3 * 32);
            ggml_tensor * embeddings = quantizer.decode(ctx_gf, utils, inp_dec);
            utils.debug_print(embeddings, "read from codebook");

            // upsample
            embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
            embeddings = mimi_conv_transpose_1d(utils, ctx_gf, embeddings, ctx.get_weight("upsample.conv.weight"), nullptr, 2, 1, true);
            utils.debug_print(embeddings, "upscaled");

            // transformer
            int n_pos = embeddings->ne[0];
            ggml_tensor * pos_dec = utils.new_input("pos_dec", GGML_TYPE_I32, n_pos);
            embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
            embeddings = decoder_transformer.forward(ctx_gf, utils, embeddings, pos_dec);
            utils.debug_print(embeddings, "embeddings_after_transformer");

            // SEANET decoder
            embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
            ggml_tensor * output = decoder.forward(ctx_gf, utils, embeddings);
            utils.debug_print(output, "output decoded");
        }
    });

    // equivalent to python code: torch.ones((1, 1, 2048))
    ctx.set_tensor_data("input", [](int, int, int, int) { return 1.0f; });

    // position data
    std::vector<int> pos_data(1024);
    for (int i = 0; i < (int)pos_data.size(); i++) {
        pos_data[i] = i;
    }
    ctx.set_tensor_data("pos_enc", pos_data.data());
    ctx.set_tensor_data("pos_dec", pos_data.data());

    // inp_dec data
    // equivalent to python code: torch.tensor([[ [i, i+1, i+2] for i in range(0, 3*32, 3) ]], dtype=torch.long)
    std::vector<int> inp_dec(3 * 32);
    for (size_t i = 0; i < inp_dec.size(); i++) {
        inp_dec[i] = i;
    }
    ctx.set_tensor_data("inp_dec", inp_dec.data());

    ctx.compute();

    // print result
    //ggml_easy::debug::print_tensor_data(result_tensor, result_data.data());

    return 0;
}
