#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>
#include <thread>
#include <cmath>

struct ultravox_encoder {
    float norm_eps = 1e-5;
    int n_head = 20;
    int n_embd;
    int n_ctx = 1500;

    ggml_tensor * position_embeddings;

    ggml_tensor * conv1d_1_w;
    ggml_tensor * conv1d_1_b;
    ggml_tensor * conv1d_2_w;
    ggml_tensor * conv1d_2_b;

    ggml_tensor * post_ln_w;
    ggml_tensor * post_ln_b;

    // projector
    ggml_tensor * mm_norm_pre_w;
    ggml_tensor * mm_norm_mid_w;
    ggml_tensor * mm_1_w;
    ggml_tensor * mm_2_w;

    struct layer {
        ggml_tensor * ln_1_w;
        ggml_tensor * ln_1_b;

        ggml_tensor * q_w;
        ggml_tensor * q_b;
        ggml_tensor * k_w;
        ggml_tensor * v_w;
        ggml_tensor * v_b;
        ggml_tensor * o_w;
        ggml_tensor * o_b;
        ggml_tensor * ln_2_w;
        ggml_tensor * ln_2_b;

        ggml_tensor * ff_up_w;
        ggml_tensor * ff_up_b;
        ggml_tensor * ff_down_w;
        ggml_tensor * ff_down_b;
    };
    std::vector<layer> layers;

    ultravox_encoder(ggml_easy::ctx & ctx, int n_layers) {
        const char * prefix = "a"; // audio
        position_embeddings = ctx.get_weight("%s.position_embd.weight", prefix);
        n_embd = position_embeddings->ne[0];
        conv1d_1_b = ctx.get_weight("%s.conv1d.1.bias",   prefix);
        conv1d_1_w = ctx.get_weight("%s.conv1d.1.weight", prefix);
        conv1d_2_b = ctx.get_weight("%s.conv1d.2.bias",   prefix);
        conv1d_2_w = ctx.get_weight("%s.conv1d.2.weight", prefix);
        post_ln_w  = ctx.get_weight("%s.post_ln.bias",    prefix);
        post_ln_b  = ctx.get_weight("%s.post_ln.weight",  prefix);

        mm_norm_pre_w = ctx.get_weight("mm.%s.norm_pre.weight", prefix);
        mm_norm_mid_w = ctx.get_weight("mm.%s.norm_mid.weight", prefix);
        mm_1_w = ctx.get_weight("mm.%s.mlp.1.weight", prefix);
        mm_2_w = ctx.get_weight("mm.%s.mlp.2.weight", prefix);

        for (int il = 0; il < n_layers; il++) {
            layers.push_back({
                .ln_1_w     = ctx.get_weight("%s.blk.%d.ln1.weight", prefix, il),
                .ln_1_b     = ctx.get_weight("%s.blk.%d.ln1.bias",   prefix, il),

                .q_w        = ctx.get_weight("%s.blk.%d.attn_q.weight",   prefix, il),
                .q_b        = ctx.get_weight("%s.blk.%d.attn_q.bias",     prefix, il),
                .k_w        = ctx.get_weight("%s.blk.%d.attn_k.weight",   prefix, il),
                .v_w        = ctx.get_weight("%s.blk.%d.attn_v.weight",   prefix, il),
                .v_b        = ctx.get_weight("%s.blk.%d.attn_v.bias",     prefix, il),
                .o_w        = ctx.get_weight("%s.blk.%d.attn_out.weight", prefix, il),
                .o_b        = ctx.get_weight("%s.blk.%d.attn_out.bias",   prefix, il),
                .ln_2_w     = ctx.get_weight("%s.blk.%d.ln2.weight",   prefix, il),
                .ln_2_b     = ctx.get_weight("%s.blk.%d.ln2.bias",     prefix, il),

                .ff_up_w    = ctx.get_weight("%s.blk.%d.ffn_up.weight", prefix, il),
                .ff_up_b    = ctx.get_weight("%s.blk.%d.ffn_up.bias",   prefix, il),
                .ff_down_w  = ctx.get_weight("%s.blk.%d.ffn_down.weight", prefix, il),
                .ff_down_b  = ctx.get_weight("%s.blk.%d.ffn_down.bias",   prefix, il),
            });
        }
    }
};

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);
    ctx.load_gguf("ultravox.gguf");

    const int n_step  = 1024;
    const int n_mel   = 128;
    const int n_pos   = n_step / 2;

    // model
    ultravox_encoder model(ctx, 32);

    const int n_layer = 32;
    const int n_head  = model.n_head;
    const int n_embd  = model.n_embd;
    const int d_head  = n_embd / n_head;
    const float eps   = model.norm_eps;

    const int proj_stack_factor = 8;

    // create cgraph
    ctx.build_graph([&](ggml_context * ctx0, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * inp_raw   = utils.new_input("inp_raw", GGML_TYPE_F32, n_step, n_mel);
        ggml_tensor * positions = utils.new_input("positions", GGML_TYPE_I32, n_pos);

        ggml_tensor * inp;

        // conv1d block
        {
            // convolution + gelu
            ggml_tensor * cur = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp_raw, 1, 1);
            cur = ggml_add(ctx0, cur, model.conv1d_1_b);

            cur = ggml_gelu(ctx0, cur);

            cur = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
            cur = ggml_add(ctx0, cur, model.conv1d_2_b);

            cur = ggml_gelu(ctx0, cur);
            // transpose
            inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        }

        utils.debug_print(inp, "after conv1d");

        // add position embeddings
        inp = ggml_add(ctx0, inp, ggml_get_rows(ctx0, model.position_embeddings, positions));

        utils.debug_print(inp, "after added pos");

        // iterate layers
        for (int il = 0; il < n_layer; ++il) {
            auto & layer = model.layers[il];
            ggml_tensor * cur = inp;

            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ln_1_w), layer.ln_1_b);

            // attention
            {
                ggml_tensor * q = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.q_w, cur), layer.q_b);
                ggml_tensor * k = ggml_mul_mat(ctx0, layer.k_w, cur); // no bias for key
                ggml_tensor * v = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.v_w, cur), layer.v_b);

                q = ggml_reshape_3d(ctx0, q, d_head, n_head, n_pos);
                k = ggml_reshape_3d(ctx0, k, d_head, n_head, n_pos);
                v = ggml_reshape_3d(ctx0, v, d_head, n_head, n_pos);

                q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
                k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));

                ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
                kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f / std::sqrt(d_head), 0.0f);

                v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

                ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
                //kqv = ggml_reshape_3d(ctx0, kqv, d_head, n_pos, n_head);
                kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                kqv = ggml_cont_2d(ctx0, kqv, n_embd, n_pos);

                cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.o_w, kqv), layer.o_b);
            }

            utils.debug_print(cur, "layer %d after attn", il);

            // residual
            cur = ggml_add(ctx0, cur, inp);

            inp = cur; // inp = residual, cur = hidden_states
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ln_2_w), layer.ln_2_b);

            // mlp
            {
                cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff_up_w, cur), layer.ff_up_b);
                cur = ggml_gelu(ctx0, cur);
                cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff_down_w, cur), layer.ff_down_b);
            }

            utils.debug_print(cur, "layer %d after ffn", il);

            // residual
            cur = ggml_add(ctx0, cur, inp);

            inp = cur;

            utils.debug_print(cur, "layer %d out", il);
            utils.debug_print(ggml_sum(ctx0, cur), "layer %d out", il);
        }

        ggml_tensor * embeddings = inp;

        // output norm
        embeddings = ggml_norm(ctx0, embeddings, eps);
        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.post_ln_w), model.post_ln_b);

        embeddings = utils.new_input("test", GGML_TYPE_F32, 1280, 512);

        utils.debug_print(embeddings, "after output norm");

        // StackAudioFrames
        // https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/blob/main/ultravox_model.py
        {
            int64_t stride = n_embd * proj_stack_factor;
            int64_t padded_len = GGML_PAD(ggml_nelements(embeddings), stride);
            int64_t pad = padded_len - ggml_nelements(embeddings);
            if (pad > 0) {
                embeddings = ggml_view_1d(ctx0, embeddings, ggml_nelements(embeddings), 0);
                embeddings = ggml_pad(ctx0, embeddings, pad, 0, 0, 0);
            }
            embeddings = ggml_view_2d(ctx0, embeddings, stride, padded_len / stride,
                                ggml_row_size(embeddings->type, stride), 0);
        }

        utils.debug_print(embeddings, "after stack");

        // UltravoxProjector
        {
            ggml_tensor * cur = embeddings;
            // pre-norm
            cur = ggml_rms_norm(ctx0, cur, 1e-6);
            cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);

            // ffn in
            cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);

            // swiglu
            {
                int64_t split_point = cur->ne[0] / 2;
                ggml_tensor * x0 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, split_point, cur->ne[1], cur->nb[1], 0));
                ggml_tensor * x1 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, split_point, cur->ne[1], cur->nb[1], split_point * ggml_element_size(cur)));

                // see SwiGLU in ultravox_model.py, the second half passed through is silu, not the first half
                x1 = ggml_silu(ctx0, x1);
                cur = ggml_mul(ctx0, x0, x1);
            }

            utils.debug_print(embeddings, "after swiglu");

            // mid-norm
            cur = ggml_rms_norm(ctx0, cur, 1e-6);
            cur = ggml_mul(ctx0, cur, model.mm_norm_mid_w);

            // ffn out
            cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);

            embeddings = cur;
        }

        utils.debug_print(embeddings, "output");
    });

    // set the input
    {
        std::vector<float> inp_raw(n_mel*n_step, 0.1f);
        for (int i = 0; i < n_step*n_mel; i++) {
            inp_raw[i] = (float)std::sin((float)i)*0.1f;
        }
        ctx.set_tensor_data("inp_raw", inp_raw.data());

        std::vector<int> positions(n_pos);
        for (int i = 0; i < n_pos; i++) positions[i] = i;
        ctx.set_tensor_data("positions", positions.data());

        std::vector<float> test(1280*512, 0.1f);
        for (int i = 0; i < (int)test.size(); i++) test[i] = (float)std::sin((float)i)*0.1f;
        ctx.set_tensor_data("test", test.data());
    }

    // compute
    ctx.compute();

    return 0;
}
