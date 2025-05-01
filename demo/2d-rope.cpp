#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>
#include <cmath>

/**
 * Experiment with 2D RoPE used on Mistral's Pixtral model
 */

// implementation of the 2D RoPE without adding a new op in ggml
// this is not efficient (use double the memory), but works on all backends
static ggml_tensor * build_rope_2d(
    ggml_context * ctx0,
    ggml_tensor * cur,
    ggml_tensor * pos_a, // first half
    ggml_tensor * pos_b, // second half
    const float freq_base,
    const bool interleave_freq
) {
    const int64_t n_dim  = cur->ne[0];
    const int64_t n_head = cur->ne[1];
    const int64_t n_pos  = cur->ne[2];

    // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
    // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
    // first half of cur will use 1e-0, 1e-2 (even)
    // second half of cur will use 1e-1, 1e-3 (odd)
    // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
    //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
    // then for the second half, we use freq_scale to shift the inv_freq
    //  ^ why? replace (2i) with (2i+1) in the above equation
    const float freq_scale_odd = interleave_freq
                                    ? std::pow(freq_base, (float)-2/n_dim)
                                    : 1.0;

    // first half
    ggml_tensor * first;
    {
        first = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            ggml_row_size(cur->type, n_dim),
            ggml_row_size(cur->type, n_dim*n_head),
            0);
        first = ggml_rope_ext(
            ctx0,
            first,
            pos_a,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    // second half
    ggml_tensor * second;
    {
        second = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            ggml_row_size(cur->type, n_dim),
            ggml_row_size(cur->type, n_dim*n_head),
            n_dim/2 * ggml_element_size(cur));
        second = ggml_cont(ctx0, second); // copy, because ggml_rope don't play well with non-contiguous tensors
        second = ggml_rope_ext(
            ctx0,
            second,
            pos_b,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            freq_scale_odd,
            0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    cur = ggml_concat(ctx0, first, second, 0);
    return cur;
}

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);

    const bool is_llama = true; // false meaning pixtral

    const int n_sz  = 3;
    const int n_pos = n_sz * n_sz + (is_llama ? 1 : 0); // 1 for CLS token
    const int n_dim = 8;
    const int n_head = 1;

    // create cgraph
    ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * pos_h = utils.new_input("pos_h", GGML_TYPE_I32, n_pos);
        ggml_tensor * pos_w = utils.new_input("pos_w", GGML_TYPE_I32, n_pos);
        ggml_tensor * vector = utils.new_input("vector", GGML_TYPE_F32, n_dim*n_head, n_pos);
        vector = ggml_reshape_3d(ctx_gf, vector, n_dim, n_head, n_pos);
        ggml_tensor * result = is_llama
            ? build_rope_2d(ctx_gf, vector, pos_w, pos_h, 10000.0f, false)
            : build_rope_2d(ctx_gf, vector, pos_h, pos_w, 10000.0f, true);
        result = ggml_reshape_2d(ctx_gf, result, n_dim*n_head, n_pos);
        utils.mark_output(result, "result");
    });

    // set data
    if (is_llama) {
        auto py_div_floor = [](int a, int b) { // mimic "//" operator in python for negative numbers
            return (a / b) - ((a % b != 0 && ((a < 0) ^ (b < 0))) ? 1 : 0);
        };
        auto py_mod = [](int a, int b) { // mimic "%" operator in python for negative numbers
            return (a % b + b) % b;
        };
        std::vector<int32_t> positions(n_pos);
        for (int i = 0; i < n_pos; ++i) {
            positions[i] = (i / n_sz) + 1;
            // printf("pos_h[%d] = %d\n", i, positions[i]);
        }
        positions[positions.size() - 1] = 0;
        printf("\n");
        ctx.set_tensor_data("pos_h", positions.data());
        for (int i = 0; i < n_pos; ++i) {
            positions[i] = (i % n_sz) + 1;
            // printf("pos_w[%d] = %d\n", i, positions[i]);
        }
        positions[positions.size() - 1] = 0;
        ctx.set_tensor_data("pos_w", positions.data());
    } else {
        std::vector<int32_t> positions(n_pos);
        for (int i = 0; i < n_pos; ++i) {
            positions[i] = i / n_sz;
        }
        ctx.set_tensor_data("pos_h", positions.data());
        for (int i = 0; i < n_pos; ++i) {
            positions[i] = i % n_sz;
        }
        ctx.set_tensor_data("pos_w", positions.data());
    }
    ctx.set_tensor_data("vector", [](int i0, int i1, int i2, int i3) {
        //return i0 * 0.1;
        return 1.0;
    });

    // compute
    ggml_status status = ctx.compute();

    // get result
    auto result = ctx.get_tensor_data("result");
    ggml_tensor * result_tensor        = result.first;
    std::vector<uint8_t> & result_data = result.second;

    // print result
    ggml_easy::debug::print_tensor_data(result_tensor, result_data.data(), 999);


    //
    // implementation using ggml_rope_multi
    //

    /*{
        // create cgraph
        ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
            ggml_tensor * pos = utils.new_input("pos", GGML_TYPE_I32, n_pos*4);
            ggml_tensor * vector = utils.new_input("vector", GGML_TYPE_F32, n_dim*n_head, n_pos);

            ggml_tensor * cur = ggml_reshape_3d(ctx_gf, vector, n_dim, n_head, n_pos);
            {
                const int n_dim  = cur->ne[0];
                const int n_head = cur->ne[1];
                const int n_pos  = cur->ne[2];
                int sections[4] = {n_dim/2, 1, 0, 0};
                cur = ggml_rope_multi(
                    ctx_gf,
                    cur,
                    pos,        // positions
                    nullptr,    // freq factors
                    n_dim,      // n_dims
                    sections,   // sections
                    GGML_ROPE_TYPE_MROPE,
                    0, 10000.0f,
                    1.0f, 0.0f, 1.0f, 0.0f, 0.0f
                );
            }

            cur = ggml_reshape_2d(ctx_gf, cur, n_dim*n_head, n_pos);
            utils.mark_output(cur, "result");
        });

        // set data
        std::vector<int32_t> positions(n_pos*4, 0);
        for (int i = 0; i < n_pos; ++i) positions[i + n_pos*0] = i / n_sz;
        for (int i = 0; i < n_pos; ++i) positions[i + n_pos*1] = i % n_sz;
        ctx.set_tensor_data("pos", positions.data());
        ctx.set_tensor_data("vector", [](int i0, int i1, int i2, int i3) {
            return 1.0;
        });

        // compute
        ctx.compute();

        // get result
        result = ctx.get_tensor_data("result");
        ggml_tensor * result_tensor        = result.first;
        std::vector<uint8_t> & result_data = result.second;

        // print result
        ggml_easy::debug::print_tensor_data(result_tensor, result_data.data(), 999);
    }*/

    return 0;
}
