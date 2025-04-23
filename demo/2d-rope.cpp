#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

/**
 * Experiment with 2D RoPE used on Mistral's Pixtral model
 */

// implementation of the 2D RoPE without adding a new op in ggml
static ggml_tensor * build_rope_2d(
    ggml_cgraph * gf,
    ggml_context * ctx0,
    ggml_tensor * cur,
    ggml_tensor * pos_h,
    ggml_tensor * pos_w,
    const float freq_base
) {
    ggml_tensor * tmp;
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
    const float freq_scale = std::pow(freq_base, (float)-2/n_dim);

    // first half
    {
        cur = ggml_rope_ext_inplace(
            ctx0,
            cur,
            pos_h,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    // second half
    {
        tmp = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            ggml_row_size(cur->type, n_dim),
            ggml_row_size(cur->type, n_dim*n_head),
            n_dim/2 * ggml_element_size(cur));
        tmp = ggml_rope_ext_inplace(
            ctx0,
            tmp,
            pos_w,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            freq_scale,
            0.0f, 1.0f, 0.0f, 0.0f
        );
        // calculate inplace (modify cur directly)
        ggml_build_forward_expand(gf, tmp);
    }

    return cur;
}

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);

    const int n_sz  = 3;
    const int n_pos = n_sz * n_sz;
    const int n_dim = 12;
    const int n_head = 1;

    // create cgraph
    ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * pos_h = utils.new_input("pos_h", GGML_TYPE_I32, n_pos);
        ggml_tensor * pos_w = utils.new_input("pos_w", GGML_TYPE_I32, n_pos);
        ggml_tensor * vector = utils.new_input("vector", GGML_TYPE_F32, n_dim*n_head, n_pos);
        vector = ggml_reshape_3d(ctx_gf, vector, n_dim, n_head, n_pos);
        ggml_tensor * result = build_rope_2d(gf, ctx_gf, vector, pos_h, pos_w, 10000.0f);
        result = ggml_reshape_2d(ctx_gf, result, n_dim*n_head, n_pos);
        utils.mark_output(result, "result");
    });

    // set data
    std::vector<int32_t> positions(n_pos);
    for (int i = 0; i < n_pos; ++i) {
        positions[i] = i / n_sz;
    }
    ctx.set_tensor_data("pos_h", positions.data());
    for (int i = 0; i < n_pos; ++i) {
        positions[i] = i % n_sz;
    }
    ctx.set_tensor_data("pos_w", positions.data());
    ctx.set_tensor_data("vector", [](int i0, int i1, int i2, int i3) {
        //return i0 * 0.1;
        return 1.0;
    });

    // optional: print backend buffer info
    ggml_easy::debug::print_backend_buffer_info(ctx);

    // compute
    ggml_status status = ctx.compute();

    // get result
    auto result = ctx.get_tensor_data("result");
    ggml_tensor * result_tensor        = result.first;
    std::vector<uint8_t> & result_data = result.second;

    // print result
    ggml_easy::debug::print_tensor_data(result_tensor, result_data.data(), 999);

    return 0;
}
