#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

/**
 * MobileNetV5 experiments
 */

using get_tensor_fn = std::function<ggml_tensor * (const std::string & name)>;
using callback_fn   = std::function<void(ggml_tensor * cur, const char * name, int il)>;


static ggml_tensor * rms_norm_act_2d(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * weight,
        int n_groups,
        bool apply_act,
        callback_fn & cb) {
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 1, 2, 0, 3)); // first dim is now channels
    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 2, 0, 1, 3)); // back to original order
    if (weight != nullptr) {
        cur = ggml_mul(ctx, cur, ggml_reshape_3d(ctx, weight, 1, 1, weight->ne[0]));
        cb(cur, "rms_norm_act.norm_w", -1);
    }
    if (apply_act) {
        cur = ggml_gelu_erf(ctx, cur);
        cb(cur, "rms_norm_act.gelu", -1);
    }
    return cur;
}

static ggml_tensor * conv2d_pw(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    GGML_ASSERT(a->ne[0] == 1 && a->ne[1] == 1); // pointwise conv expects 1x1 kernel
    //return ggml_conv_2d(ctx, a, b, 1, 1, 0, 0, 1, 1);
    int w = b->ne[0];
    int h = b->ne[1];
    int c = b->ne[2];
    GGML_ASSERT(b->ne[3] == 1); // not support batch size > 1 for now
    ggml_tensor * cur = ggml_cont(ctx, ggml_permute(ctx, b, 1, 2, 0, 3)); // first dim is now channels
    a = ggml_reshape_2d(ctx, a, a->ne[2], a->ne[3]);
    cur = ggml_mul_mat(ctx, a, cur);
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 2, 0, 1, 3)); // back to original order
    return cur;
}

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);

    int n_inp_elem = 0;

    ctx.build_graph([&](ggml_context * ctx0, ggml_cgraph * gf, auto & utils) {
        callback_fn cb = [&](ggml_tensor * cur, const char * name, int) {
            utils.debug_print(cur, name);
        };

        ggml_tensor * inp = utils.new_input("inp", GGML_TYPE_F32, 3, 3, 4);
        n_inp_elem = ggml_nelements(inp);

        ggml_tensor * cur = ggml_scale(ctx0, inp, 0.1f);
        cb(cur, "inp_scaled", -1);

        cur = rms_norm_act_2d(ctx0, cur, nullptr, 4, true, cb);

        ggml_tensor * pw = utils.new_input("pw", GGML_TYPE_F32, 1, 1, 4, 2);
        cur = conv2d_pw(ctx0, pw, inp);
        utils.debug_print(inp, "inp");
        utils.debug_print(pw, "pw");
        cb(cur, "conv_pw", -1);
    });

    std::vector<float> inp_data(n_inp_elem);
    for (int i = 0; i < n_inp_elem; ++i) {
        inp_data[i] = (float)i;
    }

    ctx.set_tensor_data("inp", inp_data.data());
    ctx.set_tensor_data("pw", inp_data.data());
    ctx.compute();

    return 0;
}
