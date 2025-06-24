#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

/**
 * Experiment with auto encoder-decoder architecture using GGML.
 */

ggml_tensor * ggml_swish(ggml_context * ctx, ggml_tensor * a) {
    return ggml_mul(ctx, a, ggml_sigmoid(ctx, a));
}

ggml_tensor * ggml_conv_2d_ext(ggml_context * ctx,
                               ggml_tensor * weight,
                               ggml_tensor * bias,
                               ggml_tensor * cur,
                               int s, int p, int d) {
    cur = ggml_conv_2d(ctx, cur, weight, s, s, p, p, d, d);
    cur = ggml_add(ctx, cur,
            ggml_cont(ctx, ggml_transpose(ctx, bias))); // TODO: do at conversion
    return cur;
}

static struct config {
    std::array<int, 4> ch_mult = {1, 2, 4, 4};
} config;

struct resnet_block {
    ggml_tensor * norm1_w;
    ggml_tensor * norm1_b;
    ggml_tensor * conv1_w;
    ggml_tensor * conv1_b;
    ggml_tensor * norm2_w;
    ggml_tensor * norm2_b;
    ggml_tensor * conv2_w;
    ggml_tensor * conv2_b;

    ggml_tensor * nin_shortcut_w = nullptr;
    ggml_tensor * nin_shortcut_b = nullptr;

    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * x) {
        ggml_tensor * cur = x;
    
        cur = ggml_add(ctx, norm1_b,
                ggml_mul(ctx,
                    ggml_group_norm(ctx, cur, 32, 1e-6), norm1_w
                ));
        cur = ggml_swish(ctx, cur);
        cur = ggml_conv_2d_ext(ctx, conv1_w, conv1_b, cur, 1, 1, 1);

        cur = ggml_add(ctx, norm2_b,
                ggml_mul(ctx,
                    ggml_group_norm(ctx, cur, 32, 1e-6), norm1_w
                ));
        cur = ggml_swish(ctx, cur);
        cur = ggml_conv_2d_ext(ctx, conv2_w, conv2_b, cur, 1, 1, 1);

        if (nin_shortcut_w && nin_shortcut_b) {
            x = ggml_conv_2d_ext(ctx, nin_shortcut_w, nin_shortcut_b, x, 1, 0, 1);
        }

        return ggml_add(ctx, x, cur);
    }
};

struct encoder {
    ggml_tensor * conv_in_w;
    ggml_tensor * conv_in_b;

    struct down {
        resnet_block blk0;
        resnet_block blk1;
        ggml_tensor * downsample_w = nullptr;
        ggml_tensor * downsample_b = nullptr;
    };
    std::vector<down> downs;

    ggml_tensor * forward(ggml_context * ctx, ggml_tensor * x) {
        int num_resolutions = config.ch_mult.size();

        x = ggml_conv_2d_ext(ctx, conv_in_w, conv_in_b, x, 1, 1, 1);

        return x;
    }
};

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);
    ctx.load_safetensors("autoencoder.safetensors", {});

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
    ctx.compute();

    return 0;
}
