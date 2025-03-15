#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

/**
 * Demo to compare performance of RMS Norm vs Dynamic Tanh (DyT)
 * Paper: https://arxiv.org/abs/2503.10622
 * 
 * Result on my Macbook M3:
 * RMS Norm: 37 ms
 * DyT     : 135 ms
 */

int main() {
    const int n_embd   = 4096;
    const int n_tokens = 1024;
    const int n_run    = 300;

    ggml_easy::ctx_params params;
    params.log_level = GGML_LOG_LEVEL_ERROR;

    // benchmark RMS Norm
    {
        ggml_easy::ctx ctx(params);

        ggml_cgraph * gf = ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf) {
            ggml_tensor * cur = ggml_new_tensor_2d(ctx_gf, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_set_name(cur, "input");
            for (int i = 0; i < n_run; i++) {
                cur = ggml_rms_norm(ctx_gf, cur, 1e-6);
                // skip bias
            }
            ggml_set_name(cur, "result");
            ggml_build_forward_expand(gf, cur);
        });

        std::vector<float> vec(n_embd * n_tokens, 0.5f);
        ctx.set_tensor_data("input", vec.data());

        int64_t t_start = ggml_time_ms();
        ctx.compute(gf);
        int64_t t_end = ggml_time_ms();

        std::cout << "RMS Norm: " << (t_end - t_start) << " ms" << std::endl;
    }

    // benchmark DyT
    {
        ggml_easy::ctx ctx(params);

        ggml_cgraph * gf = ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf) {
            ggml_tensor * cur = ggml_new_tensor_2d(ctx_gf, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_set_name(cur, "input");
            ggml_tensor * alpha = ggml_new_tensor_1d(ctx_gf, GGML_TYPE_F32, n_embd);
            ggml_set_name(alpha, "alpha");
            ggml_tensor * gamma = ggml_new_tensor_1d(ctx_gf, GGML_TYPE_F32, n_embd);
            ggml_set_name(gamma, "gamma");
            for (int i = 0; i < n_run; i++) {
                // DyT(x) = gamma * tanh(alpha * x) + β
                cur = ggml_mul(ctx_gf, cur, alpha);
                cur = ggml_tanh(ctx_gf, cur);
                cur = ggml_mul(ctx_gf, cur, gamma);
                // skip beta
            }
            ggml_set_name(cur, "result");
            ggml_build_forward_expand(gf, cur);
        });

        std::vector<float> vec(n_embd * n_tokens, 0.5f);
        ctx.set_tensor_data("input", vec.data());
        ctx.set_tensor_data("alpha", vec.data());
        ctx.set_tensor_data("gamma", vec.data());

        int64_t t_start = ggml_time_ms();
        ctx.compute(gf);
        int64_t t_end = ggml_time_ms();

        std::cout << "DyT     : " << (t_end - t_start) << " ms" << std::endl;
    }

    return 0;
}
