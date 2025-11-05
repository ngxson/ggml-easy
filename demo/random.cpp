#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

/**
 * Random experiment, do not use it
 */





 static std::vector<std::vector<std::vector<float>>> get_1d_sincos_pos_embed_from_grid_new(int embed_dim, const std::vector<std::vector<float>> & pos) {
    GGML_ASSERT(embed_dim % 2 == 0);
    int H = pos.size();
    int W = pos[0].size();

    std::vector<float> omega(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; ++i) {
        omega[i] = 1.0 / pow(10000.0, static_cast<float>(i) / (embed_dim / 2));
    }

    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                float out_value = pos[h][w] * omega[d];
                emb[h][w][d] = sin(out_value);
                emb[h][w][d + embed_dim / 2] = cos(out_value);
            }
        }
    }

    return emb;
}

static std::vector<std::vector<std::vector<float>>> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<std::vector<float>>> & grid) {
    GGML_ASSERT(embed_dim % 2 == 0);
    std::vector<std::vector<std::vector<float>>> emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[0]); // (H, W, D/2)
    std::vector<std::vector<std::vector<float>>> emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[1]); // (H, W, D/2)

    int H = emb_h.size();
    int W = emb_h[0].size();
    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                emb[h][w][d] = emb_h[h][w][d];
                emb[h][w][d + embed_dim / 2] = emb_w[h][w][d];
            }
        }
    }
    return emb;
}

static std::vector<std::vector<float>> get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int> image_size) {
    int grid_h_size = image_size.first;
    int grid_w_size = image_size.second;

    std::vector<float> grid_h(grid_h_size);
    std::vector<float> grid_w(grid_w_size);

    for (int i = 0; i < grid_h_size; ++i) {
        grid_h[i] = static_cast<float>(i);
    }
    for (int i = 0; i < grid_w_size; ++i) {
        grid_w[i] = static_cast<float>(i);
    }

    std::vector<std::vector<float>> grid(grid_h_size, std::vector<float>(grid_w_size));
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid[h][w] = grid_w[w];
        }
    }
    std::vector<std::vector<std::vector<float>>> grid_2d = {grid, grid};
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid_2d[0][h][w] = grid_h[h];
            grid_2d[1][h][w] = grid_w[w];
        }
    }

    std::vector<std::vector<std::vector<float>>> pos_embed_3d = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d);

    int H = image_size.first;
    int W = image_size.second;
    std::vector<std::vector<float>> pos_embed_2d(H * W, std::vector<float>(embed_dim));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            pos_embed_2d[w * H + h] = pos_embed_3d[h][w];
        }
    }

    return pos_embed_2d;
}




int compare_minicpmv_pos_embd_ggml_cpp(int n_embd, int nx, int ny) {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);

    {
        ctx.build_graph([&](ggml_context * ctx0, ggml_cgraph * gf, auto & utils) {
            GGML_ASSERT(n_embd % 4 == 0);
            int n_pos = nx * ny;
            ggml_tensor * pos_x = utils.new_input("pos_x", GGML_TYPE_F32, 1, n_pos);
            ggml_tensor * pos_y = utils.new_input("pos_y", GGML_TYPE_F32, 1, n_pos);
            ggml_tensor * omega = utils.new_input("omega", GGML_TYPE_F32, n_embd/4);
            utils.debug_print(pos_x, "pos_x");
            utils.debug_print(pos_y, "pos_y");
            utils.debug_print(omega, "omega");

            // outer product
            ggml_tensor * omega_b = ggml_repeat_4d(ctx0, omega, n_embd/4, n_pos, 1, 1); // n_pos rows
            ggml_tensor * theta_x = ggml_mul(ctx0, omega_b, pos_x);
            ggml_tensor * theta_y = ggml_mul(ctx0, omega_b, pos_y);

            // sin and cos
            ggml_tensor * pos_embd_x = ggml_concat(
                ctx0,
                ggml_sin(ctx0, theta_x),
                ggml_cos(ctx0, theta_x),
                0 // concat on first dim
            );
            ggml_tensor * pos_embd_y = ggml_concat(
                ctx0,
                ggml_sin(ctx0, theta_y),
                ggml_cos(ctx0, theta_y),
                0 // concat on first dim
            );

            ggml_tensor * pos_embd = ggml_concat(ctx0, pos_embd_x, pos_embd_y, 0);
            utils.debug_print(pos_embd, "pos_embd");

            // cpp version for comparison
            ggml_tensor * cpp = utils.new_input("cpp", GGML_TYPE_F32, n_embd, nx*ny);
            utils.debug_print(cpp, "cpp", true);

            ggml_tensor * x = ggml_sub(ctx0, pos_embd, cpp);
            x = ggml_abs(ctx0, ggml_sum(ctx0, x));
            utils.debug_print(x, "diff");
        });

        // set input
        std::vector<float> tmp(nx * ny);
        for (int i = 0; i < nx * ny; ++i) {
            tmp[i] = static_cast<float>(i % nx);
        }
        ctx.set_tensor_data("pos_x", tmp.data());
        for (int i = 0; i < nx * ny; ++i) {
            tmp[i] = static_cast<float>(i / nx);
        }
        ctx.set_tensor_data("pos_y", tmp.data());
        tmp.resize(n_embd / 4);
        const float base_freq = 10000.0f;
        for (int i = 0; i < n_embd / 4; ++i) {
            tmp[i] = 1.0f / std::pow(base_freq, static_cast<float>(i) / (n_embd / 4));
        }
        ctx.set_tensor_data("omega", tmp.data());

        // cpp version for comparison
        auto pos_embed_t = get_2d_sincos_pos_embed(n_embd, std::make_pair(nx, ny));
        std::vector<float> pos_embed(nx * ny * n_embd);
        for(int i = 0; i < nx * ny; ++i){
            for(int j = 0; j < n_embd; ++j){
                pos_embed[i * n_embd + j] = pos_embed_t[i][j];
            }
        }
        ctx.set_tensor_data("cpp", pos_embed.data());


        ctx.compute();

        auto result = ctx.get_tensor_data("diff");
        ggml_tensor * result_tensor        = result.first;
        std::vector<uint8_t> & result_data = result.second;
        float * result_data_f32 = (float *) result_data.data();
        float diff = result_data_f32[0];
        std::cout << "diff = " << diff << std::endl;
        GGML_ASSERT(diff < 1e-4);
    }

    return 0;
}

int main() {
    compare_minicpmv_pos_embd_ggml_cpp(8, 3, 2);
    compare_minicpmv_pos_embd_ggml_cpp(16, 5, 4);
    compare_minicpmv_pos_embd_ggml_cpp(64, 4, 7);
    compare_minicpmv_pos_embd_ggml_cpp(128, 8, 16);
    std::cout << "all tests passed" << std::endl;
    return 0;
}
