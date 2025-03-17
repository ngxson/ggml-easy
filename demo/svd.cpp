#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>
#include <ctime>
#include <random>

const time_t seed = std::time(0);

const int   rank   = 4;
const float delta  = 0.001;
const float eps    = 0.97;
const float lambda = 2;

const int rows_A = 3;
const int cols_A = 2;
float matrix_A[rows_A * cols_A] = {
    1, 2,
    3, 4,
    5, 6,
};

/**
 * This program computes the singular value decomposition (SVD) of a matrix A using the power iteration method.
 * The matrix A is decomposed into the product of three matrices U, S, and V such that A = U * S * V^T.
 *
 * After decomposed the matrix A, the program reconstructs the matrix A using the decomposed matrices U, S, and V.
 * The reconstructed matrix should be the same as the original matrix A.
 * 
 * Ref python implementation: https://gist.github.com/Zhenye-Na/cbf4e534b44ef94fdbad663ef56dd333
 */

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);

    const int n_iters = log(4.0f * log(2.0f * rows_A / delta) / (eps * delta)) / (2 * lambda);
    printf("n_iters = %d\n", n_iters);

    auto norm = [&](ggml_context * ctx_gf, ggml_tensor * t) {
        return ggml_sqrt(ctx_gf, ggml_sum_rows(ctx_gf, ggml_sqr(ctx_gf, t)));
    };

    auto power_iteration = [&](ggml_context * ctx_gf, ggml_cgraph * gf, ggml_tensor * A, ggml_tensor * x) {
        ggml_tensor * B = ggml_mul_mat(ctx_gf, A, A);
        for (int i = 0; i < n_iters; i++) {
            x = ggml_mul_mat(ctx_gf, B, x);
            x = ggml_div(ctx_gf, x, norm(ctx_gf, x));
        }
        ggml_tensor * v = x;
        ggml_tensor * AT = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, A));
        ggml_tensor * A_v = ggml_mul_mat(ctx_gf, AT, v);
        ggml_tensor * s = norm(ctx_gf, A_v);
        ggml_tensor * u = ggml_div(ctx_gf, A_v, s);
        return std::vector<ggml_tensor *>{u, s, v};
    };

    ggml_cgraph * gf = ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf) {
        ggml_tensor * A = ggml_new_tensor_2d(ctx_gf, GGML_TYPE_F32, cols_A, rows_A);
        ggml_set_name(A, "A");
        ggml_set_input(A);
        ggml_tensor * x = ggml_new_tensor_1d(ctx_gf, GGML_TYPE_F32, rows_A);
        ggml_set_name(x, "x");
        ggml_set_input(x);

        // normalize x
        x = ggml_div(ctx_gf, x, norm(ctx_gf, x));

        ggml_tensor * out_u; // final shape: [cols_A, rank]
        ggml_tensor * out_s; // final shape: [rank]
        ggml_tensor * out_v; // final shape: [rows_A, rank]
        
        for (int i = 0; i < rank; i++) {
            std::vector<ggml_tensor *> result = power_iteration(ctx_gf, gf, A, x);
            ggml_tensor * u = result[0];
            ggml_tensor * s = result[1];
            ggml_tensor * v = result[2];

            ggml_tensor * vT = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, v));
            ggml_tensor * uT = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, u));
            ggml_tensor * A_minus = ggml_mul(ctx_gf, ggml_mul_mat(ctx_gf, uT, vT), s);
            A = ggml_add(ctx_gf, A, ggml_scale(ctx_gf, A_minus, -1));

            if (i == 0) {
                out_u = u;
                out_s = s;
                out_v = v;
            } else {
                out_u = ggml_concat(ctx_gf, out_u, u, 1);
                out_s = ggml_concat(ctx_gf, out_s, s, 0);
                out_v = ggml_concat(ctx_gf, out_v, v, 1);
            }
        }

        ggml_set_name(out_u, "u");
        ggml_set_name(out_s, "s");
        ggml_set_name(out_v, "v");

        ggml_build_forward_expand(gf, out_u);
        ggml_build_forward_expand(gf, out_s);
        ggml_build_forward_expand(gf, out_v);
    });

    // set data
    {
        ctx.set_tensor_data("A", matrix_A);

        // initialize eigenvector to random vector
        std::default_random_engine generator(static_cast<unsigned int>(seed));
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        ctx.set_tensor_data("x", [&](int, int, int, int) {
            return distribution(generator);
        });
    }

    // optional: print backend buffer info
    ggml_easy::debug::print_backend_buffer_info(ctx);

    // compute
    ggml_status status = ctx.compute(gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::cerr << "error: ggml compute return status: " << status << std::endl;
        return 1;
    }

    // get result
    auto print_result = [&](ggml_easy::ctx & ctx, const char * tensor_name) {
        auto result = ctx.get_tensor_data(tensor_name);
        ggml_tensor * result_tensor        = result.first;
        std::vector<uint8_t> & result_data = result.second;
        std::cout << "\n\n" << tensor_name << ":\n";
        ggml_easy::debug::print_tensor_data(result_tensor, result_data.data());
        return result_data;
    };

    std::vector<uint8_t> data_u = print_result(ctx, "u");
    std::vector<uint8_t> data_s = print_result(ctx, "s");
    std::vector<uint8_t> data_v = print_result(ctx, "v");


    // VERIFY THE RESULT!!


    gf = ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf) {
        ggml_tensor * u = ggml_new_tensor_2d(ctx_gf, GGML_TYPE_F32, cols_A, rank);
        ggml_tensor * s = ggml_new_tensor_1d(ctx_gf, GGML_TYPE_F32, rank);
        ggml_tensor * v = ggml_new_tensor_2d(ctx_gf, GGML_TYPE_F32, rows_A, rank);

        ggml_set_name(u, "u");
        ggml_set_name(s, "s");
        ggml_set_name(v, "v");

        ggml_set_input(v);
        ggml_set_input(s);
        ggml_set_input(u);

        s = ggml_diag(ctx_gf, s);

        ggml_tensor * uT = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, u));
        ggml_tensor * vT = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, v));
        ggml_tensor * temp = ggml_mul_mat(ctx_gf, s, uT);
        ggml_tensor * result = ggml_mul_mat(ctx_gf, temp, vT);

        ggml_set_name(result, "result");
        ggml_build_forward_expand(gf, result);
    });

    ctx.set_tensor_data("u", data_u.data());
    ctx.set_tensor_data("s", data_s.data());
    ctx.set_tensor_data("v", data_v.data());

    status = ctx.compute(gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::cerr << "error: ggml compute return status: " << status << std::endl;
        return 1;
    }

    print_result(ctx, "result");

    return 0;
}
