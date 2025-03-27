#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

/**
 * This example demonstrates how to perform matrix multiplication using ggml-easy.h
 * 
 * Given 2 matrices A and B, the result matrix C is calculated as follows:
 *   C = (A x B) * 2
 *
 * We will use utils.debug_print() to debug the intermediate result of (A x B)
 * Then, we will use utils.mark_output() to get the final result of C
 *
 * The final result can be printed using ggml_easy::debug::print_tensor_data()
 * Or, can be used to perform further computations
 */


// lookup nearest vector in codebook based on euclidean distance
static ggml_tensor * ggml_lookup_vec(ggml_context * ctx0, ggml_tensor * codebook, ggml_tensor * x) {
    ggml_tensor * tmp = ggml_add(ctx0, codebook, ggml_scale(ctx0, x, -1.0f)); // a - x
    tmp = ggml_mul(ctx0, tmp, tmp); // (a - x) ** 2
    tmp = ggml_sum_rows(ctx0, tmp);
    tmp = ggml_sqrt(ctx0, tmp);
    tmp = ggml_cont(ctx0, ggml_transpose(ctx0, tmp));
    // alternative for argmin
    tmp = ggml_scale(ctx0, tmp, -1.0f);
    tmp = ggml_argmax(ctx0, tmp);
    return tmp;
}

static ggml_tensor * quantize_vectors(ggml_easy::ctx::build_utils & utils, ggml_context * ctx0, ggml_cgraph * gf, ggml_tensor * codebook, ggml_tensor * list_vec) {
    int64_t n_col = list_vec->ne[0];
    int64_t n_row = list_vec->ne[1];
    ggml_tensor * out = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_row);
    for (int64_t ir = 0; ir < n_row; ir++) {
        ggml_tensor * row = ggml_view_1d(ctx0, list_vec, n_col, ir*n_col*ggml_element_size(list_vec));
        ggml_tensor * idx = ggml_lookup_vec(ctx0, codebook, row);
        //utils.debug_print(idx, "idx");
        //ggml_build_forward_expand(gf, idx);
        out = ggml_set_1d(ctx0, out, idx, ir*ggml_element_size(out));
    }
    return out;
}

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 3, cols_A = 2;
    float matrix_A[rows_A * cols_A] = {
        0.9041, 0.0196, -0.3108, -2.4423, -0.4821, 1.059
    };
    const int rows_B = 2, cols_B = 2;
    float matrix_B[rows_B * cols_B] = {
        -0.1763, -0.4713, -0.6986, 1.3702
    };

    // create cgraph
    ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * a = utils.new_input("a", GGML_TYPE_F32, cols_A, rows_A);
        ggml_tensor * b = utils.new_input("b", GGML_TYPE_F32, cols_B, rows_B);

        ggml_tensor * a_mul_b = quantize_vectors(utils, ctx_gf, gf, a, b);
        utils.debug_print(a_mul_b, "a_mul_b");
        ggml_tensor * result = a_mul_b;
        utils.mark_output(result, "result");
        ggml_graph_print(gf);
    });

    // set data
    ctx.set_tensor_data("a", matrix_A);
    ctx.set_tensor_data("b", matrix_B);

    // optional: print backend buffer info
    ggml_easy::debug::print_backend_buffer_info(ctx);

    // compute
    ggml_status status = ctx.compute();
    if (status != GGML_STATUS_SUCCESS) {
        std::cerr << "error: ggml compute return status: " << status << std::endl;
        return 1;
    }

    // get result
    auto result = ctx.get_tensor_data("result");
    ggml_tensor * result_tensor        = result.first;
    std::vector<uint8_t> & result_data = result.second;

    // print result
    ggml_easy::debug::print_tensor_data(result_tensor, result_data.data());

    return 0;
}
