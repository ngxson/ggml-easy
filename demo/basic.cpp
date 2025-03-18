#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;
    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };
    const int rows_B = 3, cols_B = 2;
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    // create cgraph
    ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * a = utils.new_input("a", GGML_TYPE_F32, cols_A, rows_A);
        ggml_tensor * b = utils.new_input("b", GGML_TYPE_F32, cols_B, rows_B);
        ggml_tensor * result = ggml_mul_mat(ctx_gf, a, b);
        utils.mark_output("result", result);
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
