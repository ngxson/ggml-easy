//
//  ggml-easy.hpp
//
//  Copyright (c) 2025 Xuan-Son Nguyen. All rights reserved.
//  MIT License
//

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <vector>
#include <cinttypes>
#include <fstream>
#include <functional>

#define LOG_INF(...) do { fprintf(stdout, __VA_ARGS__); } while (0)
#define LOG_WRN(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_ERR(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_DBG(...) do { fprintf(stdout, __VA_ARGS__); } while (0)

namespace ggml_easy {

struct ctx_params {
    bool use_gpu = true;
    int max_nodes = 8192;
};

struct ctx {
    gguf_context * ctx_gguf = nullptr;
    ggml_context * ctx_data = nullptr;
    ggml_context * ctx_gf   = nullptr;

    std::vector<ggml_tensor *> tensors;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend     = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf  = nullptr;

    ggml_backend_sched_ptr sched;

    /**
     * Construct a new ctx object
     * If use_gpu is true, the GPU backend will be used, otherwise the CPU backend will be used
     */
    ctx(const ctx_params & params) {
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        backend     = params.use_gpu
                        ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
                        : nullptr;
    
        if (backend) {
            LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: CLIP using CPU backend\n", __func__);
        }
    
        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));
    
        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), params.max_nodes, false)
        );

        buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
    }

    /**
     * Load a GGUF model file
     * The tensors will be loaded into the context and can be accessed via `ctx.tensors`
     * The GGUF metadata will be loaded into `ctx.ctx_gguf`
     */
    void load_gguf(std::string & fname) {
        ggml_context * meta = nullptr;

        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_init_from_file(fname.c_str(), params);

        // load tensors
        const int n_tensors = gguf_get_n_tensors(ctx_gguf);

        std::vector<uint8_t> read_buf;
        ggml_init_params ggml_params = {
            /*.mem_size   =*/ (n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ctx_data = ggml_init(ggml_params);
        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            ggml_free(meta);
            throw std::runtime_error("cannot open model file for loading tensors");
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            ggml_tensor * t = ggml_get_tensor(meta, name);
            ggml_tensor * cur = ggml_dup_tensor(ctx_data, t);
            ggml_set_name(cur, name);
            tensors.push_back(cur);
        }

        // alloc memory and offload data
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
            const size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                ggml_free(meta);
                throw std::runtime_error("failed to seek for tensor: " + std::string(name));
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buft_is_host(buft)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        LOG_INF("%s: Loaded %d tensors\n", __func__, n_tensors);
        fin.close();

        ggml_free(meta);
    }

    /**
     * Build a cgraph using the given builder function
     */
    ggml_cgraph * build_graph(std::function<void(ggml_context *, ggml_cgraph *)> builder_fn) {
        ggml_free(ctx_gf);
        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute_meta.size(),
            /*.mem_buffer =*/ buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };

        ctx_gf = ggml_init(params);
        struct ggml_cgraph * gf = ggml_new_graph(ctx_gf);

        ggml_backend_sched_reset(sched.get());
        builder_fn(ctx_gf, gf);
        ggml_backend_sched_alloc_graph(sched.get(), gf);

        return gf;
    }

    /**
     * Compute the given cgraph
     */
    ggml_status compute(ggml_cgraph * gf) {
        return ggml_backend_sched_graph_compute(sched.get(), gf);
    }

    /**
     * Set the data of a tensor by name
     */
    void set_tensor_data(const std::string & name, const void * data) {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error("tensor not found: " + name);
        }
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    }

    /**
     * Set the data of a tensor by name using a function.
     * 
     * Example usage:
     * 
     * ```
     * ctx.set_tensor_data("x", [](int i0, int i1, int i2, int i3) {
     *     return i0 + i1 + i2 + i3;
     * });
     * ```
     */
    void set_tensor_data(const std::string & name, std::function<float(int, int, int, int)> data_fn) {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error("tensor not found: " + name);
        }
        if (t->type != GGML_TYPE_F32) {
            throw std::runtime_error("tensor type must be GGML_TYPE_F32");
        }
        std::vector<float> data(ggml_nelements(t));
        for (int d3 = 0; d3 < t->ne[3]; ++d3) {
            for (int d2 = 0; d2 < t->ne[2]; ++d2) {
                for (int d1 = 0; d1 < t->ne[1]; ++d1) {
                    for (int d0 = 0; d0 < t->ne[0]; ++d0) {
                        int i = d3 * t->nb[3] + d2 * t->nb[2] + d1 * t->nb[1] + d0 * t->nb[0];
                        data[i] = data_fn(d0, d1, d2, d3);
                    }
                }
            }
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
    }

    /**
     * Get the data of a tensor by name.
     * 
     * Example usage:
     * 
     * ```
     * auto result = ctx.get_tensor_data("result");
     * ggml_tensor * result_tensor        = result.first;
     * std::vector<uint8_t> & result_data = result.second;
     * float * result_data_f32 = (float *) result_data.data();
     * ```
     */
    std::pair<ggml_tensor *, std::vector<uint8_t>> get_tensor_data(const std::string & name) {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error("tensor not found: " + name);
        }
        std::vector<uint8_t> data(ggml_nbytes(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        return std::make_pair(t, data);
    }

    ~ctx() {
        ggml_free(ctx_data);
        gguf_free(ctx_gguf);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        if (backend_cpu != backend) {
            ggml_backend_free(backend_cpu);
        }
    }
};

namespace debug {
    void print_backend_buffer_info(ctx & gctx) {
        for (size_t i = 0; i < gctx.backend_ptrs.size(); ++i) {
            ggml_backend_t backend = gctx.backend_ptrs[i];
            ggml_backend_buffer_type_t buft = gctx.backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(gctx.sched.get(), backend);
            if (size > 1) {
                LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }
    }

    static void print_tensor_data(ggml_tensor * t, uint8_t * data, int64_t n = 3) {
        ggml_type type = t->type;
        int64_t * ne = t->ne;
        size_t * nb = t->nb;
        for (int64_t i3 = 0; i3 < ne[3]; i3++) {
            LOG_DBG("                                     [\n");
            for (int64_t i2 = 0; i2 < ne[2]; i2++) {
                if (i2 == n && ne[2] > 2*n) {
                    LOG_DBG("                                      ..., \n");
                    i2 = ne[2] - n;
                }
                LOG_DBG("                                      [\n");
                for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                    if (i1 == n && ne[1] > 2*n) {
                        LOG_DBG("                                       ..., \n");
                        i1 = ne[1] - n;
                    }
                    LOG_DBG("                                       [");
                    for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                        if (i0 == n && ne[0] > 2*n) {
                            LOG_DBG("..., ");
                            i0 = ne[0] - n;
                        }
                        size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                        float v;
                        if (type == GGML_TYPE_F16) {
                            v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                        } else if (type == GGML_TYPE_F32) {
                            v = *(float *) &data[i];
                        } else if (type == GGML_TYPE_I32) {
                            v = (float) *(int32_t *) &data[i];
                        } else if (type == GGML_TYPE_I16) {
                            v = (float) *(int16_t *) &data[i];
                        } else if (type == GGML_TYPE_I8) {
                            v = (float) *(int8_t *) &data[i];
                        } else {
                            GGML_ABORT("fatal error");
                        }
                        LOG_DBG("%12.4f", v);
                        if (i0 < ne[0] - 1) LOG_DBG(", ");
                    }
                    LOG_DBG("],\n");
                }
                LOG_DBG("                                      ],\n");
            }
            LOG_DBG("                                     ]\n");
            //LOG_DBG("                                     sum = %f\n", sum);
        }
    }
} // namespace debug

} // namespace ggml_easy
