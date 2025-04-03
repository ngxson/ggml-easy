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

#include <limits.h>
#include <vector>
#include <map>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <unordered_map>

namespace ggml_easy {

struct ctx_params {
    bool use_gpu = true;
    int max_nodes = 8192;
    ggml_log_level log_level = GGML_LOG_LEVEL_INFO;
};

void log_cb(ggml_log_level level, const char * text, void * cur_lvl_ptr) {
    ggml_log_level cur_lvl = *(ggml_log_level *) cur_lvl_ptr;
    if (cur_lvl > level) {
        return;
    }
    fputs(text, stderr);
    fflush(stderr);
}

// forward declaration
namespace debug {
    static void print_tensor_shape(ggml_tensor * t);
    static void print_tensor_data(ggml_tensor * t, uint8_t * data, int64_t n = 3);
}

// forward declaration for safetensors (lightweight) JSON parser
struct safetensors_json_parser {
    enum state {
        STATE_ROOT,
        STATE_OBJ_METADATA,
        STATE_OBJ_TENSOR,
    };
    struct tensor {
        std::string name;
        ggml_type type = GGML_TYPE_F32; // only F32, F16, BF16 are supported
        std::array<int64_t, 4> shape = {0, 1, 1, 1}; // row-major order
        uint64_t offset = 0;
        void print() {
            printf("tensor: %-60s, type: %s, shape: [%4" PRId64 ", %4" PRId64 ", %4" PRId64 ", %4" PRId64 "], offset: %" PRIu64 "\n",
                name.c_str(), ggml_type_name(type), shape[0], shape[1], shape[2], shape[3], offset);
        }
    };
    std::vector<tensor> tensors;
    size_t metadata_size = 0;
    safetensors_json_parser(const char * json, size_t metadata_size, std::map<std::string, std::string> name_replace_map);
    uint64_t get_data_offset();
};

std::string string_format(const char * fmt, ...);
void string_replace_all(std::string & s, const std::string & search, const std::string & replace);

////////////////////////////////////////

struct ctx {
    ggml_log_level log_level;

    std::unordered_map<std::string, ggml_tensor *> tensors;

    ggml_cgraph  * gf     = nullptr;
    ggml_context * ctx_gf = nullptr;
    std::vector<uint8_t> buf_compute_meta;
    int max_nodes;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend     = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf  = nullptr;

    ggml_backend_sched_ptr sched;

private:
    // private data members
    struct loaded_gguf {
        gguf_context_ptr ctx_gguf;
        ggml_context_ptr ctx_data;
    };
    std::vector<loaded_gguf> loaded_ggufs;

    struct printed_tensor {
        ggml_tensor * t;
        bool full;
    };
    std::vector<printed_tensor> dbg_printed_tensors;


public:
    /**
     * Construct a new ctx object
     * If use_gpu is true, the GPU backend will be used, otherwise the CPU backend will be used
     */
    ctx(const ctx_params & params) : log_level(params.log_level), max_nodes(params.max_nodes) {
        ggml_log_set(log_cb, &log_level);
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        backend     = params.use_gpu
                        ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
                        : nullptr;
    
        if (backend) {
            log(GGML_LOG_LEVEL_INFO, "%s: using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            log(GGML_LOG_LEVEL_INFO, "%s: using CPU backend\n", __func__);
        }
    
        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));
    
        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false)
        );

        buf_compute_meta.resize(max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());
    }

    /**
     * Get a weight tensor by name, can only be used after model is loaded.
     * Throws an exception if the tensor is not found.
     */
    ggml_tensor * get_weight(const char *fmt, ...) {
        std::vector<char> str(128);
        va_list va;
        va_start(va, fmt);
        vsnprintf(str.data(), 128, fmt, va);
        va_end(va);
        auto it = tensors.find(str.data());
        if (it == tensors.end()) {
            throw std::runtime_error(string_format("weight tensor not found: %s", str.data()));
        }
        return it->second;
    }

    /**
     * Load a GGUF model file
     * The tensors will be loaded into the context and can be accessed via `ctx.get_weight(name)`
     * The GGUF metadata will be loaded into `ctx.ctx_gguf`
     */
    void load_gguf(const char * fname) {
        ggml_context * meta = nullptr;

        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        gguf_context * ctx_gguf = gguf_init_from_file(fname, params);

        // load tensors
        const int n_tensors = gguf_get_n_tensors(ctx_gguf);
        ggml_init_params ggml_params = {
            /*.mem_size   =*/ (n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ggml_context * ctx_data = ggml_init(ggml_params);
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
            tensors.insert({name, cur});
        }

        // alloc memory and offload data
        std::map<ggml_tensor *, uint64_t> offset_map; // empty map, use default value
        if (!load_tensors_to_backend(fin, offset_map, ctx_gguf, ctx_data)) {
            ggml_free(meta);
            throw std::runtime_error("failed to load tensors to backend");
        }
        log(GGML_LOG_LEVEL_INFO, "%s: Loaded %d tensors from %s\n", __func__, n_tensors, fname);
        ggml_free(meta);

        loaded_ggufs.push_back({
            gguf_context_ptr(ctx_gguf),
            ggml_context_ptr(ctx_data),
        });
    }

    /**
     * Load a Safetensors model file
     * The tensors will be loaded into the context and can be accessed via `ctx.get_weight(name)`
     * In some cases, the tensor name is too long and GGML won't accept it. You can provide a name_replace_map to replace the name.
     * For example:
     *      name_replace_map = {{".acoustic_residual_vector_quantizer", ".arvq"}}
     */
    void load_safetensors(const char * fname, std::map<std::string, std::string> name_replace_map) {
        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            throw std::runtime_error("cannot open model file: " + std::string(fname));
        }

        uint64_t metadata_size = 0;
        fin.read(reinterpret_cast<char *>(&metadata_size), sizeof(metadata_size));
        if (metadata_size < 2) {
            throw std::runtime_error("invalid metadata size, got " + std::to_string(metadata_size));
        }

        std::vector<char> buf(metadata_size);
        fin.read(buf.data(), metadata_size);
        if (!fin) {
            throw std::runtime_error("failed to read metadata");
        }

        safetensors_json_parser parser(buf.data(), metadata_size, name_replace_map);

        ggml_init_params ggml_params = {
            /*.mem_size   =*/ (parser.tensors.size() + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx_data = ggml_init(ggml_params);
        gguf_context * ctx_gguf = gguf_init_empty();

        std::map<ggml_tensor *, uint64_t> offset_map;
        for (auto & tensor : parser.tensors) {
            ggml_tensor * t = ggml_new_tensor(ctx_data, tensor.type, 4, tensor.shape.data());
            ggml_set_name(t, tensor.name.c_str());
            gguf_add_tensor(ctx_gguf, t);
            tensors.insert({tensor.name, t});
            offset_map.insert({t, parser.get_data_offset() + tensor.offset});
        }

        // alloc memory and offload data
        if (!load_tensors_to_backend(fin, offset_map, ctx_gguf, ctx_data)) {
            throw std::runtime_error("failed to load tensors to backend");
        }
        log(GGML_LOG_LEVEL_INFO, "%s: Loaded %d tensors from %s\n", __func__, (int)gguf_get_n_tensors(ctx_gguf), fname);

        loaded_ggufs.push_back({
            gguf_context_ptr(ctx_gguf),
            ggml_context_ptr(ctx_data),
        });
    }

    /**
     * Various utility functions for building a cgraph.
     * 
     * This object will be provided to the user's builder function as the last argument.
     */
    struct build_utils {
        ggml_context * gf_ctx;
        ggml_cgraph  * gf;
        std::vector<printed_tensor> printed_tensors;
        build_utils(ggml_context * gf_ctx, ggml_cgraph * gf) : gf_ctx(gf_ctx), gf(gf) {}
        /**
         * Add an input tensor, this function does these steps:
         * 1. ggml_new_tensor_4d
         * 2. ggml_set_name
         * 3. ggml_set_input
         */
        ggml_tensor * new_input(const char * name, ggml_type dtype, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
            ggml_tensor * t = ggml_new_tensor_4d(gf_ctx, dtype, ne0, ne1, ne2, ne3);
            ggml_set_name(t, name);
            ggml_set_input(t);
            return t;
        }
        /**
         * Mark this tensor as output, this function does these steps:
         * 1. ggml_set_name
         * 2. ggml_set_output
         * 3. ggml_build_forward_expand
         */
        void mark_output(ggml_tensor * t, const char * name) {
            ggml_set_name(t, name);
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
        }
        /**
         * Print this tensor as soon as it is computed, useful for debugging.
         * name is optional, if not provided, the existing name of the tensor will be used
         */
        template <typename ...Params>
        void debug_print(ggml_tensor * t, Params&&... params) {
            std::string name = string_format(std::forward<Params>(params)...);
            if (t->flags) {
                // prevent renaming input/output tensor name by accident
                t = ggml_cpy(gf_ctx, t, ggml_dup_tensor(gf_ctx, t));
            }
            mark_output(t, name.c_str());
            printed_tensors.push_back({t, false});
        }
        /**
         * Same with `debug_print` but also print the full tensor shape and data.
         */
        template <typename ...Params>
        void debug_print_full(ggml_tensor * t, Params&&... params) {
            std::string name = string_format(std::forward<Params>(params)...);
            if (t->flags) {
                // prevent renaming input/output tensor name by accident
                t = ggml_cpy(gf_ctx, t, ggml_dup_tensor(gf_ctx, t));
            }
            mark_output(t, name.c_str());
            printed_tensors.push_back({t, true});
        }
    };

    /**
     * Build a cgraph using the given builder function.
     * 
     * The built cgraph will be stored in `ctx.gf`
     */
    void build_graph(std::function<void(ggml_context *, ggml_cgraph *, build_utils &)> builder_fn) {
        ggml_free(ctx_gf);
        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute_meta.size(),
            /*.mem_buffer =*/ buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };

        ctx_gf = ggml_init(params);
        ggml_backend_sched_reset(sched.get());
        gf = ggml_new_graph_custom(ctx_gf, max_nodes, false);

        build_utils utils(ctx_gf, gf);

        builder_fn(ctx_gf, gf, utils);
        ggml_backend_sched_alloc_graph(sched.get(), gf);
        dbg_printed_tensors = std::move(utils.printed_tensors);
    }

    /**
     * Same as `build_graph` but without `build_utils`
     */
    void build_graph(std::function<void(ggml_context *, ggml_cgraph *)> builder_fn) {
        build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, build_utils & utils) {
            builder_fn(ctx_gf, gf);
        });
    }

    /**
     * Compute the given cgraph
     */
    ggml_status compute() {
        ggml_status status = ggml_backend_sched_graph_compute(sched.get(), gf);
        if (status == GGML_STATUS_SUCCESS) {
            for (auto & p : dbg_printed_tensors) {
                std::vector<uint8_t> data(ggml_nbytes(p.t));
                ggml_backend_tensor_get(p.t, data.data(), 0, ggml_nbytes(p.t));
                ggml_easy::debug::print_tensor_shape(p.t);
                ggml_easy::debug::print_tensor_data(p.t, data.data(), p.full ? LONG_MAX : 3);
            }
        }
        return status;
    }

    /**
     * Set the data of a tensor by name
     */
    void set_tensor_data(const std::string & name, const void * data) {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error(string_format("tensor not found: %s", name.c_str()));
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
            throw std::runtime_error(string_format("tensor not found: %s", name.c_str()));
        }
        if (t->type != GGML_TYPE_F32) {
            throw std::runtime_error(string_format("tensor type must be GGML_TYPE_F32: %s", name.c_str()));
        }
        ggml_easy::debug::print_tensor_shape(t);
        std::vector<float> data(ggml_nelements(t));
        for (int d3 = 0; d3 < t->ne[3]; ++d3) {
            for (int d2 = 0; d2 < t->ne[2]; ++d2) {
                for (int d1 = 0; d1 < t->ne[1]; ++d1) {
                    for (int d0 = 0; d0 < t->ne[0]; ++d0) {
                        int i = d3 * t->ne[2] + d2 * t->ne[1] + d1 * t->ne[0] + d0;
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
            throw std::runtime_error(string_format("tensor not found: %s", name.c_str()));
        }
        std::vector<uint8_t> data(ggml_nbytes(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        return std::make_pair(t, data);
    }

    ~ctx() {
        ggml_backend_buffer_free(buf);
    }

private:
    bool load_tensors_to_backend(std::ifstream & fin, std::map<ggml_tensor *, uint64_t> & offset_map, gguf_context * ctx_gguf, ggml_context * ctx_data) {
        std::vector<uint8_t> read_buf;
        const bool use_custom_offset = !offset_map.empty();

        // alloc memory and offload data
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
            const size_t offset = use_custom_offset
                ? offset_map[cur]
                : gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
            log(GGML_LOG_LEVEL_DEBUG, "%s: Loading tensor \"%s\"\n", __func__, name);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                log(GGML_LOG_LEVEL_ERROR, "failed to seek for tensor: %s", name);
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
        return true;
    }

    void log(ggml_log_level level, const char * format, ...) {
        va_list args;
        va_start(args, format);
        log_impl(level, format, args);
        va_end(args);
    }

    void log_impl(ggml_log_level level, const char * format, va_list args) {
        va_list args_copy;
        va_copy(args_copy, args);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            log_cb(level, buffer, &log_level);
        } else {
            char * buffer2 = new char[len + 1];
            vsnprintf(buffer2, len + 1, format, args_copy);
            buffer2[len] = 0;
            log_cb(level, buffer2, &log_level);
            delete[] buffer2;
        }
        va_end(args_copy);
    }
}; // struct ctx

using gf_build_fn = std::function<void(ggml_context *, ggml_cgraph *, ctx::build_utils &)>;

////////////////////////////////////////

safetensors_json_parser::safetensors_json_parser(
        const char * json, size_t metadata_size, std::map<std::string, std::string> name_replace_map
) : metadata_size(metadata_size) {
    size_t i = 0;
    state s = STATE_ROOT;
    std::vector<char> buf;
    tensor cur_tensor;
    buf.reserve(128);
    auto i_pp = [&]() {
        if (++i > metadata_size) {
            throw std::runtime_error("unexpected end of JSON");
        }
        return i - 1;
    };
    auto pp_i = [&]() {
        if (++i > metadata_size) {
            throw std::runtime_error("unexpected end of JSON");
        }
        return i;
    };
    auto read_until = [&](char end) -> std::string {
        buf.clear(); i_pp();
        while (json[i] != end) buf.push_back(json[i_pp()]);
        return std::string(buf.data(), buf.size());
    };
    auto read_number = [&]() -> std::string {
        buf.clear(); i_pp();
        while ('0' <= json[i] && json[i] <= '9') buf.push_back(json[i_pp()]);
        return std::string(buf.data(), buf.size());
    };
    while (i < metadata_size) {
        char c = json[i];
        if (i == 0) GGML_ASSERT(c == '{' && "json must start with open curly bracket");

        // string
        if (c == '\"') {
            std::string key = read_until('\"');

            if (s == STATE_ROOT) {
                if (key == "__metadata__") {
                    s = STATE_OBJ_METADATA;
                    i_pp();
                    continue;
                } else {
                    cur_tensor.name = key;
                    for (auto & p : name_replace_map) {
                        string_replace_all(cur_tensor.name, p.first, p.second);
                    }
                    if (cur_tensor.name.empty()) {
                        throw std::runtime_error("empty tensor name");
                    }
                    if (cur_tensor.name.size() > GGML_MAX_NAME - 1) {
                        throw std::runtime_error("tensor name too long: '" + cur_tensor.name + "'; please use name_replace_map to rename it");
                    }
                    i_pp();
                    s = STATE_OBJ_TENSOR;
                    continue;
                }
            } else if (s == STATE_OBJ_TENSOR) {
                if (key == "dtype") {
                    GGML_ASSERT(json[pp_i()] == ':');
                    GGML_ASSERT(json[pp_i()] == '\"');
                    std::string value = read_until('\"');
                    /**/ if (value == "F32")  cur_tensor.type = GGML_TYPE_F32;
                    else if (value == "F16")  cur_tensor.type = GGML_TYPE_F16;
                    else if (value == "BF16") cur_tensor.type = GGML_TYPE_BF16;
                    else throw std::runtime_error("unsupported dtype: " + value);
                } else if (key == "shape") {
                    GGML_ASSERT(json[pp_i()] == ':');
                    GGML_ASSERT(json[pp_i()] == '[');
                    std::vector<int64_t> values;
                    for (int j = 0; j < 4; j++) {
                        std::string value = read_number();
                        if (value.empty()) break;
                        values.push_back(std::stoll(value));
                    }
                    GGML_ASSERT(values.size() > 0);
                    // flip column-major to row-major
                    for (size_t j = 0; j < values.size(); j++) {
                        cur_tensor.shape[j] = values[values.size() - j - 1];
                    }
                } else if (key == "data_offsets") {
                    GGML_ASSERT(json[pp_i()] == ':');
                    GGML_ASSERT(json[pp_i()] == '[');
                    std::string off_start = read_number();
                    GGML_ASSERT(!off_start.empty());
                    cur_tensor.offset = std::stoull(off_start);
                    std::string off_end = read_number();
                    GGML_ASSERT(!off_end.empty()); // unused
                }
            }
        }

        // object
        else if (c == '{') {
            if (s == STATE_OBJ_METADATA) {
                // skip metadata object
                while (json[pp_i()] != '}') {}
                s = STATE_ROOT;
            } else if (s == STATE_OBJ_TENSOR) {
                // read next string
            }
        } else if (c == '}') {
            if (s == STATE_OBJ_TENSOR) {
                // cur_tensor.print(); // debug
                tensors.push_back(cur_tensor);
                cur_tensor = {};
                s = STATE_ROOT;
            }
        }

        // ignore ',' and ':'
        i++;
    }
}

uint64_t safetensors_json_parser::get_data_offset() {
    // alignment: https://github.com/huggingface/safetensors/blob/7d5af853631628137a79341ddc5611d18a17f3fe/safetensors/src/tensor.rs#L202
    static const int alignment = 8; // bytes
    return GGML_PAD(8 + metadata_size, alignment);
}

////////////////////////////////////////

namespace debug {
    static void print_backend_buffer_info(ctx & gctx) {
        if (gctx.backend && gctx.buf) {
            auto buft_weight = ggml_backend_get_default_buffer_type(gctx.backend);
            size_t size_weight = ggml_backend_buffer_get_size(gctx.buf);
            if (size_weight > 1) {
                printf("%s: %10s weight buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft_weight),
                        size_weight / 1024.0 / 1024.0);
            }
        }
        for (size_t i = 0; i < gctx.backend_ptrs.size(); ++i) {
            ggml_backend_t backend = gctx.backend_ptrs[i];
            ggml_backend_buffer_type_t buft = gctx.backend_buft[i];
            size_t size_sched = ggml_backend_sched_get_buffer_size(gctx.sched.get(), backend);
            if (size_sched > 1) {
                printf("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size_sched / 1024.0 / 1024.0);
            }
        }
    }

    static void print_tensor_shape(ggml_tensor * t) {
        printf("%s.shape = [", t->name);
        for (int i = 0; i < ggml_n_dims(t); ++i) {
            printf("%" PRId64, t->ne[i]);
            if (i < ggml_n_dims(t) - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }

    static void print_tensor_data(ggml_tensor * t, uint8_t * data, int64_t n) {
        ggml_type type = t->type;
        int64_t * ne = t->ne;
        size_t * nb = t->nb;
        for (int64_t i3 = 0; i3 < ne[3]; i3++) {
            printf("%s.data: [\n", t->name);
            for (int64_t i2 = 0; i2 < ne[2]; i2++) {
                if (i2 == n && ne[2] > 2*n) {
                    printf("     ..., \n");
                    i2 = ne[2] - n;
                }
                printf("     [\n");
                for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                    if (i1 == n && ne[1] > 2*n) {
                        printf("      ..., \n");
                        i1 = ne[1] - n;
                    }
                    printf("      [");
                    for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                        if (i0 == n && ne[0] > 2*n) {
                            printf("..., ");
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
                        printf("%12.4f", v);
                        if (i0 < ne[0] - 1) printf(", ");
                    }
                    printf("],\n");
                }
                printf("     ],\n");
            }
            printf("    ]\n");
            //printf("    sum = %f\n", sum);
        }
    }
} // namespace debug

////////////////////////////////////////

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

} // namespace ggml_easy
