#pragma once

#include "ggml.h"

#include <cstdarg>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

// mobilenet v5 implementation

namespace mobilenet {

using get_tensor_fn = std::function<ggml_tensor * (const std::string & name)>;
using callback_fn   = std::function<void(ggml_tensor * cur, const char * name, int il)>;

static std::string str_fmt(const char * fmt, ...) {
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
    return std::string(buf.data(), buf.size());
}

static std::string str_concat(const std::string & a, const std::string & b) {
    return str_fmt("%s%s", a.c_str(), b.c_str()); // the "+" operator does not work, why?
}

struct v5_blk {
    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur, callback_fn & cb) = 0;
    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) = 0;
    virtual ~v5_blk() = default;
};

enum conv_type {
    CONV_TYPE_NORMAL,    // ggml_conv_2d
    CONV_TYPE_POINTWISE, // ggml_mul_mat
    CONV_TYPE_DEPTHWISE, // ggml_conv_2d_dw
};

static ggml_tensor * rms_norm_act_2d(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * scale,
        int n_groups,
        bool apply_act,
        callback_fn & cb) {
    cur = ggml_group_norm(ctx, cur, n_groups, 1e-6f);
    cb(cur, "rms_norm_act.norm", -1);
    if (scale != nullptr) {
        cur = ggml_mul(ctx, cur, ggml_reshape_3d(ctx, scale, 1, 1, scale->ne[0]));
        cb(cur, "rms_norm_act.norm_scaled", -1);
    }
    if (apply_act) {
        cur = ggml_gelu(ctx, cur);
        cb(cur, "rms_norm_act.gelu", -1);
    }
    return cur;
}

// ConvNormAct
struct v5_cna : v5_blk {
    conv_type type = CONV_TYPE_NORMAL;
    int kernel_size = 0;
    int stride = 1;
    int dilation = 1;
    int padding = 0;
    bool apply_act = false;
    float expand_ratio = 1.0f;

    int in_chs  = 0;
    int out_chs = 0; // aka filters

    ggml_tensor * norm = nullptr;
    ggml_tensor * conv = nullptr;

    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) {
        norm = get_tensor(str_concat(prefix, ".bn.weight"));
        conv = get_tensor(str_concat(prefix, ".conv.weight"));

        if (type == CONV_TYPE_POINTWISE) {
            GGML_ASSERT(kernel_size == 1);
            GGML_ASSERT(stride      == 1);
            GGML_ASSERT(padding     == 0);
            GGML_ASSERT(dilation    == 1);
            GGML_ASSERT(conv->ne[0] == 1 && conv->ne[1] == 1);
        } else {
            GGML_ASSERT(conv->ne[0] == kernel_size && conv->ne[1] == kernel_size);
            GGML_ASSERT(conv->ne[3] == norm->ne[0]); // norm size matches
        }

        in_chs  = conv->ne[2];
        out_chs = conv->ne[3];
    }

    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur, callback_fn & cb) {
        if (type == CONV_TYPE_POINTWISE) {
            cur = ggml_conv_2d(ctx, conv, cur, 1, 1, 0, 0, 1, 1);
            cb(cur, "conv_norm_act.pw", -1);
        } else if (type == CONV_TYPE_DEPTHWISE) {
            cur = ggml_conv_2d_dw(ctx, conv, cur,
                stride, stride,
                padding, padding,
                dilation, dilation);
            cb(cur, "conv_norm_act.dw", -1);
        } else {
            cur = ggml_conv_2d(ctx, conv, cur,
                stride, stride,
                padding, padding,
                dilation, dilation);
            cb(cur, "conv_norm_act", -1);
        }

        cur = rms_norm_act_2d(ctx, cur, norm, out_chs, apply_act, cb);

        return cur;
    }
};

// EdgeResidual
struct v5_er : v5_blk {
    int kernel_size = 0;
    int stride = 1;
    int filters = 0;

    ggml_tensor * norm1    = nullptr;
    ggml_tensor * norm2    = nullptr;
    ggml_tensor * conv_exp = nullptr;
    ggml_tensor * conv_pwl = nullptr;

    v5_er(int kernel_size, int filters, int stride) :
        kernel_size(kernel_size), stride(stride), filters(filters) {}

    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) {
        norm1    = get_tensor(str_concat(prefix, ".bn1.weight"));
        norm2    = get_tensor(str_concat(prefix, ".bn2.weight"));
        conv_exp = get_tensor(str_concat(prefix, ".conv_exp.weight"));
        conv_pwl = get_tensor(str_concat(prefix, ".conv_pwl.weight"));

        GGML_ASSERT(ggml_n_dims(conv_exp) == 4); // expected 4D tensor
        GGML_ASSERT(ggml_n_dims(conv_pwl) == 4); // expected 4D tensor
        GGML_ASSERT(conv_exp->ne[0] == kernel_size && conv_exp->ne[1] == kernel_size);
        GGML_ASSERT(conv_pwl->ne[0] == 1           && conv_pwl->ne[1] == 1);
    }

    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur, callback_fn & cb) {
        int padding = (kernel_size - 1) / 2;
        cur = ggml_conv_2d(ctx, conv_exp, cur,
            stride, stride,
            padding, padding,
            1, 1);
        cb(cur, "edge_residual.conv_exp", -1);

        int mid_chs = conv_exp->ne[3];
        cur = rms_norm_act_2d(ctx, cur, norm1, mid_chs, true, cb);
        cb(cur, "edge_residual.norm1", -1);

        cur = ggml_conv_2d(ctx, conv_pwl, cur, 1, 1, 0, 0, 1, 1);
        cb(cur, "edge_residual.conv_pwl", -1);

        int out_chs = conv_pwl->ne[1];
        cur = rms_norm_act_2d(ctx, cur, norm2, out_chs, false, cb);
        cb(cur, "edge_residual.norm2", -1);

        return cur;
    }
};

// UniversalInvertedResidual
struct v5_uir : v5_blk {
    int dw_kernel_size_start = 0;
    int dw_kernel_size_mid = 0;
    bool multiscale = false;

    v5_cna dw_start;
    v5_cna pw_exp;
    v5_cna dw_mid;
    v5_cna pw_proj;

    ggml_tensor * layer_scale = nullptr;

    v5_uir(int dw_kernel_size_start, int dw_kernel_size_mid, int filters, int stride = 1, float expand_ratio = 4.0f, bool multiscale = false) :
            dw_kernel_size_start(dw_kernel_size_start),
            dw_kernel_size_mid(dw_kernel_size_mid),
            multiscale(multiscale) {
        GGML_UNUSED(filters);
        GGML_UNUSED(expand_ratio);

        dw_start.type = CONV_TYPE_DEPTHWISE;
        dw_start.kernel_size = dw_kernel_size_start;
        dw_start.stride = !dw_kernel_size_mid ? stride : 1;
        dw_start.padding = (dw_kernel_size_start - 1) / 2;

        pw_exp.type = CONV_TYPE_POINTWISE;
        pw_exp.kernel_size = 1;

        dw_mid.type = CONV_TYPE_DEPTHWISE;
        dw_mid.kernel_size = dw_kernel_size_mid;
        dw_mid.stride = 1;
        dw_mid.padding = (dw_kernel_size_mid - 1) / 2;

        pw_proj.type = CONV_TYPE_POINTWISE;
        pw_proj.kernel_size = 1;
    }

    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) {
        if (dw_kernel_size_start) {
            dw_start.load_tensors(str_concat(prefix, ".dw_start"), get_tensor);
        }
        pw_exp.load_tensors(str_concat(prefix, ".pw_exp"), get_tensor);
        if (dw_kernel_size_mid) {
            dw_mid.load_tensors(str_concat(prefix, ".dw_mid"), get_tensor);
        }
        pw_proj.load_tensors(str_concat(prefix, ".pw_proj"), get_tensor);
        layer_scale = get_tensor(str_concat(prefix, ".layer_scale.weight"));
    }

    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur, callback_fn & cb) {
        if (dw_kernel_size_start) {
            cur = dw_start.build(ctx, cur, cb);
        }
        cur = pw_exp.build(ctx, cur, cb);
        if (dw_kernel_size_mid) {
            cur = dw_mid.build(ctx, cur, cb);
        }
        cur = pw_proj.build(ctx, cur, cb);
        cur = ggml_mul(ctx, cur, ggml_reshape_3d(ctx, layer_scale, 1, 1, layer_scale->ne[0]));
        return cur;
    }
};

// MultiQueryAttentionBlock
struct v5_mmqa : v5_blk {
    int num_heads = 0;
    int kv_strides = 0;
    int kv_dim = 0;
    bool mmqa_avg_pool_kv = false;
    bool multiscale = false;

    ggml_tensor * k_down_conv = nullptr;
    ggml_tensor * k_norm      = nullptr;
    ggml_tensor * k_proj      = nullptr;
    ggml_tensor * q_proj      = nullptr;
    ggml_tensor * v_down_conv = nullptr;
    ggml_tensor * v_norm      = nullptr;
    ggml_tensor * v_proj      = nullptr;
    ggml_tensor * o_proj      = nullptr;
    ggml_tensor * layer_scale = nullptr;
    ggml_tensor * norm        = nullptr;

    v5_mmqa(int num_heads, int kv_dim, int kv_strides,
                       bool mmqa_avg_pool_kv = false, bool multiscale = false) :
        num_heads(num_heads), kv_strides(kv_strides), kv_dim(kv_dim),
        mmqa_avg_pool_kv(mmqa_avg_pool_kv), multiscale(multiscale) {}

    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) {
        if (kv_strides > 1) {
            k_down_conv = get_tensor(str_concat(prefix, ".attn.key.down_conv.weight"));
            v_down_conv = get_tensor(str_concat(prefix, ".attn.value.down_conv.weight"));
            k_norm      = get_tensor(str_concat(prefix, ".attn.key.norm.weight"));
            v_norm      = get_tensor(str_concat(prefix, ".attn.value.norm.weight"));
        }
        k_proj      = get_tensor(str_concat(prefix, ".attn.key.proj.weight"));
        q_proj      = get_tensor(str_concat(prefix, ".attn.query.proj.weight"));
        v_proj      = get_tensor(str_concat(prefix, ".attn.value.proj.weight"));
        o_proj      = get_tensor(str_concat(prefix, ".attn.output.proj.weight"));
        layer_scale = get_tensor(str_concat(prefix, ".layer_scale.weight"));
        norm        = get_tensor(str_concat(prefix, ".norm.weight"));
    }

    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur, callback_fn & cb) {
        ggml_tensor * k = nullptr;
        ggml_tensor * q = nullptr;
        ggml_tensor * v = nullptr;

        if (kv_strides > 1) {
            k = ggml_conv_2d_dw(ctx, k_down_conv, cur, kv_strides, kv_strides, 0, 0, 1, 1);
            cb(k, "mmqa.k_down_conv", -1);
            k = rms_norm_act_2d(ctx, k, k_norm, kv_dim, false, cb);
            k = ggml_conv_2d(ctx, k_proj, k, 1, 1, 0, 0, 1, 1);
            cb(k, "mmqa.k_proj", -1);
        } else {
            k = ggml_conv_2d(ctx, k_proj, cur, 1, 1, 0, 0, 1, 1);
            cb(k, "mmqa.k_proj", -1);
        }

        if (kv_strides > 1) {
            v = ggml_conv_2d_dw(ctx, v_down_conv, cur, kv_strides, kv_strides, 0, 0, 1, 1);
            cb(v, "mmqa.v_down_conv", -1);
            v = rms_norm_act_2d(ctx, v, v_norm, kv_dim, false, cb);
            v = ggml_conv_2d(ctx, v_proj, v, 1, 1, 0, 0, 1, 1);
            cb(v, "mmqa.v_proj", -1);
        } else {
            v = ggml_conv_2d(ctx, v_proj, cur, 1, 1, 0, 0, 1, 1);
            cb(v, "mmqa.v_proj", -1);
        }

        q = ggml_conv_2d(ctx, q_proj, cur, 1, 1, 0, 0, 1, 1);
        cb(q, "mmqa.q_proj", -1);

        // reshape k, v, q

        q = ggml_reshape_3d(ctx, q, kv_dim, num_heads, q->ne[0] * q->ne[1]);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        cb(q, "mmqa.q_reshape", -1);

        k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3));
        k = ggml_reshape_2d(ctx, k, k->ne[0], k->ne[1] * k->ne[2]);
        cb(k, "mmqa.k_reshape", -1);

        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        v = ggml_reshape_2d(ctx, v, v->ne[0], v->ne[1] * v->ne[2]);
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        cb(v, "mmqa.v_reshape", -1);

        float kq_scale = 1.0f / std::sqrt(static_cast<float>(kv_dim));
        build_attn(ctx, o_proj, q, k, v, nullptr, kq_scale, cb);
        cb(cur, "mmqa.attn_output", -1);

        return cur;
    }

    ggml_tensor * build_attn(
            ggml_context * ctx0,
            ggml_tensor * wo,
            ggml_tensor * q,
            ggml_tensor * k,
            ggml_tensor * v,
            ggml_tensor * kq_mask,
            float kq_scale,
            callback_fn & cb) const {
        ggml_tensor * cur;

        {
            const auto n_tokens = q->ne[1];
            const auto n_head   = q->ne[2];

            ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

            ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
            cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*n_head, n_tokens);
        }

        cb(cur, "kqv_out", -1);

        {
            int h = std::sqrt(cur->ne[1]);
            int w = h;
            int c = cur->ne[0];
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
            cur = ggml_reshape_3d(ctx0, cur, w, h, c);
            cb(cur, "kqv_out_reshape", -1);
        }

        // output projection
        cur = ggml_conv_2d(ctx0, wo, cur, 1, 1, 0, 0, 1, 1);

        return cur;
    }
};

// MobileNetV5MultiScaleFusionAdapter
struct v5_msfa : v5_blk {
    v5_cna pw_exp;
    v5_cna pw_proj;

    ggml_tensor * norm;

    v5_msfa() {
        pw_exp.type = CONV_TYPE_POINTWISE;
        pw_exp.kernel_size = 1;

        pw_proj.type = CONV_TYPE_POINTWISE;
        pw_proj.kernel_size = 1;
    }

    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) {
        pw_exp .load_tensors(str_concat(prefix, ".ffn.pw_exp"),  get_tensor);
        pw_proj.load_tensors(str_concat(prefix, ".ffn.pw_proj"), get_tensor);
        norm    = get_tensor(str_concat(prefix, ".norm.weight"));
    }

    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur, callback_fn & cb) {
        int target_res = pw_exp.conv->ne[2];

        cur = ggml_upscale_ext(ctx, cur,
            cur->ne[0], cur->ne[1], target_res, 1, GGML_SCALE_MODE_NEAREST);
        cb(cur, "msfa.ffn.pw_exp.upscale", -1);

        cur = pw_exp.build(ctx, cur, cb);
        cb(cur, "msfa.ffn.pw_exp.output", -1);
        cur = pw_proj.build(ctx, cur, cb);
        cb(cur, "msfa.ffn.pw_proj.output", -1);

        cur = ggml_mul(ctx, cur, ggml_reshape_3d(ctx, norm, 1, 1, norm->ne[0]));
        cb(cur, "msfa.norm", -1);

        return cur;
    }
};

struct v5_model {
    v5_cna  conv_stem; // input
    v5_msfa msfa;      // output

    // mapping block to prefix
    std::vector<std::pair<v5_blk *, std::string>> blocks;

    // temporary variables
    int stg_idx = 0;
    int blk_idx = 0;

    v5_model() {
        conv_stem.type = CONV_TYPE_NORMAL;
        conv_stem.kernel_size = 3;
        conv_stem.stride = 2;
        conv_stem.padding = 1;
    }

    ~v5_model() {
        for (auto & blk : blocks) {
            delete blk.first;
        }
    }

    void load(get_tensor_fn & get_tensor) {
        // Convolution Stem
        conv_stem.load_tensors("v.mobilenet.conv_stem", get_tensor);

        // Stage 1: Edge Residuals
        stg_idx = 0; blk_idx = 0;
        add(    new v5_er(3, 128, 2));
        add(2,  new v5_er(3, 128, 1));

        // Stage 2: Universal Inverted Residuals
        stg_idx = 1; blk_idx = 0;
        add(    new v5_uir(3, 5, 256, 2, 6.0f));
        add(    new v5_uir(5, 0, 256));
        add(    new v5_uir(3, 0, 256));
        add(    new v5_uir(5, 0, 256));
        add(    new v5_uir(3, 0, 256));

        // Stage 3: Universal Inverted Residuals with Multi-Query Attention
        stg_idx = 2; blk_idx = 0;
        add(    new v5_uir(5, 5, 640, 2, 6.0f));
        add(7,  new v5_uir(5, 0, 640));
        add(    new v5_uir(0, 0, 640, 1, 1.0f));
        add(13, new v5_mmqa(12, 64, 2), new v5_uir(0, 0, 640, 1, 2.0f));
        add(    new v5_mmqa(12, 64, 2), new v5_uir(0, 0, 640, 1, 2.0f, true));

        // Stage 4: Universal Inverted Residuals with Multi-Query Attention
        stg_idx = 3; blk_idx = 0;
        add(    new v5_uir(5, 5, 1280, 2, 6.0f));
        add(18, new v5_mmqa(16, 96, 1), new v5_uir(0, 0, 1280, 1, 2.0f));
        add(    new v5_mmqa(16, 96, 1), new v5_uir(0, 0, 1280, 1, 2.0f, true));

        for (auto blk : blocks) {
            blk.first->load_tensors(blk.second, get_tensor);
        }

        // Output
        msfa.load_tensors("v.mobilenet.msfa", get_tensor);
    }

    ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur, callback_fn & cb) {
        cur = conv_stem.build(ctx, cur, cb);
        cb(cur, "conv_stem.output", -1);

        for (auto & blk : blocks) {
            cur = blk.first->build(ctx, cur, cb);
            cb(cur, str_concat(blk.second, ".output").c_str(), -1);
        }

        cur = msfa.build(ctx, cur, cb);
        cb(cur, "msfa.output", -1);

        return cur;
    }

    void add(v5_blk * blk) {
        blocks.emplace_back(blk, str_fmt("v.mobilenet.blocks.%d.%d", stg_idx, blk_idx++));
    }

    void add(v5_blk * blk0, v5_blk * blk1) {
        add(blk0);
        add(blk1);
    }

    void add(int count, v5_blk * blk) {
        for (int i = 0; i < count; ++i) {
            add(blk);
        }
    }

    void add(int count, v5_blk * blk0, v5_blk * blk1) {
        for (int i = 0; i < count; ++i) {
            add(blk0, blk1);
        }
    }
};

} // namespace mobilenet
