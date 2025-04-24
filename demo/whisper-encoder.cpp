#include "ggml.h"
#include "ggml-easy.h"
#include <iostream>
#include <thread>
#include <cmath>

#define WHISPER_ASSERT GGML_ASSERT

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

namespace whisper_preprocessor {

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

#define SIN_COS_N_COUNT WHISPER_N_FFT
namespace {
struct whisper_global_cache {
    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    float hann_window[WHISPER_N_FFT];

    whisper_global_cache() {
        fill_sin_cos_table();
        fill_hann_window(sizeof(hann_window)/sizeof(hann_window[0]), true, hann_window);
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic, float * output) {
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }
} global_cache;
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n]*global_cache.cos_vals[idx]; // cos(t)
            im -= in[n]*global_cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N*2 == 1) {
        dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i]= in[2*i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2*i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = global_cache.cos_vals[idx]; // cos(t)
        float im = -global_cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

static void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters & filters, whisper_mel & mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft = filters.n_fft;
    int i = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    WHISPER_ASSERT(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
        const float * samples,
        const int   n_samples,
        const int   /*sample_rate*/,
        const int   frame_size,
        const int   frame_step,
        const int   n_mel,
        const int   n_threads,
        const whisper_filters & filters,
        const bool   debug,
        whisper_mel & mel) {
    const int64_t t_start_us = ggml_time_us();

    // Hann window
    WHISPER_ASSERT(frame_size == WHISPER_N_FFT && "Unsupported frame_size");
    const float * hann = global_cache.hann_window;

    // Calculate the length of padding
    int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel     = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len     = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                    log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded),
                    n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                    std::cref(filters), std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }

    // Dump log_mel_spectrogram
    if (debug) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

} // namespace whisper_preprocessor

struct whisper_encoder {
    float norm_eps = 1e-5;
    int n_head = 6;
    int n_embd;
    int n_ctx = 1500;

    ggml_tensor * pos_embd;

    ggml_tensor * conv_1_w;
    ggml_tensor * conv_1_b;
    ggml_tensor * conv_2_w;
    ggml_tensor * conv_2_b;

    ggml_tensor * out_norm_w;
    ggml_tensor * out_norm_b;

    struct layer {
        ggml_tensor * inp_norm_w;
        ggml_tensor * inp_norm_b;

        ggml_tensor * attn_q;
        ggml_tensor * attn_q_b;
        ggml_tensor * attn_k;
        ggml_tensor * attn_v;
        ggml_tensor * attn_v_b;
        ggml_tensor * attn_o;
        ggml_tensor * attn_o_b;
        ggml_tensor * attn_post_norm_w;
        ggml_tensor * attn_post_norm_b;

        ggml_tensor * ffn_up;
        ggml_tensor * ffn_up_b;
        ggml_tensor * ffn_down;
        ggml_tensor * ffn_down_b;
    };
    std::vector<layer> layers;

    whisper_encoder(ggml_easy::ctx & ctx, int n_layers) {
        const char * prefix = "encoder";
        pos_embd = ctx.get_weight("model.%s.embed_positions.weight", prefix);
        n_embd = pos_embd->ne[0];
        conv_1_b = ctx.get_weight("model.%s.conv1.bias",   prefix);
        conv_1_w = ctx.get_weight("model.%s.conv1.weight", prefix);
        conv_2_b = ctx.get_weight("model.%s.conv2.bias",   prefix);
        conv_2_w = ctx.get_weight("model.%s.conv2.weight", prefix);
        out_norm_b = ctx.get_weight("model.%s.layer_norm.bias",   prefix);
        out_norm_w = ctx.get_weight("model.%s.layer_norm.weight", prefix);
        for (int il = 0; il < n_layers; il++) {
            layers.push_back({
                .inp_norm_w = ctx.get_weight("model.%s.layers.%d.self_attn_layer_norm.weight", prefix, il),
                .inp_norm_b = ctx.get_weight("model.%s.layers.%d.self_attn_layer_norm.bias",   prefix, il),

                .attn_q           = ctx.get_weight("model.%s.layers.%d.self_attn.q_proj.weight",   prefix, il),
                .attn_q_b         = ctx.get_weight("model.%s.layers.%d.self_attn.q_proj.bias",     prefix, il),
                .attn_k           = ctx.get_weight("model.%s.layers.%d.self_attn.k_proj.weight",   prefix, il),
                .attn_v           = ctx.get_weight("model.%s.layers.%d.self_attn.v_proj.weight",   prefix, il),
                .attn_v_b         = ctx.get_weight("model.%s.layers.%d.self_attn.v_proj.bias",     prefix, il),
                .attn_o           = ctx.get_weight("model.%s.layers.%d.self_attn.out_proj.weight", prefix, il),
                .attn_o_b         = ctx.get_weight("model.%s.layers.%d.self_attn.out_proj.bias",   prefix, il),
                .attn_post_norm_w = ctx.get_weight("model.%s.layers.%d.final_layer_norm.weight",   prefix, il),
                .attn_post_norm_b = ctx.get_weight("model.%s.layers.%d.final_layer_norm.bias",     prefix, il),

                .ffn_up     = ctx.get_weight("model.%s.layers.%d.fc1.weight",              prefix, il),
                .ffn_up_b   = ctx.get_weight("model.%s.layers.%d.fc1.bias",                prefix, il),
                .ffn_down   = ctx.get_weight("model.%s.layers.%d.fc2.weight",              prefix, il),
                .ffn_down_b = ctx.get_weight("model.%s.layers.%d.fc2.bias",                prefix, il),
            });
        }
    }

    ggml_tensor * forward(ggml_context * ctx0, ggml_easy::ctx::build_utils & utils, ggml_tensor * input, ggml_tensor * input_pos) {
        int n_tokens    = n_ctx; //;input->ne[1];
        ggml_tensor * x = input;

        auto layer_norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            x = ggml_norm(ctx0, x, norm_eps);
            x = ggml_mul(ctx0, x, w);
            x = ggml_add(ctx0, x, b);
            return x;
        };

        auto add_pos = [&](ggml_tensor * x) {
            //ggml_tensor * pos_embd_selected = ggml_get_rows(ctx0, pos_embd, input_pos);
            //x = ggml_add(ctx0, x, pos_embd_selected);
            x = ggml_add(ctx0, x, pos_embd);
            return x;
        };

        // TODO: do this at conversion time, see LlamaModel.permute in convert_hf_to_gguf.py
        auto llama_permute = [&](ggml_tensor * w) {
            ggml_tensor * tmp = ggml_reshape_4d(ctx0, w, w->ne[0], w->ne[1] / n_head / 2, 2, n_head);
            tmp = ggml_permute(ctx0, tmp, 0, 2, 1, 3);
            tmp = ggml_cont(ctx0, tmp);
            return ggml_reshape_2d(ctx0, tmp, w->ne[0], w->ne[1]);
        };

        // convolution + gelu
        {
            ggml_tensor * tmp;
            tmp = ggml_cast(ctx0, conv_1_w, GGML_TYPE_F16); // TODO: do this at conversion time
            x = ggml_conv_1d_ph(ctx0, tmp, input, 1, 1);
            tmp = ggml_cont(ctx0, ggml_transpose(ctx0, conv_1_b)); // TODO: do this at conversion time
            x = ggml_add(ctx0, x, tmp);

            x = ggml_gelu(ctx0, x);

            tmp = ggml_cast(ctx0, conv_2_w, GGML_TYPE_F16); // TODO: do this at conversion time
            x = ggml_conv_1d_ph(ctx0, tmp, x, 2, 1);
            tmp = ggml_cont(ctx0, ggml_transpose(ctx0, conv_2_b)); // TODO: do this at conversion time
            x = ggml_add(ctx0, x, tmp);

            x = ggml_gelu(ctx0, x);
        }

        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        x = add_pos(x);
        ggml_tensor * residual = x;

        int i = 0; // for debugging
        for (auto & layer : layers) {
            residual = x;

            // input layer norm
            x = layer_norm(x, layer.inp_norm_w, layer.inp_norm_b);

            // self attention
            {
                ggml_tensor * q = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_q, x), layer.attn_q_b);
                ggml_tensor * k = ggml_mul_mat(ctx0, layer.attn_k, x); // no bias for key
                ggml_tensor * v = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_v, x), layer.attn_v_b);

                int n_embd_head = n_embd / n_head;
                q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
                k = ggml_reshape_3d(ctx0, k, n_embd_head, n_head, n_tokens);
                v = ggml_reshape_3d(ctx0, v, n_embd_head, n_head, n_tokens);

                int n_rot = n_embd_head;
                q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
                q = ggml_scale(ctx0, q, 1.0f / std::sqrt(n_embd_head));
                // utils.debug_print(q, "q rope");

                k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
                // utils.debug_print(k, "k rope");

                ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
                kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f, 0.0f);
                // utils.debug_print(kq, "kq softmax");

                v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

                ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
                //kqv = ggml_reshape_3d(ctx0, kqv, n_embd_head, n_tokens, n_head);
                kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                kqv = ggml_cont_2d(ctx0, kqv, n_embd, n_tokens);
                // utils.debug_print(kqv, "kqv");
                // utils.debug_print(ggml_sum(ctx0, kqv), "kqv_sum");

                x = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_o, kqv), layer.attn_o_b);
            }

            // residual
            x = ggml_add(ctx0, x, residual);

            residual = x;
            x = layer_norm(x, layer.attn_post_norm_w, layer.attn_post_norm_b);

            // mlp
            {
                x = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_up, x), layer.ffn_up_b);
                x = ggml_gelu(ctx0, x);
                x = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_down, x), layer.ffn_down_b);
            }

            // residual
            x = ggml_add(ctx0, x, residual);
            // utils.debug_print(x, "output_layer_%d", i);
            // utils.debug_print(ggml_sum(ctx0, x), "output_layer_%d_sum", i); i++;
        }

        // output norm
        x = layer_norm(x, out_norm_w, out_norm_b);

        return x;
    }
};

int main() {
    ggml_easy::ctx_params params;
    ggml_easy::ctx ctx(params);
    ctx.load_gguf("models/whisper-mel-filters.gguf");
    ctx.load_safetensors("whisper-tiny.safetensors", {});

    whisper_preprocessor::whisper_filters mel_filters;
    {
        auto mel_80 = ctx.get_weight("mel_80");
        ggml_easy::debug::print_tensor_shape(mel_80);
        mel_filters.n_mel = mel_80->ne[1];
        mel_filters.n_fft = mel_80->ne[0];
        mel_filters.data.resize(ggml_nelements(mel_80));
        ggml_backend_tensor_get(mel_80, mel_filters.data.data(), 0, mel_filters.data.size());

        // for (int row = 0; row < mel_filters.n_mel; row++) {
        //     for (int i = 0; i < mel_filters.n_fft; i++) {
        //         float elem = mel_filters.data[row * mel_filters.n_fft + i];
        //         if (elem != 0.0) {
        //             printf("[%d, %d] %f\n", row, i, elem);
        //         }
        //     }
        //     printf("\n");
        // }
    }

    std::vector<float> samples(3000, 1.0);

    whisper_preprocessor::whisper_mel mel;
    whisper_preprocessor::log_mel_spectrogram(
            samples.data(),
            samples.size(),
            WHISPER_SAMPLE_RATE,
            WHISPER_N_FFT,
            WHISPER_HOP_LENGTH,
            mel_filters.n_mel,
            4, // threads
            mel_filters,
            false,
            mel);

    printf("mel.n_len: %d\n", mel.n_len);
    printf("mel.n_mel: %d\n", mel.n_mel);
    printf("mel.size:  %zu\n", mel.data.size());
    // print first and last 10 elements
    for (int i = 0; i < 10; i++) {
        printf("%f ", mel.data[i]);
    }
    printf("\n");
    for (int i = mel.data.size() - 10; i < mel.data.size(); i++) {
        printf("%f ", mel.data[i]);
    }
    printf("\n");


    // model
    whisper_encoder encoder(ctx, 4);

    // create cgraph
    ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf, auto & utils) {
        ggml_tensor * input = utils.new_input("mel", GGML_TYPE_F32, 2*encoder.n_ctx, mel.n_mel);
        ggml_easy::debug::print_tensor_shape(input);
        utils.debug_print(input, "input");
        ggml_tensor * pos   = nullptr; //utils.new_input("pos", GGML_TYPE_I32, mel.n_len);
        ggml_tensor * result = encoder.forward(ctx_gf, utils, input, pos);
        utils.debug_print(result, "result");
        utils.mark_output(result, "result");
    });

    // set data
    ctx.set_tensor_data("mel", mel.data.data());
    // set the input
    {
        int mel_offset = 0;
        int n_ctx = encoder.n_ctx;
        std::vector<float> dst(2*n_ctx * mel.n_mel, 0.0f);

        const int i0 = std::min(mel_offset,           mel.n_len);
        const int i1 = std::min(mel_offset + 2*n_ctx, mel.n_len);

        for (int j = 0; j < mel.n_mel; ++j) {
            for (int i = i0; i < i1; ++i) {
                dst[j*2*n_ctx + (i - i0)] = mel.data[j*mel.n_len + i];
            }
        }

        ctx.set_tensor_data("mel", dst.data());
    }

    // set pos
    // std::vector<int> pos(mel.n_len);
    // for (size_t i = 0; i < pos.size(); i++) {
    //     pos[i] = i;
    // }
    // ctx.set_tensor_data("pos", pos.data());

    // compute
    ctx.compute();

    return 0;
}
