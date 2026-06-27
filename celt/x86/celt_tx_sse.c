/* Copyright (c) 2026 Xiph.Org Foundation */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mdct.h"
#include "_kiss_fft_guts.h"
#include "stack_alloc.h"
#include "x86/mdct_x86.h"

#if defined(OPUS_X86_TX_MDCT)

#include <stddef.h>
#include <xmmintrin.h>
#include "celt_tx.h"

typedef __m128 v4sf;
typedef kiss_fft_cpx cpx;

#define V_SWAP_RI(x) _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1))

#define LOAD_V2(var, ptr0, ptr1) \
    var = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(ptr0)); \
    var = _mm_loadh_pi(var, (const __m64*)(ptr1))

static const unsigned int lh_mask_int[4] = {0xFFFFFFFF, 0xFFFFFFFF, 0, 0};

#define FFT4_CORE(x01, x23, out02, out13) \
    do { \
        v4sf s01 = _mm_add_ps(x01, x23); \
        v4sf s13 = _mm_sub_ps(x01, x23); \
        v4sf s00 = _mm_shuffle_ps(s01, s01, _MM_SHUFFLE(1, 0, 1, 0)); \
        v4sf s22 = _mm_shuffle_ps(s01, s01, _MM_SHUFFLE(3, 2, 3, 2)); \
        v4sf s22_signed = _mm_mul_ps(s22, _mm_setr_ps(1.0f, 1.0f, -1.0f, -1.0f)); \
        out02 = _mm_add_ps(s00, s22_signed); \
        v4sf s11 = _mm_shuffle_ps(s13, s13, _MM_SHUFFLE(1, 0, 1, 0)); \
        v4sf s33 = _mm_shuffle_ps(s13, s13, _MM_SHUFFLE(3, 2, 3, 2)); \
        v4sf s33_swapped = V_SWAP_RI(s33); \
        v4sf s33_rot = _mm_mul_ps(s33_swapped, _mm_setr_ps(1.0f, -1.0f, -1.0f, 1.0f)); \
        out13 = _mm_add_ps(s11, s33_rot); \
    } while(0)

#define FFT8_CORE(v0, v1, v2, v3, out0, out1, out2, out3) \
    do { \
        v4sf v_ev0 = _mm_movelh_ps(v0, v1); \
        v4sf v_ev1 = _mm_movelh_ps(v2, v3); \
        v4sf v_od0 = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 2, 3, 2)); \
        v4sf v_od1 = _mm_shuffle_ps(v2, v3, _MM_SHUFFLE(3, 2, 3, 2)); \
        v4sf ev_out02, ev_out13; \
        v4sf od_out02, od_out13; \
        FFT4_CORE(v_ev0, v_ev1, ev_out02, ev_out13); \
        FFT4_CORE(v_od0, v_od1, od_out02, od_out13); \
        v4sf lh_mask = _mm_loadu_ps((const float*)lh_mask_int); \
        v4sf rot_sign_O2 = _mm_setr_ps(1.0f, 1.0f, -1.0f, 1.0f); \
        v4sf od_out02_swapped = V_SWAP_RI(od_out02); \
        v4sf od_out02_rot = _mm_mul_ps(od_out02_swapped, rot_sign_O2); \
        v4sf od_out02_tw = _mm_or_ps(_mm_and_ps(od_out02, lh_mask), _mm_andnot_ps(lh_mask, od_out02_rot)); \
        v4sf od_out13_swapped = V_SWAP_RI(od_out13); \
        v4sf sum = _mm_add_ps(od_out13, od_out13_swapped); \
        v4sf diff = _mm_sub_ps(od_out13, od_out13_swapped); \
        v4sf low = _mm_unpacklo_ps(sum, diff); \
        v4sf high = _mm_unpackhi_ps(sum, diff); \
        v4sf blended = _mm_shuffle_ps(low, high, _MM_SHUFFLE(0, 1, 1, 0)); \
        float c = 0.707106781f; \
        v4sf twiddle_sign_c = _mm_setr_ps(-c, c, c, c); \
        v4sf od_out13_tw = _mm_mul_ps(blended, twiddle_sign_c); \
        v4sf sum02 = _mm_add_ps(ev_out02, od_out02_tw); \
        v4sf diff02 = _mm_sub_ps(ev_out02, od_out02_tw); \
        v4sf sum13 = _mm_add_ps(ev_out13, od_out13_tw); \
        v4sf diff13 = _mm_sub_ps(ev_out13, od_out13_tw); \
        out0 = _mm_shuffle_ps(sum02, diff13, _MM_SHUFFLE(1, 0, 1, 0)); \
        out1 = _mm_shuffle_ps(diff02, diff13, _MM_SHUFFLE(3, 2, 3, 2)); \
        out2 = _mm_shuffle_ps(diff02, sum13, _MM_SHUFFLE(1, 0, 1, 0)); \
        out3 = _mm_shuffle_ps(sum02, sum13, _MM_SHUFFLE(3, 2, 3, 2)); \
    } while(0)

#define MUL_TWIDDLE(A, W, R, rot_sign) \
    do { \
        v4sf W_r = _mm_shuffle_ps(W, W, _MM_SHUFFLE(2, 2, 0, 0)); \
        v4sf W_i = _mm_shuffle_ps(W, W, _MM_SHUFFLE(3, 3, 1, 1)); \
        v4sf part1 = _mm_mul_ps(A, W_r); \
        v4sf part2 = _mm_mul_ps(V_SWAP_RI(A), W_i); \
        R = _mm_add_ps(part1, _mm_mul_ps(part2, rot_sign)); \
    } while(0)

#define FFT16_CORE(x0, x1, x2, x3, x4, x5, x6, x7, out0, out1, out2, out3, out4, out5, out6, out7) \
    do { \
        v4sf m_ev0 = _mm_movelh_ps(x0, x1); \
        v4sf m_ev1 = _mm_movelh_ps(x2, x3); \
        v4sf m_ev2 = _mm_movelh_ps(x4, x5); \
        v4sf m_ev3 = _mm_movelh_ps(x6, x7); \
        v4sf m_od0 = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(3, 2, 3, 2)); \
        v4sf m_od1 = _mm_shuffle_ps(x2, x3, _MM_SHUFFLE(3, 2, 3, 2)); \
        v4sf m_od2 = _mm_shuffle_ps(x4, x5, _MM_SHUFFLE(3, 2, 3, 2)); \
        v4sf m_od3 = _mm_shuffle_ps(x6, x7, _MM_SHUFFLE(3, 2, 3, 2)); \
        v4sf m_E0, m_E1, m_E2, m_E3; \
        v4sf m_O0, m_O1, m_O2, m_O3; \
        FFT8_CORE(m_ev0, m_ev1, m_ev2, m_ev3, m_E0, m_E1, m_E2, m_E3); \
        FFT8_CORE(m_od0, m_od1, m_od2, m_od3, m_O0, m_O1, m_O2, m_O3); \
        float m_c_1 = 0.92387953f; \
        float m_s_1 = 0.38268343f; \
        float m_c = 0.70710678f; \
        v4sf m_W_v0 = _mm_setr_ps(1.0f, 0.0f, m_c_1, m_s_1); \
        v4sf m_W_v1 = _mm_setr_ps(m_c, m_c, m_s_1, m_c_1); \
        v4sf m_W_v2 = _mm_setr_ps(0.0f, 1.0f, -m_s_1, m_c_1); \
        v4sf m_W_v3 = _mm_setr_ps(-m_c, m_c, -m_c_1, m_s_1); \
        v4sf m_rot_sign = _mm_setr_ps(1.0f, -1.0f, 1.0f, -1.0f); \
        v4sf m_O0_tw, m_O1_tw, m_O2_tw, m_O3_tw; \
        MUL_TWIDDLE(m_O0, m_W_v0, m_O0_tw, m_rot_sign); \
        MUL_TWIDDLE(m_O1, m_W_v1, m_O1_tw, m_rot_sign); \
        MUL_TWIDDLE(m_O2, m_W_v2, m_O2_tw, m_rot_sign); \
        MUL_TWIDDLE(m_O3, m_W_v3, m_O3_tw, m_rot_sign); \
        out0 = _mm_add_ps(m_E0, m_O0_tw); \
        out4 = _mm_sub_ps(m_E0, m_O0_tw); \
        out1 = _mm_add_ps(m_E1, m_O1_tw); \
        out5 = _mm_sub_ps(m_E1, m_O1_tw); \
        out2 = _mm_add_ps(m_E2, m_O2_tw); \
        out6 = _mm_sub_ps(m_E2, m_O2_tw); \
        out3 = _mm_add_ps(m_E3, m_O3_tw); \
        out7 = _mm_sub_ps(m_E3, m_O3_tw); \
    } while(0)


void celt_tx_fft4_fwd_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride) {
   (void)s;
   v4sf x01, x23;
   LOAD_V2(x01, in, (char*)in + stride);
   LOAD_V2(x23, (char*)in + 2*stride, (char*)in + 3*stride);

   v4sf v_a = _mm_shuffle_ps(x01, x23, _MM_SHUFFLE(3, 2, 1, 0));
   v4sf v_b = _mm_shuffle_ps(x23, x01, _MM_SHUFFLE(3, 2, 1, 0));

   v4sf s01 = _mm_add_ps(v_a, v_b);
   v4sf s13 = _mm_sub_ps(v_a, v_b);

   v4sf s00 = _mm_shuffle_ps(s01, s01, _MM_SHUFFLE(1, 0, 1, 0));
   v4sf s22 = _mm_shuffle_ps(s01, s01, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf s22_signed = _mm_mul_ps(s22, _mm_setr_ps(1.0f, 1.0f, -1.0f, -1.0f));
   v4sf out02 = _mm_add_ps(s00, s22_signed);

   v4sf s11 = _mm_shuffle_ps(s13, s13, _MM_SHUFFLE(1, 0, 1, 0));
   v4sf s33 = _mm_shuffle_ps(s13, s13, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf s33_swapped = V_SWAP_RI(s33);
   v4sf s33_rot = _mm_mul_ps(s33_swapped, _mm_setr_ps(1.0f, -1.0f, -1.0f, 1.0f));
   v4sf out13 = _mm_add_ps(s11, s33_rot);

   _mm_storel_pi((__m64*)out, out02);
   _mm_storeh_pi((__m64*)((char*)out + 2*stride), out02);
   _mm_storel_pi((__m64*)((char*)out + stride), out13);
   _mm_storeh_pi((__m64*)((char*)out + 3*stride), out13);
}
void celt_tx_fft8_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride) {
   (void)s;
   celt_assert2(stride == sizeof(cpx), "SSE power-of-two FFT called with non-contiguous stride");
   
   v4sf v0 = _mm_loadu_ps((float*)in);
   v4sf v1 = _mm_loadu_ps((float*)in + 4);
   v4sf v2 = _mm_loadu_ps((float*)in + 8);
   v4sf v3 = _mm_loadu_ps((float*)in + 12);

   // 1. De-interleave (basis4 style)
   v4sf v_ev0 = _mm_shuffle_ps(v0, v2, _MM_SHUFFLE(3, 2, 1, 0)); // [x[0], x[2]]
   v4sf v_ev1 = _mm_shuffle_ps(v2, v0, _MM_SHUFFLE(3, 2, 1, 0)); // [x[4], x[6]]
   v4sf v_od0 = _mm_shuffle_ps(v1, v3, _MM_SHUFFLE(1, 0, 3, 2)); // [x[1], x[3]]
   v4sf v_od1 = _mm_shuffle_ps(v3, v1, _MM_SHUFFLE(1, 0, 3, 2)); // [x[5], x[7]]

   // 2. FFT4 (IDFT)
   v4sf ev_out02, ev_out13;
   v4sf od_out02, od_out13;

   // Inlined FFT4_CORE_IDFT for evens
   {
       v4sf s01 = _mm_add_ps(v_ev0, v_ev1);
       v4sf s13 = _mm_sub_ps(v_ev0, v_ev1);
       v4sf s00 = _mm_shuffle_ps(s01, s01, _MM_SHUFFLE(1, 0, 1, 0));
       v4sf s22 = _mm_shuffle_ps(s01, s01, _MM_SHUFFLE(3, 2, 3, 2));
       v4sf s22_signed = _mm_mul_ps(s22, _mm_setr_ps(1.0f, 1.0f, -1.0f, -1.0f));
       ev_out02 = _mm_add_ps(s00, s22_signed);
       v4sf s11 = _mm_shuffle_ps(s13, s13, _MM_SHUFFLE(1, 0, 1, 0));
       v4sf s33 = _mm_shuffle_ps(s13, s13, _MM_SHUFFLE(3, 2, 3, 2));
       v4sf s33_swapped = V_SWAP_RI(s33);
       v4sf s33_rot = _mm_mul_ps(s33_swapped, _mm_setr_ps(1.0f, -1.0f, -1.0f, 1.0f));
       ev_out13 = _mm_add_ps(s11, s33_rot);
   }

   // Inlined FFT4_CORE_IDFT for odds
   {
       v4sf s01 = _mm_add_ps(v_od0, v_od1);
       v4sf s13 = _mm_sub_ps(v_od0, v_od1);
       v4sf s00 = _mm_shuffle_ps(s01, s01, _MM_SHUFFLE(1, 0, 1, 0));
       v4sf s22 = _mm_shuffle_ps(s01, s01, _MM_SHUFFLE(3, 2, 3, 2));
       v4sf s22_signed = _mm_mul_ps(s22, _mm_setr_ps(1.0f, 1.0f, -1.0f, -1.0f));
       od_out02 = _mm_add_ps(s00, s22_signed);
       v4sf s11 = _mm_shuffle_ps(s13, s13, _MM_SHUFFLE(1, 0, 1, 0));
       v4sf s33 = _mm_shuffle_ps(s13, s13, _MM_SHUFFLE(3, 2, 3, 2));
       v4sf s33_swapped = V_SWAP_RI(s33);
       v4sf s33_rot = _mm_mul_ps(s33_swapped, _mm_setr_ps(1.0f, -1.0f, -1.0f, 1.0f));
       od_out13 = _mm_add_ps(s11, s33_rot);
   }

   // 3. Twiddle odd
   v4sf lh_mask = _mm_loadu_ps((const float*)lh_mask_int);
   v4sf rot_sign_O2 = _mm_setr_ps(1.0f, 1.0f, -1.0f, 1.0f);
   v4sf od_out02_swapped = V_SWAP_RI(od_out02);
   v4sf od_out02_rot = _mm_mul_ps(od_out02_swapped, rot_sign_O2);
   // Blend: low of od_out02, high of od_out02_rot
   v4sf od_out02_tw = _mm_or_ps(_mm_and_ps(od_out02, lh_mask), _mm_andnot_ps(lh_mask, od_out02_rot));

   v4sf od_out13_swapped = V_SWAP_RI(od_out13);
   v4sf sum = _mm_add_ps(od_out13, od_out13_swapped);
   v4sf diff = _mm_sub_ps(od_out13, od_out13_swapped);
   v4sf low = _mm_unpacklo_ps(sum, diff);
   v4sf high = _mm_unpackhi_ps(sum, diff);
   v4sf blended = _mm_shuffle_ps(low, high, _MM_SHUFFLE(0, 1, 1, 0));
   float c = 0.707106781f;
   v4sf twiddle_sign_c = _mm_setr_ps(-c, c, c, c);
   v4sf od_out13_tw = _mm_mul_ps(blended, twiddle_sign_c);

   // 4. Combine
   v4sf sum02 = _mm_add_ps(ev_out02, od_out02_tw);
   v4sf diff02 = _mm_sub_ps(ev_out02, od_out02_tw);
   v4sf sum13 = _mm_add_ps(ev_out13, od_out13_tw);
   v4sf diff13 = _mm_sub_ps(ev_out13, od_out13_tw);

   // 5. Reconstruct
   v4sf out_v0 = _mm_shuffle_ps(sum02, diff13, _MM_SHUFFLE(1, 0, 1, 0));
   v4sf out_v1 = _mm_shuffle_ps(diff02, diff13, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf out_v2 = _mm_shuffle_ps(diff02, sum13, _MM_SHUFFLE(1, 0, 1, 0));
   v4sf out_v3 = _mm_shuffle_ps(sum02, sum13, _MM_SHUFFLE(3, 2, 3, 2));

   _mm_storeu_ps((float*)out, out_v0);
   _mm_storeu_ps((float*)out + 4, out_v1);
   _mm_storeu_ps((float*)out + 8, out_v2);
   _mm_storeu_ps((float*)out + 12, out_v3);
}
void celt_tx_fft16_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride) {
   (void)s;
   celt_assert2(stride == sizeof(cpx), "SSE power-of-two FFT called with non-contiguous stride");
   
   v4sf x0 = _mm_loadu_ps((float*)in);
   v4sf x1 = _mm_loadu_ps((float*)in + 4);
   v4sf x2 = _mm_loadu_ps((float*)in + 8);
   v4sf x3 = _mm_loadu_ps((float*)in + 12);
   v4sf x4 = _mm_loadu_ps((float*)in + 16);
   v4sf x5 = _mm_loadu_ps((float*)in + 20);
   v4sf x6 = _mm_loadu_ps((float*)in + 24);
   v4sf x7 = _mm_loadu_ps((float*)in + 28);

   v4sf v_a0 = _mm_movelh_ps(x0, x6);
   v4sf v_a1 = _mm_shuffle_ps(x1, x5, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a2 = _mm_shuffle_ps(x2, x7, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a3 = _mm_movelh_ps(x3, x5);
   v4sf v_a4 = _mm_movelh_ps(x2, x7);
   v4sf v_a5 = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a6 = _mm_shuffle_ps(x0, x6, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a7 = _mm_movelh_ps(x1, x4);

   v4sf out0, out1, out2, out3, out4, out5, out6, out7;
   FFT16_CORE(v_a0, v_a1, v_a2, v_a3, v_a4, v_a5, v_a6, v_a7, out0, out1, out2, out3, out4, out5, out6, out7);

   _mm_storeu_ps((float*)out, out0);
   _mm_storeu_ps((float*)out + 4, out1);
   _mm_storeu_ps((float*)out + 8, out2);
   _mm_storeu_ps((float*)out + 12, out3);
   _mm_storeu_ps((float*)out + 16, out4);
   _mm_storeu_ps((float*)out + 20, out5);
   _mm_storeu_ps((float*)out + 24, out6);
   _mm_storeu_ps((float*)out + 28, out7);
}
void celt_tx_fft32_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride) {
   (void)s;
   celt_assert2(stride == sizeof(cpx), "SSE power-of-two FFT called with non-contiguous stride");
   
   v4sf x0 = _mm_loadu_ps((float*)in);
   v4sf x1 = _mm_loadu_ps((float*)in + 4);
   v4sf x2 = _mm_loadu_ps((float*)in + 8);
   v4sf x3 = _mm_loadu_ps((float*)in + 12);
   v4sf x4 = _mm_loadu_ps((float*)in + 16);
   v4sf x5 = _mm_loadu_ps((float*)in + 20);
   v4sf x6 = _mm_loadu_ps((float*)in + 24);
   v4sf x7 = _mm_loadu_ps((float*)in + 28);
   v4sf x8 = _mm_loadu_ps((float*)in + 32);
   v4sf x9 = _mm_loadu_ps((float*)in + 36);
   v4sf x10 = _mm_loadu_ps((float*)in + 40);
   v4sf x11 = _mm_loadu_ps((float*)in + 44);
   v4sf x12 = _mm_loadu_ps((float*)in + 48);
   v4sf x13 = _mm_loadu_ps((float*)in + 52);
   v4sf x14 = _mm_loadu_ps((float*)in + 56);
   v4sf x15 = _mm_loadu_ps((float*)in + 60);

   // 1. Reconstruct basis4
   v4sf v_a0  = _mm_movelh_ps(x0, x12);
   v4sf v_a1  = _mm_shuffle_ps(x6, x9, _MM_SHUFFLE(3, 2, 1, 0));
   v4sf v_a2  = _mm_shuffle_ps(x1, x13, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a3  = _mm_shuffle_ps(x5, x10, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a4  = _mm_shuffle_ps(x2, x14, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a5  = _mm_shuffle_ps(x7, x11, _MM_SHUFFLE(1, 0, 3, 2));
   v4sf v_a6  = _mm_movelh_ps(x3, x15);
   v4sf v_a7  = _mm_movelh_ps(x5, x10);
   v4sf v_a8  = _mm_movelh_ps(x2, x14);
   v4sf v_a9  = _mm_shuffle_ps(x7, x11, _MM_SHUFFLE(3, 2, 1, 0));
   v4sf v_a10 = _mm_shuffle_ps(x3, x15, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a11 = _mm_shuffle_ps(x4, x8, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a12 = _mm_shuffle_ps(x0, x12, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf v_a13 = _mm_shuffle_ps(x6, x9, _MM_SHUFFLE(1, 0, 3, 2));
   v4sf v_a14 = _mm_movelh_ps(x1, x13);
   v4sf v_a15 = _mm_movelh_ps(x4, x8);

   // 2. De-interleave (basis2 style) on v_a
   v4sf ev0 = _mm_movelh_ps(v_a0, v_a1);
   v4sf ev1 = _mm_movelh_ps(v_a2, v_a3);
   v4sf ev2 = _mm_movelh_ps(v_a4, v_a5);
   v4sf ev3 = _mm_movelh_ps(v_a6, v_a7);
   v4sf ev4 = _mm_movelh_ps(v_a8, v_a9);
   v4sf ev5 = _mm_movelh_ps(v_a10, v_a11);
   v4sf ev6 = _mm_movelh_ps(v_a12, v_a13);
   v4sf ev7 = _mm_movelh_ps(v_a14, v_a15);

   v4sf od0 = _mm_shuffle_ps(v_a0, v_a1, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf od1 = _mm_shuffle_ps(v_a2, v_a3, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf od2 = _mm_shuffle_ps(v_a4, v_a5, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf od3 = _mm_shuffle_ps(v_a6, v_a7, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf od4 = _mm_shuffle_ps(v_a8, v_a9, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf od5 = _mm_shuffle_ps(v_a10, v_a11, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf od6 = _mm_shuffle_ps(v_a12, v_a13, _MM_SHUFFLE(3, 2, 3, 2));
   v4sf od7 = _mm_shuffle_ps(v_a14, v_a15, _MM_SHUFFLE(3, 2, 3, 2));

   // 2. FFT16
   v4sf E0, E1, E2, E3, E4, E5, E6, E7;
   v4sf O0, O1, O2, O3, O4, O5, O6, O7;
   FFT16_CORE(ev0, ev1, ev2, ev3, ev4, ev5, ev6, ev7, E0, E1, E2, E3, E4, E5, E6, E7);
   FFT16_CORE(od0, od1, od2, od3, od4, od5, od6, od7, O0, O1, O2, O3, O4, O5, O6, O7);

   // 3. Twiddle O
   float c_1 = 0.98078528f;
   float s_1 = 0.19509032f;
   float c_2 = 0.92387953f;
   float s_2 = 0.38268343f;
   float c_3 = 0.83146961f;
   float s_3 = 0.55557023f;
   float c = 0.70710678f;

   v4sf W_v0 = _mm_setr_ps(1.0f, 0.0f, c_1, s_1);
   v4sf W_v1 = _mm_setr_ps(c_2, s_2, c_3, s_3);
   v4sf W_v2 = _mm_setr_ps(c, c, s_3, c_3);
   v4sf W_v3 = _mm_setr_ps(s_2, c_2, s_1, c_1);
   v4sf W_v4 = _mm_setr_ps(0.0f, 1.0f, -s_1, c_1);
   v4sf W_v5 = _mm_setr_ps(-s_2, c_2, -s_3, c_3);
   v4sf W_v6 = _mm_setr_ps(-c, c, -c_3, s_3);
   v4sf W_v7 = _mm_setr_ps(-c_2, s_2, -c_1, s_1);

   v4sf rot_sign = _mm_setr_ps(1.0f, -1.0f, 1.0f, -1.0f);
   v4sf O0_tw, O1_tw, O2_tw, O3_tw, O4_tw, O5_tw, O6_tw, O7_tw;
   MUL_TWIDDLE(O0, W_v0, O0_tw, rot_sign);
   MUL_TWIDDLE(O1, W_v1, O1_tw, rot_sign);
   MUL_TWIDDLE(O2, W_v2, O2_tw, rot_sign);
   MUL_TWIDDLE(O3, W_v3, O3_tw, rot_sign);
   MUL_TWIDDLE(O4, W_v4, O4_tw, rot_sign);
   MUL_TWIDDLE(O5, W_v5, O5_tw, rot_sign);
   MUL_TWIDDLE(O6, W_v6, O6_tw, rot_sign);
   MUL_TWIDDLE(O7, W_v7, O7_tw, rot_sign);

   // 4. Combine & Store (Natural Order)
   v4sf out_v0 = _mm_add_ps(E0, O0_tw);
   v4sf out_v8 = _mm_sub_ps(E0, O0_tw);
   v4sf out_v1 = _mm_add_ps(E1, O1_tw);
   v4sf out_v9 = _mm_sub_ps(E1, O1_tw);
   v4sf out_v2 = _mm_add_ps(E2, O2_tw);
   v4sf out_v10 = _mm_sub_ps(E2, O2_tw);
   v4sf out_v3 = _mm_add_ps(E3, O3_tw);
   v4sf out_v11 = _mm_sub_ps(E3, O3_tw);
   v4sf out_v4 = _mm_add_ps(E4, O4_tw);
   v4sf out_v12 = _mm_sub_ps(E4, O4_tw);
   v4sf out_v5 = _mm_add_ps(E5, O5_tw);
   v4sf out_v13 = _mm_sub_ps(E5, O5_tw);
   v4sf out_v6 = _mm_add_ps(E6, O6_tw);
   v4sf out_v14 = _mm_sub_ps(E6, O6_tw);
   v4sf out_v7 = _mm_add_ps(E7, O7_tw);
   v4sf out_v15 = _mm_sub_ps(E7, O7_tw);

   _mm_storeu_ps((float*)out, out_v0);
   _mm_storeu_ps((float*)out + 4, out_v1);
   _mm_storeu_ps((float*)out + 8, out_v2);
   _mm_storeu_ps((float*)out + 12, out_v3);
   _mm_storeu_ps((float*)out + 16, out_v4);
   _mm_storeu_ps((float*)out + 20, out_v5);
   _mm_storeu_ps((float*)out + 24, out_v6);
   _mm_storeu_ps((float*)out + 28, out_v7);
   _mm_storeu_ps((float*)out + 32, out_v8);
   _mm_storeu_ps((float*)out + 36, out_v9);
   _mm_storeu_ps((float*)out + 40, out_v10);
   _mm_storeu_ps((float*)out + 44, out_v11);
   _mm_storeu_ps((float*)out + 48, out_v12);
   _mm_storeu_ps((float*)out + 52, out_v13);
   _mm_storeu_ps((float*)out + 56, out_v14);
   _mm_storeu_ps((float*)out + 60, out_v15);
}
void celt_tx_fft_sr_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride) {
   celt_assert2(stride == sizeof(cpx), "SSE power-of-two FFT called with non-contiguous stride");
   int N = s->len;
   const opus_int16 *map = s->map;
   if (map) {
      VARDECL(cpx, tmp);
      SAVE_STACK;
      ALLOC(tmp, N, cpx);
      for (int i = 0; i < N; i++) {
         tmp[i] = ((cpx*)in)[map[i]];
      }
      celt_tx_fft_p2_c((cpx*)out, tmp, N);
      RESTORE_STACK;
   } else {
      celt_tx_fft_p2_c((cpx*)out, (const cpx*)in, N);
   }
}
#define SIN_2PI_3  0.866025388f
#define COS_2PI_5  0.309017003f
#define SIN_2PI_5  0.951056540f
#define COS_4PI_5  0.809016994f
#define SIN_4PI_5  0.587785252f
#define HALF_VAL   0.5f

static const int celt_tx_pfa_P[15] = {2, 5, 7, 14, 4, 1, 13, 3, 10, 12, 0, 9, 11, 6, 8};

static OPUS_INLINE void winograd_fft3_v2(v4sf in0, v4sf in1, v4sf in2, v4sf *out0, v4sf *out1, v4sf *out2) {
   v4sf sum = _mm_add_ps(in1, in2);
   v4sf diff = _mm_sub_ps(in1, in2);

   *out0 = _mm_add_ps(in0, sum);

   v4sf t2 = _mm_mul_ps(sum, _mm_set1_ps(HALF_VAL));
   v4sf diff_swapped = V_SWAP_RI(diff);

   v4sf sin_const = _mm_setr_ps(SIN_2PI_3, -SIN_2PI_3, SIN_2PI_3, -SIN_2PI_3);
   v4sf t1 = _mm_mul_ps(diff_swapped, sin_const);

   v4sf base = _mm_sub_ps(in0, t2);
   *out1 = _mm_add_ps(base, t1);
   *out2 = _mm_sub_ps(base, t1);
}

static OPUS_INLINE void decl_fft5_v2(const v4sf *in, int r3, cpx *out0, cpx *out1, int stride) {
   v4sf dc = in[0];
   v4sf sum14 = _mm_add_ps(in[1], in[4]);
   v4sf diff14 = _mm_sub_ps(in[1], in[4]);
   v4sf sum23 = _mm_add_ps(in[2], in[3]);
   v4sf diff23 = _mm_sub_ps(in[2], in[3]);

   int idx0 = (5 * r3) % 15;
   if (idx0 < 0) idx0 += 15;

   v4sf out0_val = _mm_add_ps(dc, _mm_add_ps(sum14, sum23));
   _mm_storel_pi((__m64*)(out0 + idx0 * stride), out0_val);
   _mm_storeh_pi((__m64*)(out1 + idx0 * stride), out0_val);

   v4sf t4 = _mm_sub_ps(_mm_mul_ps(sum14, _mm_set1_ps(COS_2PI_5)), _mm_mul_ps(sum23, _mm_set1_ps(COS_4PI_5)));
   v4sf t0 = _mm_sub_ps(_mm_mul_ps(sum23, _mm_set1_ps(COS_2PI_5)), _mm_mul_ps(sum14, _mm_set1_ps(COS_4PI_5)));

   v4sf sin_5_const1 = _mm_setr_ps(SIN_2PI_5, -SIN_2PI_5, SIN_2PI_5, -SIN_2PI_5);
   v4sf sin_5_const2 = _mm_setr_ps(SIN_4PI_5, -SIN_4PI_5, SIN_4PI_5, -SIN_4PI_5);

   v4sf t5_part1 = _mm_mul_ps(V_SWAP_RI(diff14), sin_5_const1);
   v4sf t5_part2 = _mm_mul_ps(V_SWAP_RI(diff23), sin_5_const2);
   v4sf t5 = _mm_add_ps(t5_part1, t5_part2);

   v4sf t1_part1 = _mm_mul_ps(V_SWAP_RI(diff14), sin_5_const2);
   v4sf t1_part2 = _mm_mul_ps(V_SWAP_RI(diff23), sin_5_const1);
   v4sf t1 = _mm_sub_ps(t1_part1, t1_part2);

   int idx1 = (5 * r3 + 3) % 15; if (idx1 < 0) idx1 += 15;
   int idx2 = (5 * r3 + 6) % 15; if (idx2 < 0) idx2 += 15;
   int idx3 = (5 * r3 + 9) % 15; if (idx3 < 0) idx3 += 15;
   int idx4 = (5 * r3 + 12) % 15; if (idx4 < 0) idx4 += 15;

   v4sf out1_val = _mm_add_ps(dc, _mm_add_ps(t4, t5));
   _mm_storel_pi((__m64*)(out0 + idx1 * stride), out1_val);
   _mm_storeh_pi((__m64*)(out1 + idx1 * stride), out1_val);

   v4sf out2_val = _mm_add_ps(dc, _mm_add_ps(t0, t1));
   _mm_storel_pi((__m64*)(out0 + idx2 * stride), out2_val);
   _mm_storeh_pi((__m64*)(out1 + idx2 * stride), out2_val);

   v4sf out3_val = _mm_add_ps(dc, _mm_sub_ps(t0, t1));
   _mm_storel_pi((__m64*)(out0 + idx3 * stride), out3_val);
   _mm_storeh_pi((__m64*)(out1 + idx3 * stride), out3_val);

   v4sf out4_val = _mm_add_ps(dc, _mm_sub_ps(t4, t5));
   _mm_storel_pi((__m64*)(out0 + idx4 * stride), out4_val);
   _mm_storeh_pi((__m64*)(out1 + idx4 * stride), out4_val);
}

static void celt_tx_fft15_sse(const cpx *in0, const cpx *in1, cpx *out0, cpx *out1, int stride) {
   v4sf tmp[15];

   for (int c5 = 0; c5 < 5; c5++) {
      v4sf in_col[3];
      v4sf out_col[3];
      for (int r3 = 0; r3 < 3; r3++) {
         int r_pfa = celt_tx_pfa_P[(10 * r3 + 6 * c5) % 15];
         LOAD_V2(in_col[r3], &in0[r_pfa], &in1[r_pfa]);
      }
      winograd_fft3_v2(in_col[0], in_col[1], in_col[2], &out_col[0], &out_col[1], &out_col[2]);
      tmp[c5] = out_col[0];
      tmp[c5 + 5] = out_col[1];
      tmp[c5 + 10] = out_col[2];
   }

   decl_fft5_v2(tmp, 0, out0, out1, stride);
   decl_fft5_v2(tmp + 5, 1, out0, out1, stride);
   decl_fft5_v2(tmp + 10, 2, out0, out1, stride);
}

void celt_tx_fft_pfa_15xM_ns_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride) {
   int i, j;
   int len = s->len;
   int M = s->sub->len;
   cpx *tmp = (cpx *)s->tmp;
   const cpx *in_cpx = (const cpx *)in;
   cpx *out_cpx = (cpx *)out;
   const opus_int16 *sub_map = s->sub->map;
   (void)stride;

   for (i = 0; i < M; i += 2) {
      cpx *out0 = tmp + sub_map[i];
      cpx *out1 = tmp + sub_map[i+1];
      celt_tx_fft15_sse(in_cpx + 15 * i, in_cpx + 15 * (i + 1), out0, out1, M);
   }

   typedef void (*p2_fn)(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
   p2_fn sub_fn = (p2_fn)s->fn;

   for (j = 0; j < 15; j++) {
      sub_fn(s->sub, tmp + j * M, tmp + j * M, sizeof(cpx));
   }

   for (i = 0; i < len; i++) {
      out_cpx[i] = tmp[s->map[i]];
   }
}
void celt_tx_mdct_inv_float_sse(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride) {
   int i;
   int N2 = s->len;
   int N4 = N2 >> 1;
   cpx *z = (cpx *)out;
   const kiss_twiddle_scalar *trig = (const kiss_twiddle_scalar *)s->exp;


   v4sf sign_mask = _mm_setr_ps(-1.0f, 1.0f, -1.0f, 1.0f);
   for (i = 0; i < N4; i += 2) {
      int k0 = s->map[i];
      int k1 = s->map[i+1];
      int j0 = k0 >> 1;
      int j1 = k1 >> 1;

      float im0 = *(const float*)((const char*)in + k0 * stride);
      float re0 = *(const float*)((const char*)in + (N2 - 1 - k0) * stride);
      float im1 = *(const float*)((const char*)in + k1 * stride);
      float re1 = *(const float*)((const char*)in + (N2 - 1 - k1) * stride);

      v4sf v_im0 = _mm_load_ss(&im0);
      v4sf v_re0 = _mm_load_ss(&re0);
      v4sf v_im1 = _mm_load_ss(&im1);
      v4sf v_re1 = _mm_load_ss(&re1);
      v4sf v_im_re_0 = _mm_unpacklo_ps(v_im0, v_re0);
      v4sf v_im_re_1 = _mm_unpacklo_ps(v_im1, v_re1);
      v4sf im_re = _mm_movelh_ps(v_im_re_0, v_im_re_1);

      v4sf trig_v;
      LOAD_V2(trig_v, trig + 2 * j0, trig + 2 * j1);

      v4sf z_val = _mm_add_ps(
         _mm_mul_ps(im_re, _mm_shuffle_ps(trig_v, trig_v, _MM_SHUFFLE(3, 3, 1, 1))),
         _mm_mul_ps(
            V_SWAP_RI(im_re),
            _mm_mul_ps(
               _mm_shuffle_ps(trig_v, trig_v, _MM_SHUFFLE(2, 2, 0, 0)),
               sign_mask
            )
         )
      );

      _mm_storeu_ps((float*)(z + i), z_val);
   }

   typedef void (*fft_fn)(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride);
   fft_fn fn = (fft_fn)s->fn;
   fn(s->sub, z, z, sizeof(cpx));

   v4sf blend_mask = _mm_castsi128_ps(_mm_setr_epi32(-1, 0, -1, 0));
   v4sf rot_sign = _mm_setr_ps(1.0f, -1.0f, 1.0f, -1.0f);

   int limit = (N4 / 2) & ~1;
   for (i = 0; i < limit; i += 2) {
      int i0_0 = i;
      int i1_1 = N4 - 2 - i;

      v4sf z0_v = _mm_loadu_ps((float*)(z + i0_0));
      v4sf z1_v = _mm_loadu_ps((float*)(z + i1_1));

      v4sf e0_v = _mm_loadu_ps(trig + 2 * i0_0);
      v4sf e1_v = _mm_loadu_ps(trig + 2 * i1_1);

      v4sf z0_i = _mm_shuffle_ps(z0_v, z0_v, _MM_SHUFFLE(3, 3, 1, 1));
      v4sf z0_r = _mm_shuffle_ps(z0_v, z0_v, _MM_SHUFFLE(2, 2, 0, 0));
      v4sf part1_0 = _mm_mul_ps(z0_i, e0_v);
      v4sf part2_0 = _mm_mul_ps(z0_r, e0_v);
      v4sf r0_v = _mm_add_ps(V_SWAP_RI(part1_0), _mm_mul_ps(part2_0, rot_sign));

      v4sf z1_i = _mm_shuffle_ps(z1_v, z1_v, _MM_SHUFFLE(3, 3, 1, 1));
      v4sf z1_r = _mm_shuffle_ps(z1_v, z1_v, _MM_SHUFFLE(2, 2, 0, 0));
      v4sf part1_1 = _mm_mul_ps(z1_i, e1_v);
      v4sf part2_1 = _mm_mul_ps(z1_r, e1_v);
      v4sf r1_v = _mm_add_ps(V_SWAP_RI(part1_1), _mm_mul_ps(part2_1, rot_sign));

      v4sf r0_shuf = _mm_shuffle_ps(r0_v, r0_v, _MM_SHUFFLE(1, 2, 3, 0));
      v4sf r1_shuf = _mm_shuffle_ps(r1_v, r1_v, _MM_SHUFFLE(1, 2, 3, 0));

      v4sf new_z0 = _mm_or_ps(_mm_and_ps(r0_shuf, blend_mask), _mm_andnot_ps(blend_mask, r1_shuf));
      v4sf new_z1 = _mm_or_ps(_mm_and_ps(r1_shuf, blend_mask), _mm_andnot_ps(blend_mask, r0_shuf));

      _mm_storeu_ps((float*)(z + i0_0), new_z0);
      _mm_storeu_ps((float*)(z + i1_1), new_z1);
   }

   if ((N4 / 2) & 1) {
      int i0 = N4 / 2 - 1;
      int i1 = N4 / 2;
      cpx z0 = z[i0];
      cpx z1 = z[i1];
      kiss_twiddle_scalar e0_r = trig[2 * i0 + 1];
      kiss_twiddle_scalar e0_i = trig[2 * i0];
      kiss_twiddle_scalar e1_r = trig[2 * i1 + 1];
      kiss_twiddle_scalar e1_i = trig[2 * i1];
      cpx r0 = (cpx){z0.i * e0_i + z0.r * e0_r, z0.i * e0_r - z0.r * e0_i};
      cpx r1 = (cpx){z1.i * e1_i + z1.r * e1_r, z1.i * e1_r - z1.r * e1_i};
      z[i0] = (cpx){r0.r, r1.i};
      z[i1] = (cpx){r1.r, r0.i};
   }
}

#endif
