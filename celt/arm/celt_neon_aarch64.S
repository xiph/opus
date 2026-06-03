/* Copyright (c) 2026 Lynne */
/* Ported and relicensed from FFmpeg, libavcodec/aarch64/opusdsp_neon.S */
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

/*
 * AArch64 NEON CELT post-filter and de-emphasis, ported from FFmpeg's
 * libavcodec/aarch64/opusdsp_neon.S:
 *
 *   comb_filter_const_neon         <- ff_opus_postfilter_neon
 *   celt_deemphasis_neon           <- ff_opus_deemphasis_neon
 *   deemphasis_stereo_simple_neon  <- ff_opus_deemphasis_neon, twice (one
 *                                     chain per channel, v16-v19 for the
 *                                     second one), interleaved output
 *
 * The loop bodies are kept identical to FFmpeg's. Differences:
 *   (N) FFmpeg's do-while loops assume N is a positive multiple of 4 / 8.
 *       libopus passes N == 0 (comb_filter() with N == overlap when the
 *       filter changed) and, in custom modes, any even N. The vector loops
 *       only run for whole vectors and a scalar tail does the rest.
 *   (M) libopus keeps mem = coef * y_last, FFmpeg keeps y_last. The initial
 *       accumulator is {1, c, c^2, c^3} * mem and the new mem is returned.
 *   (S) Output is scaled by 1/CELT_SIG_SCALE (SIG2RES) before the store.
 *   (W) Weights {c, c^2, c^3, c^4} are computed from coef, not passed in.
 *   (C) Stereo: every channel-A line has a channel-B twin.
 */

#if defined(__aarch64__) && !defined(FIXED_POINT)

        .arch   armv8-a

        .section .rodata
        .align  2
LSIG2RES_inv_scale:
        .float  3.0517578125e-5         /* 1.f / CELT_SIG_SCALE */

        .text

        .macro  function name
        .global \name
#ifdef __ELF__
        .type   \name, %function
#endif
        .align  2
\name\():
        .endm

        .macro  endfunc name
#ifdef __ELF__
        .size   \name, .-\name
#endif
        .endm


/*
 * void comb_filter_const_neon(opus_val32 *y, opus_val32 *x, int T, int N,
 *                             opus_val16 g10, opus_val16 g11, opus_val16 g12);
 * x0 = y, x1 = x, w2 = T (>= COMBFILTER_MINPERIOD), w3 = N (<= 0 is a no-op),
 * s0..s2 = g10..g12. In-place when x == y.
 */
        function comb_filter_const_neon
        sub     x4, x1, w2, sxtw #2         /* x - T */
        cmp     w3, #4
        b.lt    2f                          /* (N) fewer than 4: tail only */

        sub     x5, x4, #8                  /* x - T - 2 */
        dup     v1.4s, v1.s[0]              /* g1 */
        dup     v2.4s, v2.s[0]              /* g2 */
        dup     v0.4s, v0.s[0]              /* g0 */

        ld1     {v3.4s}, [x5], #16
        sub     x6, x4, #4                  /* x - T - 1 */
        add     x7, x4, #4                  /* x - T + 1 */
        fmul    v3.4s, v3.4s, v2.4s
        sub     w3, w3, #4                  /* (N) */

1:      ld1     {v7.4s}, [x5], #16
        ld1     {v4.4s}, [x6], #16
        fmla    v3.4s, v7.4s, v2.4s
        ld1     {v6.4s}, [x7], #16
        ld1     {v5.4s}, [x4], #16
        fadd    v6.4s, v6.4s, v4.4s
        fmla    v3.4s, v5.4s, v0.4s
        ld1     {v4.4s}, [x1], #16          /* x[i..i+3] (FFmpeg reads [x0]) */
        fmla    v3.4s, v6.4s, v1.4s
        fadd    v4.4s, v4.4s, v3.4s
        fmul    v3.4s, v7.4s, v2.4s
        st1     {v4.4s}, [x0], #16          /* y[i..i+3] */
        subs    w3, w3, #4
        b.ge    1b                          /* (N) FFmpeg: b.gt */
        add     w3, w3, #4                  /* (N) w3 = N & 3 */

        /* (N) Scalar tail, same tap order as the vector loop. */
2:      cmp     w3, #0
        b.le    3f
4:      ldr     s3, [x4, #-8]               /* x[i-T-2] */
        ldr     s7, [x4, #8]                /* x[i-T+2] */
        ldr     s4, [x4, #-4]               /* x[i-T-1] */
        ldr     s6, [x4, #4]                /* x[i-T+1] */
        ldr     s5, [x4], #4                /* x[i-T]   */
        ldr     s16, [x1], #4               /* x[i]     */
        fmul    s3, s3, s2
        fadd    s6, s6, s4
        fmadd   s3, s7, s2, s3
        fmadd   s3, s5, s0, s3
        fmadd   s3, s6, s1, s3
        fadd    s16, s16, s3
        str     s16, [x0], #4               /* y[i]     */
        subs    w3, w3, #1
        b.gt    4b
3:      ret
        endfunc comb_filter_const_neon


/*
 * opus_val32 celt_deemphasis_neon(opus_res *y, const opus_val32 *x,
 *                                 opus_val16 coef, opus_val32 m, int N);
 * x0 = y, x1 = x, s0 = coef, s1 = m, w2 = N (<= 0 is a no-op).
 * Returns the new m (= coef * y_last) in s0.
 */
        function celt_deemphasis_neon
        /* (W) v4 = {c, c^2, c^3, c^4}, v5..v7 = v4 shifted right by 1..3 lanes */
        fmul    s17, s0, s0
        fmul    s18, s0, s17
        fmul    s19, s17, s17
        movi    v4.16b, #0
        ins     v4.s[0], v0.s[0]
        ins     v4.s[1], v17.s[0]
        ins     v4.s[2], v18.s[0]
        ins     v4.s[3], v19.s[0]
        movi    v20.16b, #0
        ext     v5.16b, v20.16b, v4.16b, #12
        ext     v6.16b, v20.16b, v4.16b, #8
        ext     v7.16b, v20.16b, v4.16b, #4

        adrp    x3, LSIG2RES_inv_scale      /* (S) */
        add     x3, x3, :lo12:LSIG2RES_inv_scale
        ld1r    {v24.4s}, [x3]

        /* (M) v0 = {1, c, c^2, c^3} * m  (FFmpeg: v4 * y_last) */
        fmul    v0.4s, v5.4s, v1.s[0]
        ins     v0.s[0], v1.s[0]

        subs    w2, w2, #8                  /* (N) */
        b.lt    2f

1:      ld1     {v1.4s, v2.4s}, [x1], #32
        fmla    v0.4s, v5.4s, v1.s[0]
        fmul    v3.4s, v7.4s, v2.s[2]
        fmla    v0.4s, v6.4s, v1.s[1]
        fmla    v3.4s, v6.4s, v2.s[1]
        fmla    v0.4s, v7.4s, v1.s[2]
        fmla    v3.4s, v5.4s, v2.s[0]
        fadd    v1.4s, v1.4s, v0.4s
        fadd    v2.4s, v2.4s, v3.4s
        fmla    v2.4s, v4.4s, v1.s[3]
        fmul    v0.4s, v4.4s, v2.s[3]       /* (M) prime; FFmpeg does this after the st1 */
        fmul    v1.4s, v1.4s, v24.4s        /* (S) */
        fmul    v2.4s, v2.4s, v24.4s
        st1     {v1.4s, v2.4s}, [x0], #32
        subs    w2, w2, #8
        b.ge    1b                          /* (N) FFmpeg: b.gt */

        /* (N) Scalar tail: tmp = x + m; m = c * tmp; y = SIG2RES(tmp). m is s0. */
2:      adds    w2, w2, #8
        b.le    3f
4:      ldr     s1, [x1], #4
        fadd    s1, s1, s0
        fmul    s0, s1, s4
        fmul    s1, s1, s24
        str     s1, [x0], #4
        subs    w2, w2, #1
        b.gt    4b
3:      ret                                 /* (M) s0 = c * y_last */
        endfunc celt_deemphasis_neon


/*
 * void deemphasis_stereo_simple_neon(celt_sig *in[2], opus_res *pcm,
 *                                    int N, opus_val16 coef, celt_sig *mem);
 * x0 = in, x1 = pcm (interleaved), w2 = N (<= 0 is a no-op), s0 = coef,
 * x3 = mem[2]. (C) celt_deemphasis_neon with v16-v19 as channel B.
 */
        function deemphasis_stereo_simple_neon
        ldp     x4, x5, [x0]                /* in[0], in[1] */

        /* (W) */
        fmul    s17, s0, s0
        fmul    s18, s0, s17
        fmul    s19, s17, s17
        movi    v4.16b, #0
        ins     v4.s[0], v0.s[0]
        ins     v4.s[1], v17.s[0]
        ins     v4.s[2], v18.s[0]
        ins     v4.s[3], v19.s[0]
        movi    v20.16b, #0
        ext     v5.16b, v20.16b, v4.16b, #12
        ext     v6.16b, v20.16b, v4.16b, #8
        ext     v7.16b, v20.16b, v4.16b, #4

        adrp    x6, LSIG2RES_inv_scale      /* (S) */
        add     x6, x6, :lo12:LSIG2RES_inv_scale
        ld1r    {v24.4s}, [x6]

        /* (M) */
        ldp     s28, s29, [x3]
        fmul    v0.4s,  v5.4s, v28.s[0]
        fmul    v16.4s, v5.4s, v29.s[0]
        ins     v0.s[0],  v28.s[0]
        ins     v16.s[0], v29.s[0]

        subs    w2, w2, #8                  /* (N) */
        b.lt    2f

1:      ld1     {v1.4s,  v2.4s},  [x4], #32
        ld1     {v17.4s, v18.4s}, [x5], #32
        fmla    v0.4s,  v5.4s, v1.s[0]
        fmla    v16.4s, v5.4s, v17.s[0]
        fmul    v3.4s,  v7.4s, v2.s[2]
        fmul    v19.4s, v7.4s, v18.s[2]
        fmla    v0.4s,  v6.4s, v1.s[1]
        fmla    v16.4s, v6.4s, v17.s[1]
        fmla    v3.4s,  v6.4s, v2.s[1]
        fmla    v19.4s, v6.4s, v18.s[1]
        fmla    v0.4s,  v7.4s, v1.s[2]
        fmla    v16.4s, v7.4s, v17.s[2]
        fmla    v3.4s,  v5.4s, v2.s[0]
        fmla    v19.4s, v5.4s, v18.s[0]
        fadd    v1.4s,  v1.4s,  v0.4s
        fadd    v17.4s, v17.4s, v16.4s
        fadd    v2.4s,  v2.4s,  v3.4s
        fadd    v18.4s, v18.4s, v19.4s
        fmla    v2.4s,  v4.4s, v1.s[3]
        fmla    v18.4s, v4.4s, v17.s[3]
        fmul    v0.4s,  v4.4s, v2.s[3]        /* (M) prime */
        fmul    v16.4s, v4.4s, v18.s[3]
        fmul    v1.4s,  v1.4s,  v24.4s        /* (S) */
        fmul    v17.4s, v17.4s, v24.4s
        fmul    v2.4s,  v2.4s,  v24.4s
        fmul    v18.4s, v18.4s, v24.4s
        zip1    v20.4s, v1.4s,  v17.4s        /* interleave L/R */
        zip2    v21.4s, v1.4s,  v17.4s
        zip1    v22.4s, v2.4s,  v18.4s
        zip2    v23.4s, v2.4s,  v18.4s
        st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
        subs    w2, w2, #8
        b.ge    1b                            /* (N) FFmpeg: b.gt */

        /* (N) Scalar tail, both channels; m0 = s0, m1 = s16. */
2:      adds    w2, w2, #8
        b.le    3f
4:      ldr     s1,  [x4], #4
        ldr     s17, [x5], #4
        fadd    s1,  s1,  s0
        fadd    s17, s17, s16
        fmul    s0,  s1,  s4
        fmul    s16, s17, s4
        fmul    s1,  s1,  s24
        fmul    s17, s17, s24
        stp     s1, s17, [x1], #8
        subs    w2, w2, #1
        b.gt    4b
3:      stp     s0, s16, [x3]                 /* (M) mem = c * y_last */
        ret
        endfunc deemphasis_stereo_simple_neon

#endif /* __aarch64__ && !FIXED_POINT */
