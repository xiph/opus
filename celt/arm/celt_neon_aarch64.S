/* Copyright (c) 2026 Lynne */
/* Ported and relicensed from FFmpeg, libavcodec/aarch64/opusdsp_neon.S */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
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
 * AArch64 NEON kernels for the CELT post-filter and CELT de-emphasis
 * (mono + stereo), ported from FFmpeg's
 *   libavcodec/aarch64/opusdsp_neon.S
 *
 * The kernels are deliberately kept STRUCTURALLY IDENTICAL to FFmpeg's
 * so that future updates to that file can be applied here mechanically:
 *
 *   - comb_filter_const_neon          mirrors ff_opus_postfilter_neon
 *   - celt_deemphasis_neon            mirrors ff_opus_deemphasis_neon
 *   - deemphasis_stereo_simple_neon   == celt_deemphasis_neon doubled,
 *                                       one chain per channel
 *
 * Lines that differ from FFmpeg are flagged with `LIBOPUS DELTA` and
 * the corresponding FFmpeg line is shown inline. The four classes of
 * delta in the de-emphasis kernels are tagged (M), (S), (W), (C):
 *
 *   (M) Memory convention. FFmpeg stores  coeff = y_last  between
 *       calls; libopus stores  mem = coef * y_last  (see arch.h's
 *       MULT16_32_Q15). The two are related by a single multiplication
 *       by coef. We compensate by initialising the accumulator as
 *       `v_init * m` with v_init = {1, c, c^2, c^3} rather than
 *       FFmpeg's `v4 * coeff` with v4 = {c, c^2, c^3, c^4}; both yield
 *       the same {c·y, c²·y, c³·y, c⁴·y}. At the return we hand back
 *       `c * y_last` (= libopus's new m), not y_last.
 *
 *   (S) SIG2RES scaling. libopus de-emphasis crosses the signal ->
 *       result domain boundary: input is celt_sig (~[-32768, 32768]
 *       in float), output is opus_res ([-1, 1]). The C reference
 *       applies  pcm = SIG2RES(tmp) = tmp / CELT_SIG_SCALE  on the
 *       way out; we do the same with an `fmul` against v24 = {1/32768}
 *       just before the store. FFmpeg's internal domain doesn't need
 *       this. NOTE: because SIG2RES clobbers v2 (the second-half data
 *       register), the prime  fmul v0, v4, v2.s[3]  is hoisted ABOVE
 *       the scaling so it sees v2 in the signal domain.
 *
 *   (W) Runtime weight generation. FFmpeg's callers pre-compute the
 *       weight matrix and pass it as a pointer (x2); libopus's don't,
 *       so each function builds the {c, c^2, c^3, c^4} powers and the
 *       three rotated rows (v5, v6, v7) from the scalar `coef` argument.
 *
 *   (C) Channel doubling (stereo only). FFmpeg's de-emphasis is
 *       single-channel. libopus's most popular path is interleaved
 *       stereo (downsample==1, C==2, !accum), so the stereo function
 *       runs the FFmpeg kernel TWICE in parallel — once per channel —
 *       sharing the weight vectors. Every FFmpeg fmla/fmul/fadd line
 *       has its (C) twin in deemphasis_stereo_simple_neon, using
 *       v16/v17/v18/v19 for channel B in place of v0/v1/v2/v3. The
 *       per-iter output is interleaved with vzip1/vzip2 + st1 (FFmpeg
 *       uses st1 of two consecutive vectors for its contiguous output).
 */

#if defined(__aarch64__) && !defined(FIXED_POINT)

/* Shared ELF+Mach-O macros: function/endfunc/const/endconst/movrel + X(). */
#include "celt_arm_asm.h"

        const   LSIG2RES_inv_scale, align=2
        .float  3.0517578125e-5         /* (S) 1.f / CELT_SIG_SCALE (= 1/32768) */
        endconst LSIG2RES_inv_scale


/*
 * void comb_filter_const_neon(opus_val32 *y, opus_val32 *x, int T, int N,
 *                             opus_val16 g10, opus_val16 g11, opus_val16 g12);
 *
 *   x0 = y, x1 = x, w2 = T, w3 = N (positive multiple of 4),
 *   s0 = g10, s1 = g11, s2 = g12.
 *
 * Mirrors ff_opus_postfilter_neon line-for-line. Two libopus deltas:
 *   - Gains arrive as three scalar args (s0, s1, s2) instead of a
 *     pointer to a gains[] array; the dup pattern after this is
 *     otherwise identical to FFmpeg's.
 *   - Separate input/output pointers (y in x0, x in x1) instead of
 *     FFmpeg's in-place `data` buffer in x0. With x == y this function
 *     is in-place, matching FFmpeg exactly.
 */
        function comb_filter_const_neon
        /*
         * Setup. FFmpeg:
         *     ld1 {v0.4s}, [x2]              ; load gains array
         *     sub x5, x0, w1, sxtw #2
         *     sub x1, x5, #8
         *     dup v1.4s, v0.s[1]             ; g1
         *     dup v2.4s, v0.s[2]             ; g2
         *     dup v0.4s, v0.s[0]             ; g0
         *     ld1 {v3.4s}, [x1], #16
         *     sub x4, x5, #4
         *     add x6, x5, #4
         *     fmul v3.4s, v3.4s, v2.4s
         * libopus mirrors this; the only divergence is that the gains
         * are already in s0/s1/s2 (no initial ld1), and we use x4-x7
         * for the four tap cursors so x1 stays alive as the input ptr.
         */
        sub     x4, x1, w2, sxtw #2         /* x - T          (FFmpeg: x5) */
        sub     x5, x4, #8                  /* x - T - 2      (FFmpeg: x1) */
        dup     v1.4s, v1.s[0]              /* g1 */
        dup     v2.4s, v2.s[0]              /* g2 */
        dup     v0.4s, v0.s[0]              /* g0 */

        ld1     {v3.4s}, [x5], #16
        sub     x6, x4, #4                  /* x - T - 1      (FFmpeg: x4) */
        add     x7, x4, #4                  /* x - T + 1      (FFmpeg: x6) */
        fmul    v3.4s, v3.4s, v2.4s

        /* ---- main loop — matches ff_opus_postfilter_neon line-for-line ---- */
1:      ld1     {v7.4s}, [x5], #16
        ld1     {v4.4s}, [x6], #16
        fmla    v3.4s, v7.4s, v2.4s
        ld1     {v6.4s}, [x7], #16
        ld1     {v5.4s}, [x4], #16
        fadd    v6.4s, v6.4s, v4.4s
        fmla    v3.4s, v5.4s, v0.4s

        /* LIBOPUS DELTA: load from separate input pointer x1.
           FFmpeg: ld1 {v4.4s}, [x0]    (in-place read of data) */
        ld1     {v4.4s}, [x1], #16          /* x[i..i+3] */
        fmla    v3.4s, v6.4s, v1.4s
        fadd    v4.4s, v4.4s, v3.4s
        fmul    v3.4s, v7.4s, v2.4s

        st1     {v4.4s}, [x0], #16          /* y[i..i+3] */

        subs    w3, w3, #4
        b.gt    1b
        ret
        endfunc comb_filter_const_neon


/*
 * opus_val32 celt_deemphasis_neon(opus_res *y, const opus_val32 *x,
 *                                 opus_val16 coef, opus_val32 m, int N);
 *
 *   x0 = y, x1 = x, s0 = coef, s1 = m  (libopus memory = coef * y_last),
 *   w2 = N (positive multiple of 8).
 *   Returns the new m (= coef * y_last_of_this_call) in s0.
 *
 * Mirrors ff_opus_deemphasis_neon. Deltas (see file-level comment for
 * the (M)/(S)/(W) glossary):
 *   (W)  Weights {c, c^2, c^3, c^4} are derived from `coef` here
 *        instead of being loaded from a precomputed table.
 *   (M)  Initial accumulator uses v_init = {1, c, c^2, c^3} instead of
 *        FFmpeg's v4 = {c, c^2, c^3, c^4}, so v_init * m yields the
 *        same {c·y, c²·y, c³·y, c⁴·y} that FFmpeg builds with v4*coeff.
 *   (M)  The return value is c * y_last (libopus's new m), not y_last.
 *        The loop-end prime fmul already left it in v0.s[0].
 *   (S)  SIG2RES output scaling (fmul against v24) is inserted before
 *        the store; the prime fmul is hoisted above it to keep the IIR
 *        state in the signal domain.
 */
        function celt_deemphasis_neon

        /*
         * (W) FFmpeg:
         *     ld1 {v4.4s}, [x2], #16        ; { c,  c^2, c^3, c^4 }
         *     ld1 {v5.4s}, [x2], #16        ; { 0,  c,   c^2, c^3 }
         *     ld1 {v6.4s}, [x2], #16        ; { 0,  0,   c,   c^2 }
         *     ld1 {v7.4s}, [x2]             ; { 0,  0,   0,   c   }
         */
        fmul    s17, s0, s0                 /* c^2 */
        fmul    s18, s0, s17                /* c^3 */
        fmul    s19, s17, s17               /* c^4 */
        movi    v4.16b, #0
        ins     v4.s[0], v0.s[0]
        ins     v4.s[1], v17.s[0]
        ins     v4.s[2], v18.s[0]
        ins     v4.s[3], v19.s[0]           /* v4 = { c, c^2, c^3, c^4 } */
        movi    v20.16b, #0
        ext     v5.16b, v20.16b, v4.16b, #12  /* v5 = { 0, c,   c^2, c^3 } */
        ext     v6.16b, v20.16b, v4.16b, #8   /* v6 = { 0, 0,   c,   c^2 } */
        ext     v7.16b, v20.16b, v4.16b, #4   /* v7 = { 0, 0,   0,   c   } */

        /* (S) Load 1/CELT_SIG_SCALE for the per-iter SIG2RES scaling. */
        movrel  x3, LSIG2RES_inv_scale
        ld1r    {v24.4s}, [x3]

        /*
         * (M) FFmpeg:  fmul v0.4s, v4.4s, v0.s[0]     ; v0 = v4 * coeff_in
         * libopus uses v_init = v5 with lane 0 = 1, so v0 = v_init * m
         * yields the same {c·y, c²·y, c³·y, c⁴·y}. We build v0 in two
         * steps: v5 * m fills lanes 1..3 correctly, then ins lane 0 = m.
         */
        fmul    v0.4s, v5.4s, v1.s[0]       /* { 0, c·m, c²·m, c³·m } */
        ins     v0.s[0], v1.s[0]            /* lane 0 = m */

        /* ---- main loop (BIT-IDENTICAL to ff_opus_deemphasis_neon body) ---- */
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

        /*
         * (M) FFmpeg's prime fmul lives AFTER the st1; libopus hoists
         * it above the (S) scaling so it reads v2 in the signal domain.
         * FFmpeg:                           ; (after st1)
         *     fmul v0.4s, v4.4s, v2.s[3]
         */
        fmul    v0.4s, v4.4s, v2.s[3]

        /* (S) LIBOPUS ADDITION: SIG2RES scaling. No FFmpeg equivalent. */
        fmul    v1.4s, v1.4s, v24.4s
        fmul    v2.4s, v2.4s, v24.4s

        st1     {v1.4s, v2.4s}, [x0], #32

        subs    w2, w2, #8
        b.gt    1b

        /*
         * (M) FFmpeg: mov s0, v2.s[3]   ; return y_last (= next call's coeff)
         * libopus returns c * y_last (= new m). The prime fmul above
         * already produced exactly that in v0.s[0] = s0, so we can ret
         * with no extra move — but we keep the explicit `mov` for
         * symmetry with FFmpeg.
         */
        mov     s0, v0.s[0]
        ret
        endfunc celt_deemphasis_neon


/*
 * void deemphasis_stereo_simple_neon(celt_sig *in[2], opus_res *pcm,
 *                                    int N, opus_val16 coef, celt_sig *mem);
 *
 *   x0 = in (array of two channel pointers), x1 = pcm (interleaved out),
 *   w2 = N (positive multiple of 8), s0 = coef, x3 = mem (two floats).
 *
 * (C) Literal doubling of celt_deemphasis_neon for stereo. Every
 * FFmpeg fmla/fmul/fadd has its twin here using v16/v17/v18/v19 for
 * channel B in place of FFmpeg's v0/v1/v2/v3 for channel A:
 *
 *   FFmpeg v0  (acc first half)    -> v0  (chan A) + v16 (chan B)
 *   FFmpeg v3  (acc second half)   -> v3  (chan A) + v19 (chan B)
 *   FFmpeg v1  (data first half)   -> v1  (chan A) + v17 (chan B)
 *   FFmpeg v2  (data second half)  -> v2  (chan A) + v18 (chan B)
 *
 * Weights v4..v7 and the SIG2RES scale v24 are shared. The interleaved
 * store at the end of the loop is done with vzip1/vzip2 + st1(4) since
 * the (v1, v17) / (v2, v18) pairs aren't in adjacent register slots and
 * st2 needs consecutive sources.
 *
 * Deltas (M), (S), (W) are exactly as in celt_deemphasis_neon — every
 * (C) line below is the channel-B twin of an FFmpeg-corresponding line.
 */
        function deemphasis_stereo_simple_neon
        ldp     x4, x5, [x0]                /* (C) in[0], in[1] */

        /* (W) Same as in celt_deemphasis_neon. */
        fmul    s17, s0, s0                 /* c^2 */
        fmul    s18, s0, s17                /* c^3 */
        fmul    s19, s17, s17               /* c^4 */
        movi    v4.16b, #0
        ins     v4.s[0], v0.s[0]
        ins     v4.s[1], v17.s[0]
        ins     v4.s[2], v18.s[0]
        ins     v4.s[3], v19.s[0]           /* v4 = { c, c^2, c^3, c^4 } */
        movi    v20.16b, #0
        ext     v5.16b, v20.16b, v4.16b, #12  /* v5 = { 0, c,   c^2, c^3 } */
        ext     v6.16b, v20.16b, v4.16b, #8   /* v6 = { 0, 0,   c,   c^2 } */
        ext     v7.16b, v20.16b, v4.16b, #4   /* v7 = { 0, 0,   0,   c   } */

        /* (S) SIG2RES scaling vector. */
        movrel  x6, LSIG2RES_inv_scale
        ld1r    {v24.4s}, [x6]

        /*
         * (M) Load mem and build initial accumulators.
         * FFmpeg: fmul v0.4s, v4.4s, v0.s[0]   ; v0 = v4 * coeff
         */
        ldp     s28, s29, [x3]              /* (C) mem[0], mem[1] */
        fmul    v0.4s,  v5.4s, v28.s[0]
        fmul    v16.4s, v5.4s, v29.s[0]     /* (C) */
        ins     v0.s[0],  v28.s[0]
        ins     v16.s[0], v29.s[0]          /* (C) */

        /* ---- main loop — every channel-A line mirrors FFmpeg ---- */
1:      ld1     {v1.4s,  v2.4s},  [x4], #32   /* FFmpeg: ld1 {v1, v2}, [x1], #32 */
        ld1     {v17.4s, v18.4s}, [x5], #32   /* (C) */

        fmla    v0.4s,  v5.4s, v1.s[0]        /* FFmpeg: fmla v0, v5, v1.s[0] */
        fmla    v16.4s, v5.4s, v17.s[0]       /* (C) */
        fmul    v3.4s,  v7.4s, v2.s[2]        /* FFmpeg: fmul v3, v7, v2.s[2] */
        fmul    v19.4s, v7.4s, v18.s[2]       /* (C) */

        fmla    v0.4s,  v6.4s, v1.s[1]        /* FFmpeg: fmla v0, v6, v1.s[1] */
        fmla    v16.4s, v6.4s, v17.s[1]       /* (C) */
        fmla    v3.4s,  v6.4s, v2.s[1]        /* FFmpeg: fmla v3, v6, v2.s[1] */
        fmla    v19.4s, v6.4s, v18.s[1]       /* (C) */

        fmla    v0.4s,  v7.4s, v1.s[2]        /* FFmpeg: fmla v0, v7, v1.s[2] */
        fmla    v16.4s, v7.4s, v17.s[2]       /* (C) */
        fmla    v3.4s,  v5.4s, v2.s[0]        /* FFmpeg: fmla v3, v5, v2.s[0] */
        fmla    v19.4s, v5.4s, v18.s[0]       /* (C) */

        fadd    v1.4s,  v1.4s,  v0.4s         /* FFmpeg: fadd v1, v1, v0 */
        fadd    v17.4s, v17.4s, v16.4s        /* (C) */
        fadd    v2.4s,  v2.4s,  v3.4s         /* FFmpeg: fadd v2, v2, v3 */
        fadd    v18.4s, v18.4s, v19.4s        /* (C) */

        fmla    v2.4s,  v4.4s, v1.s[3]        /* FFmpeg: fmla v2, v4, v1.s[3] */
        fmla    v18.4s, v4.4s, v17.s[3]       /* (C) */

        /* (M) Hoisted prime fmul (see celt_deemphasis_neon for why). */
        fmul    v0.4s,  v4.4s, v2.s[3]
        fmul    v16.4s, v4.4s, v18.s[3]       /* (C) */

        /* (S) SIG2RES scaling for all four output vectors. */
        fmul    v1.4s,  v1.4s,  v24.4s
        fmul    v17.4s, v17.4s, v24.4s        /* (C) */
        fmul    v2.4s,  v2.4s,  v24.4s
        fmul    v18.4s, v18.4s, v24.4s        /* (C) */

        /*
         * (C) Interleaved store. FFmpeg uses
         *     st1 {v1.4s, v2.4s}, [x0], #32
         * for its 8 contiguous samples; we need 16 samples interleaved
         * L,R,L,R,... so we zip {chanA, chanB} pairs first and stream
         * out four vectors with one st1.
         */
        zip1    v20.4s, v1.4s,  v17.4s        /* { A0, B0, A1, B1 } */
        zip2    v21.4s, v1.4s,  v17.4s        /* { A2, B2, A3, B3 } */
        zip1    v22.4s, v2.4s,  v18.4s        /* { A4, B4, A5, B5 } */
        zip2    v23.4s, v2.4s,  v18.4s        /* { A6, B6, A7, B7 } */
        st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64

        subs    w2, w2, #8                    /* FFmpeg: subs w3, w3, #8 */
        b.gt    1b

        /*
         * (M) Store updated mem (= coef * y_last). The prime fmul above
         * already produced exactly that in lane 0 of v0 / v16.
         * FFmpeg: mov s0, v2.s[3]   ; returns y_last (= next call's coeff)
         */
        stp     s0, s16, [x3]
        ret
        endfunc deemphasis_stereo_simple_neon

#endif /* __aarch64__ && !FIXED_POINT */
