/***********************************************************************
Copyright (c) 2026 LPC NEON Pilot
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include "SigProc_FIX.h"
#include <arm_neon.h>

/* NEON optimized silk_LPC_analysis_filter.
 *
 * Same algorithmic idea (even/odd accumulator pair)
 * with widening multiply-accumulate, on 128-bit NEON vectors, processing
 * 8 output samples per iteration (4 even + 4 odd int32 accumulators).
 * The inner loop is vectorized over the filter order d.
 *
 * Machine arithmetic matches the C reference bit-exactly:
 *   - vmlal_s16        == silk_SMLABB_ovflw (wrapping widen mul-accumulate)
 *   - vsubq_s32        == silk_SUB32_ovflw   (wrapping 32-bit subtract)
 *   - ((x>>11)+1)>>1   == silk_RSHIFT_ROUND(x,12)
 *   - min/max clamp    == silk_SAT16
 *
 * A scalar tail handles the remainder when (len-d) is not a multiple of 8,
 * reusing the exact C arithmetic. Bit-exact with silk_LPC_analysis_filter_c
 * for all valid inputs (d even, d >= 6, d <= len).
 */
void silk_LPC_analysis_filter_neon(
    opus_int16                  *out,               /* O    Output signal                                               */
    const opus_int16            *in,                /* I    Input signal                                                */
    const opus_int16            *B,                 /* I    MA prediction coefficients, Q12 [order]                     */
    const opus_int32            len,                /* I    Signal length                                               */
    const opus_int32            d,                  /* I    Filter order                                                */
    int                         arch                /* I    Run-time architecture                                       */
)
{
    int j, i0, lim;

    (void)arch;

    /* Set first d output samples to zero (same as C reference) */
    silk_memset( out, 0, (size_t)d * sizeof( opus_int16 ) );

    /* lim is the largest index <= len such that (len - lim) is a multiple of 8.
     * The main NEON loop processes [d, lim) in blocks of 8 outputs. */
    lim = len - ((len - d) & 7);
    if (lim < d) lim = d;

    i0 = d;
    while (i0 < lim) {
        int32x4_t acc_e = vdupq_n_s32(0);   /* even outputs i0+0,2,4,6 */
        int32x4_t acc_o = vdupq_n_s32(0);   /* odd  outputs i0+1,3,5,7 */

        for (j = 0; j < d; j++) {
            /* 8 consecutive input samples starting at in[i0-1-j].
             * x8[m] = in[i0-1-j+m]; even outputs (i0+2k) use x8[2k],
             * odd outputs (i0+2k+1) use x8[2k+1]. */
            int16x8_t x8 = vld1q_s16( &in[ i0 - 1 - j ] );
            int16x4_t e4 = vget_low_s16( vuzp1q_s16( x8, x8 ) );  /* x8[0,2,4,6] */
            int16x4_t o4 = vget_low_s16( vuzp2q_s16( x8, x8 ) );  /* x8[1,3,5,7] */
            int16x4_t b4 = vdup_n_s16( B[j] );
            acc_e = vmlal_s16( acc_e, e4, b4 );   /* acc over even outputs */
            acc_o = vmlal_s16( acc_o, o4, b4 );   /* acc over odd  outputs */
        }

        /* in[i0 .. i0+7] */
        int16x8_t inv8 = vld1q_s16( &in[i0] );
        int16x4_t inv_e = vget_low_s16( vuzp1q_s16( inv8, inv8 ) );
        int16x4_t inv_o = vget_low_s16( vuzp2q_s16( inv8, inv8 ) );

        /* (in << 12) - dot */
        int32x4_t res_e = vsubq_s32( vshll_n_s16( inv_e, 12 ), acc_e );
        int32x4_t res_o = vsubq_s32( vshll_n_s16( inv_o, 12 ), acc_o );

        /* RSHIFT_ROUND(.,12): ((x>>11)+1)>>1 */
        res_e = vshrq_n_s32( vaddq_s32( vshrq_n_s32( res_e, 11 ), vdupq_n_s32( 1 ) ), 1 );
        res_o = vshrq_n_s32( vaddq_s32( vshrq_n_s32( res_o, 11 ), vdupq_n_s32( 1 ) ), 1 );

        /* SAT16 */
        res_e = vminq_s32( vmaxq_s32( res_e, vdupq_n_s32( -32768 ) ), vdupq_n_s32( 32767 ) );
        res_o = vminq_s32( vmaxq_s32( res_o, vdupq_n_s32( -32768 ) ), vdupq_n_s32( 32767 ) );

        /* narrow + interleave back to out[i0 .. i0+7] */
        int16x4_t oe = vqmovn_s32( res_e );
        int16x4_t oo = vqmovn_s32( res_o );
        int16x4_t lo = vzip1_s16( oe, oo );
        int16x4_t hi = vzip2_s16( oe, oo );
        vst1q_s16( &out[i0], vcombine_s16( lo, hi ) );

        i0 += 8;
    }

    /* Scalar tail for the remaining outputs [lim, len) */
    if (lim < len) {
        int ix;
        opus_int32 out32_Q12;
        for (ix = lim; ix < len; ix++) {
            const opus_int16 *in_ptr = &in[ ix - 1 ];

            out32_Q12 = silk_SMULBB( in_ptr[ 0 ], B[ 0 ] );
            for (j = 1; j < d; j++) {
                out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -j ], B[ j ] );
            }
            out32_Q12 = silk_SUB32_ovflw( silk_LSHIFT( (opus_int32)in_ptr[ 1 ], 12 ), out32_Q12 );
            out32_Q12 = silk_RSHIFT_ROUND( out32_Q12, 12 );
            out[ ix ] = (opus_int16)silk_SAT16( out32_Q12 );
        }
    }
}
