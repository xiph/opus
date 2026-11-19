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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
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

#include <arm_neon.h>
#ifdef OPUS_CHECK_ASM
# include <math.h>
# include "main_FLP.h"
#endif
#include "SigProc_FLP.h"
#include "define.h"

/* NEON implementation of silk_warped_autocorrelation_FLP.
 *
 * The reference runs a serial all-pass cascade per sample (loop-carried) and
 * accumulates  C[i] += state[0] * state[i]  interleaved with that cascade.
 * The all-pass chain is inherently serial, so it stays scalar in double
 * precision -- producing the SAME state[] as the C reference bit-for-bit --
 * while the per-lag correlation accumulation is a SAXPY
 * (C[i] += input[n]*state[i], where state[0]==input[n]) that we vectorise.
 *
 * Design notes / why this layout:
 *
 *  - The accumulation is done INLINE inside the all-pass loop, consuming the
 *    freshly computed tmp1/tmp2 straight from registers.  An earlier version
 *    instead ran a separate vector pass that re-read the whole state[] array
 *    from the stack; that second traversal dominates at order 24
 *    (MAX_SHAPE_LPC_ORDER == 24) and turned the gain into a ~0.85x regression.
 *    Folding the SAXPY back in removes the second pass entirely.
 *
 *  - The all-pass advances two taps per iteration (state[i], state[i+1]), so
 *    we unroll two sections (four lags, i..i+3) per iteration: C[i..i+3] are
 *    loaded once into two float64x2 accumulators, both 2-wide fma are fused,
 *    and the pair is written back once.  This halves the C[] memory traffic
 *    versus a one-section-per-iteration loop and shrinks the per-sample branch
 *    count.  The remaining tail (one section when order % 4 == 2) uses the same
 *    2-wide step.
 *
 * Bit-exactness: every C[k] receives exactly one fused  C[k] += input[n]*state[k]
 * per sample, in the same n-order as the C reference (which compiles to scalar
 * fmadd).  f32 operands widen to f64 exactly, and the 2-way fma.2d updates C[i],
 * C[i+1] on independent lanes, so the result matches silk_warped_autocorrelation
 * _FLP_c bit-for-bit. */
void silk_warped_autocorrelation_FLP_neon(
          silk_float                *corr,                          /* O    Result [order + 1]                          */
    const silk_float                *input,                         /* I    Input data to correlate                     */
    const silk_float                warping,                        /* I    Warping coefficient                         */
    const opus_int                  length,                         /* I    Length of input                             */
    const opus_int                  order                           /* I    Correlation order (even)                    */
)
{
    opus_int     n, i;
    double       tmp1, tmp2;
    double       state[ MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };
    double       C[     MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };
    const double w = (double)warping;

    silk_assert( ( order & 1 ) == 0 );
    silk_assert( order <= MAX_SHAPE_LPC_ORDER );

    /* Loop over samples */
    for( n = 0; n < length; n++ ) {
        /* state[0] == input[n] for this sample; broadcast once. */
        const double      d     = (double)input[ n ];
        const float64x2_t sinv  = vdupq_n_f64( d );
        tmp1 = d;

        /* Two all-pass sections (four lags) per iteration. */
        for( i = 0; i + 4 <= order; i += 4 ) {
            /* Section i: compute tmp2 from state[i..i+1], store state[i]. */
            tmp2          = state[ i ] + w * state[ i + 1 ] - w * tmp1;
            state[ i ]    = tmp1;
            /* Accumulate lag i+0 (new state[i]==tmp1) and lag i+1 (tmp2),
               still holding the pre-reassignment tmp1.  C[i..i+1] stay in a
               register across the next section. */
            float64x2_t sv0 = vsetq_lane_f64( tmp2, vsetq_lane_f64( tmp1, vdupq_n_f64( 0 ), 0 ), 1 );
            float64x2_t cv0 = vld1q_f64( &C[ i ] );
            cv0 = vfmaq_f64( cv0, sv0, sinv );

            tmp1          = state[ i + 1 ] + w * state[ i + 2 ] - w * tmp2;
            state[ i + 1 ] = tmp2;

            /* Section i+2: compute tmp2 from state[i+2..i+3], store state[i+2]. */
            tmp2          = state[ i + 2 ] + w * state[ i + 3 ] - w * tmp1;
            state[ i + 2 ] = tmp1;
            float64x2_t sv1 = vsetq_lane_f64( tmp2, vsetq_lane_f64( tmp1, vdupq_n_f64( 0 ), 0 ), 1 );
            float64x2_t cv1 = vld1q_f64( &C[ i + 2 ] );
            cv1 = vfmaq_f64( cv1, sv1, sinv );

            tmp1          = state[ i + 3 ] + w * state[ i + 4 ] - w * tmp2;
            state[ i + 3 ] = tmp2;

            /* Write back both 2-lag blocks. */
            vst1q_f64( &C[ i ],     cv0 );
            vst1q_f64( &C[ i + 2 ], cv1 );
        }

        /* Remaining sections (used when order % 4 == 2). */
        for( ; i < order; i += 2 ) {
            tmp2          = state[ i ] + w * state[ i + 1 ] - w * tmp1;
            state[ i ]    = tmp1;
            float64x2_t sv = vsetq_lane_f64( tmp2, vsetq_lane_f64( tmp1, vdupq_n_f64( 0 ), 0 ), 1 );
            float64x2_t cv = vld1q_f64( &C[ i ] );
            cv = vfmaq_f64( cv, sv, sinv );
            vst1q_f64( &C[ i ], cv );
            tmp1          = state[ i + 1 ] + w * state[ i + 2 ] - w * tmp2;
            state[ i + 1 ] = tmp2;
        }

        /* Final lag (state[order] == tmp1 from the last section). */
        state[ order ] = tmp1;
        C[ order ] += d * tmp1;
    }

    /* Copy correlations in silk_float output format */
    for( i = 0; i < order + 1; i++ ) {
        corr[ i ] = (silk_float)C[ i ];
    }

#ifdef OPUS_CHECK_ASM
    /* The all-pass state is computed in double identically to the C reference,
       and each C[k] accumulates input[n]*state[k] once per sample in f64 -- so
       the result matches silk_warped_autocorrelation_FLP_c bit-for-bit. */
    {
        silk_float corr_c[ MAX_SHAPE_LPC_ORDER + 1 ];
        silk_warped_autocorrelation_FLP_c( corr_c, input, warping, length, order );
        for( i = 0; i < order + 1; i++ ) {
            celt_assert( corr[ i ] == corr_c[ i ] );
        }
    }
#endif
}
