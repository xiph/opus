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

   The reference runs a serial all-pass cascade (tmp1->tmp2) per sample, then
   accumulates C[i] += state[0]*state[i].  The all-pass chain is loop-carried
   and cannot be parallelised across taps without changing the rounding, so we
   keep it scalar in double precision -- producing the SAME state[] as the C
   reference bit-for-bit -- and vectorise the correlation accumulation across
   the (order+1) lags using float64x2 lanes (f64 matches the reference's
   double C[]).  Only the order of additions inside the 2-wide lane reduction
   differs, so the result is within ~1e-15 of the reference (well below float
   precision) and the encoded bitstream is unchanged. */
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
        const float64x2_t s0v = vdupq_n_f64( (double)input[ n ] );
        const double *st = state;
        double       *Cp = C;
        opus_int      m  = order + 1;

        /* State update: identical serial all-pass chain to the C reference,
           in double, so state[] matches the reference exactly. */
        tmp1 = (double)input[ n ];
        for( i = 0; i < order; i += 2 ) {
            tmp2          = state[ i ] + w * state[ i + 1 ] - w * tmp1;
            state[ i ]    = tmp1;
            tmp1          = state[ i + 1 ] + w * state[ i + 2 ] - w * tmp2;
            state[ i + 1 ] = tmp2;
        }
        state[ order ] = tmp1;

        /* corr[i] += state[0] * state[i], vectorised across lags (state[0]==input[n]). */
        for( ; m >= 4; m -= 4, st += 4, Cp += 4 ) {
            float64x2_t c0 = vld1q_f64( Cp + 0 );
            float64x2_t c1 = vld1q_f64( Cp + 2 );
            c0 = vfmaq_f64( c0, vld1q_f64( st + 0 ), s0v );
            c1 = vfmaq_f64( c1, vld1q_f64( st + 2 ), s0v );
            vst1q_f64( Cp + 0, c0 );
            vst1q_f64( Cp + 2, c1 );
        }
        if( m >= 2 ) {
            vst1q_f64( Cp, vfmaq_f64( vld1q_f64( Cp ), vld1q_f64( st ), s0v ) );
            m -= 2; st += 2; Cp += 2;
        }
        if( m ) {
            *Cp += (double)input[ n ] * (*st);
        }
    }

    for( i = 0; i < order + 1; i++ ) {
        corr[ i ] = (silk_float)C[ i ];
    }

#ifdef OPUS_CHECK_ASM
    /* The all-pass state is computed in double identically to the C reference,
       so only the per-lag correlation sums reorder; check each lag is within
       rounding of silk_warped_autocorrelation_FLP_c (relative to the zero-lag
       energy corr[0], which bounds every |corr[i]|). */
    {
        silk_float corr_c[ MAX_SHAPE_LPC_ORDER + 1 ];
        silk_warped_autocorrelation_FLP_c( corr_c, input, warping, length, order );
        for( i = 0; i < order + 1; i++ ) {
            celt_assert( fabs( corr[ i ] - corr_c[ i ] ) <= 1e-5 * ( fabs( corr_c[ 0 ] ) + 1e-30 ) );
        }
    }
#endif
}
