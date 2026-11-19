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
#endif
#include "SigProc_FLP.h"

/* NEON implementation of silk_energy_FLP (sum of squares).  Like the C
   reference (and the inner_product kernel) it accumulates in double: each
   f32 is widened to f64 before squaring and summed in two float64x2 lanes,
   so the result is within rounding of the scalar reference. */
double silk_energy_FLP_neon(
    const silk_float    *data,
    opus_int            dataSize
)
{
    opus_int    i;
    float64x2_t acc0 = vdupq_n_f64( 0.0 );
    float64x2_t acc1 = vdupq_n_f64( 0.0 );
    double      result;

    for( i = 0; i + 4 <= dataSize; i += 4 ) {
        float32x4_t x  = vld1q_f32( &data[ i ] );
        float64x2_t lo = vcvt_f64_f32( vget_low_f32(  x ) );
        float64x2_t hi = vcvt_f64_f32( vget_high_f32( x ) );
        acc0 = vfmaq_f64( acc0, lo, lo );
        acc1 = vfmaq_f64( acc1, hi, hi );
    }
    result = vaddvq_f64( vaddq_f64( acc0, acc1 ) );

    for( ; i < dataSize; i++ ) {
        result += data[ i ] * (double)data[ i ];
    }

#ifdef OPUS_CHECK_ASM
    /* Float NEON kernel: f64 accumulation in a different order than the C
       reference, so within-rounding rather than bit-exact (cf. fixed-point). */
    {
        double result_c = silk_energy_FLP_c( data, dataSize );
        celt_assert( fabs( result - result_c ) <= 1e-5 * ( fabs( result_c ) + 1e-30 ) );
    }
#endif
    return result;
}
