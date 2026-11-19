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

/* NEON implementation of silk_inner_product_FLP.  The C reference accumulates
   the products in double precision; we widen each float operand to double
   before multiplying (matching data1[i]*(double)data2[i]) and accumulate in
   two float64x2 lanes, exactly as the x86 AVX2 implementation does.  The
   result is within rounding of the C reference (well below float precision). */
double silk_inner_product_FLP_neon(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
)
{
    opus_int    i;
    float64x2_t acc0 = vdupq_n_f64( 0.0 );
    float64x2_t acc1 = vdupq_n_f64( 0.0 );
    double      result;

    for( i = 0; i + 4 <= dataSize; i += 4 ) {
        float32x4_t a = vld1q_f32( &data1[ i ] );
        float32x4_t b = vld1q_f32( &data2[ i ] );
        acc0 = vfmaq_f64( acc0, vcvt_f64_f32( vget_low_f32(  a ) ), vcvt_f64_f32( vget_low_f32(  b ) ) );
        acc1 = vfmaq_f64( acc1, vcvt_f64_f32( vget_high_f32( a ) ), vcvt_f64_f32( vget_high_f32( b ) ) );
    }
    result = vaddvq_f64( vaddq_f64( acc0, acc1 ) );

    for( ; i < dataSize; i++ ) {
        result += data1[ i ] * (double)data2[ i ];
    }

#ifdef OPUS_CHECK_ASM
    /* Float NEON kernels accumulate in double like the C reference but in a
       different summation order, so (unlike fixed-point kernels) they are not
       bit-exact. The rounding error of an f64 dot product is bounded by
       ~N*eps*sum|terms|, so we check against the accumulation magnitude rather
       than the result (which can be near zero from cancellation). */
    {
        double result_c = silk_inner_product_FLP_c( data1, data2, dataSize );
        double sum_abs = 0;
        for( i = 0; i < dataSize; i++ ) sum_abs += fabs( data1[ i ] * (double)data2[ i ] );
        celt_assert( fabs( result - result_c ) <= 1e-6 * sum_abs + 1e-30 );
    }
#endif
    return result;
}
