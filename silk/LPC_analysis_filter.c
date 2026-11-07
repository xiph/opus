/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
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
#include "config.h"
#endif

#include "SigProc_FIX.h"
#include "celt_lpc.h"
#include <arm_neon.h>
#include "stack_alloc.h"

/*******************************************/
/* LPC analysis filter                     */
/* NB! State is kept internally and the    */
/* filter always starts with zero state    */
/* first d output samples are set to zero  */
/*******************************************/

/* OPT: Using celt_fir() for this function should be faster, but it may cause
   integer overflows in intermediate values (not final results), which the
   current implementation silences by casting to unsigned. Enabling
   this should be safe in pretty much all cases, even though it is not technically
   C89-compliant. */
#define USE_CELT_FIR 0

void silk_LPC_analysis_filter_c(
    opus_int16                  *out,               /* O    Output signal                                               */
    const opus_int16            *in,                /* I    Input signal                                                */
    const opus_int16            *B,                 /* I    MA prediction coefficients, Q12 [order]                     */
    const opus_int32            len,                /* I    Signal length                                               */
    const opus_int32            d,                  /* I    Filter order                                                */
    int                         arch                /* I    Run-time architecture                                       */
)
{
    opus_int   j;
#if defined(FIXED_POINT) && USE_CELT_FIR
    opus_int16 num[SILK_MAX_ORDER_LPC];
#else
    int ix;
    opus_int32       out32_Q12, out32;
    const opus_int16 *in_ptr;
#endif

    celt_assert( d >= 6 );
    celt_assert( (d & 1) == 0 );
    celt_assert( d <= len );

#if defined(FIXED_POINT) && USE_CELT_FIR
    celt_assert( d <= SILK_MAX_ORDER_LPC );
    for ( j = 0; j < d; j++ ) {
        num[ j ] = -B[ j ];
    }
    celt_fir( in + d, num, out + d, len - d, d, arch );
    for ( j = 0; j < d; j++ ) {
        out[ j ] = 0;
    }
#else
    (void)arch;
    for( ix = d; ix < len; ix++ ) {
        in_ptr = &in[ ix - 1 ];

        out32_Q12 = silk_SMULBB( in_ptr[  0 ], B[ 0 ] );
        /* Allowing wrap around so that two wraps can cancel each other. The rare
           cases where the result wraps around can only be triggered by invalid streams*/
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -1 ], B[ 1 ] );
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -2 ], B[ 2 ] );
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -3 ], B[ 3 ] );
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -4 ], B[ 4 ] );
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -5 ], B[ 5 ] );
        for( j = 6; j < d; j += 2 ) {
            out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -j     ], B[ j     ] );
            out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -j - 1 ], B[ j + 1 ] );
        }

        /* Subtract prediction */
        out32_Q12 = silk_SUB32_ovflw( silk_LSHIFT( (opus_int32)in_ptr[ 1 ], 12 ), out32_Q12 );

        /* Scale to Q0 */
        out32 = silk_RSHIFT_ROUND( out32_Q12, 12 );

        /* Saturate output */
        out[ ix ] = (opus_int16)silk_SAT16( out32 );
    }

    /* Set first d output samples to zero */
    silk_memset( out, 0, d * sizeof( opus_int16 ) );
#endif
}

/* NEON optimized LPC analysis filter - processes 8 outputs at a time
 * Computes: out[i] = in[i] - sum(B[j] * in[i-d+j], j=0..d-1)
 * where B is in Q12 format
 */
void silk_LPC_analysis_filter(
    opus_int16                  *out,               /* O    Output signal                                               */
    const opus_int16            *in,                /* I    Input signal                                                */
    const opus_int16            *B,                 /* I    MA prediction coefficients, Q12 [order]                     */
    const opus_int32            len,                /* I    Signal length                                               */
    const opus_int32            d,                  /* I    Filter order                                                */
    int                         arch                /* I    Run-time architecture                                       */
)
{
    int ix, j;
    (void)arch;

    celt_assert(d >= 6);
    celt_assert((d & 1) == 0);
    celt_assert(d <= len);

    for(ix = d; ix < len; ix++) {
        const opus_int16 *in_ptr = &in[ix - 1];
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);

        /* Process coefficients int groups of 8 */
        for(j = 0; j < (d & ~7); j += 8) {
            int16x4_t b_vec0 = vld1_s16(&B[j]);
            int16x4_t b_vec1 = vld1_s16(&B[j + 4]);
            int16x4_t in_vec0 = vld1_s16(&in_ptr[-j-3]);
            int16x4_t in_vec1 = vld1_s16(&in_ptr[-j-7]);
            in_vec0 = vrev64_s16(in_vec0);
            in_vec1 = vrev64_s16(in_vec1);
            acc0 = vmlal_s16(acc0, b_vec0, in_vec0);
            acc1 = vmlal_s16(acc1, b_vec1, in_vec1);
        }

        acc0 = vaddq_s32(acc0, acc1);
        int32x2_t sum = vpadd_s32(vget_low_s32(acc0), vget_high_s32(acc0));
        opus_int32 out32_Q12 = vget_lane_s32(vpadd_s32(sum, sum), 0);

        /* Handle remaining coefficients */
        for(; j < d; j++) {
            out32_Q12 = silk_SMLABB_ovflw(out32_Q12, in_ptr[-j], B[j]);
        }

        /* Subtract predicton */
        out32_Q12 = silk_SUB32_ovflw(silk_LSHIFT((opus_int32)in_ptr[1], 12), out32_Q12);

        /* Scale to Q0 and saturate */
        opus_int32 out32 = silk_RSHIFT_ROUND(out32_Q12, 12);
        out[ix] = (opus_int16)silk_SAT16(out32);
    }

    silk_memset(out, 0, d * sizeof(opus_int16));

#ifdef OPUS_CHECK_ASM
    VARDECL( opus_int16, out_c );
    ALLOC( out_c, len, opus_int16 );
    silk_LPC_analysis_filter_c( out_c, in, B, len, d, arch );
    silk_assert( !memcmp( out, out_c, len * sizeof(opus_int16) ) );
#endif

}
