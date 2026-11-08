/***********************************************************************
Copyright (c) 2026, Xiph.Org Foundation. All rights reserved.
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

#include <arm_neon.h>
#include "SigProc_FIX.h"
#include "resampler_private.h"

/* See the SSE version for a description; vmull_s16()/vmlal_s16() accumulate in
   32 bits exactly as the C code does, so the result is bit-exact. */
opus_int16 *silk_resampler_private_IIR_FIR_INTERPOL_neon(
    opus_int16  *out,
    opus_int16  *buf,
    opus_int32  max_index_Q16,
    opus_int32  index_increment_Q16
)
{
    opus_int32 index_Q16, res_Q15;
    opus_int16 *buf_ptr;
    opus_int32 table_index;
    int16x8_t coefs[ 12 ];
    opus_int t;

    for( t = 0; t < 12; t++ ) {
        const opus_int16 *cA = silk_resampler_frac_FIR_12[ t ];
        const opus_int16 *cB = silk_resampler_frac_FIR_12[ 11 - t ];
        opus_int16 tmp[ 8 ];
        tmp[ 0 ] = cA[ 0 ]; tmp[ 1 ] = cA[ 1 ]; tmp[ 2 ] = cA[ 2 ]; tmp[ 3 ] = cA[ 3 ];
        tmp[ 4 ] = cB[ 3 ]; tmp[ 5 ] = cB[ 2 ]; tmp[ 6 ] = cB[ 1 ]; tmp[ 7 ] = cB[ 0 ];
        coefs[ t ] = vld1q_s16( tmp );
    }

    /* Interpolate upsampled signal and store in output array */
    for( index_Q16 = 0; index_Q16 < max_index_Q16; index_Q16 += index_increment_Q16 ) {
        int16x8_t x, c;
        int32x4_t p;
        int32x2_t s;
        table_index = silk_SMULWB( index_Q16 & 0xFFFF, 12 );
        buf_ptr = &buf[ index_Q16 >> 16 ];

        x = vld1q_s16( buf_ptr );
        c = coefs[ table_index ];
        p = vmull_s16( vget_low_s16( x ), vget_low_s16( c ) );
        p = vmlal_s16( p, vget_high_s16( x ), vget_high_s16( c ) );
        s = vadd_s32( vget_low_s32( p ), vget_high_s32( p ) );
        res_Q15 = vget_lane_s32( vpadd_s32( s, s ), 0 );
#ifdef OPUS_CHECK_ASM
        {
            opus_int32 res_Q15_c;
            res_Q15_c = silk_SMULBB(            buf_ptr[ 0 ], silk_resampler_frac_FIR_12[      table_index ][ 0 ] );
            res_Q15_c = silk_SMLABB( res_Q15_c, buf_ptr[ 1 ], silk_resampler_frac_FIR_12[      table_index ][ 1 ] );
            res_Q15_c = silk_SMLABB( res_Q15_c, buf_ptr[ 2 ], silk_resampler_frac_FIR_12[      table_index ][ 2 ] );
            res_Q15_c = silk_SMLABB( res_Q15_c, buf_ptr[ 3 ], silk_resampler_frac_FIR_12[      table_index ][ 3 ] );
            res_Q15_c = silk_SMLABB( res_Q15_c, buf_ptr[ 4 ], silk_resampler_frac_FIR_12[ 11 - table_index ][ 3 ] );
            res_Q15_c = silk_SMLABB( res_Q15_c, buf_ptr[ 5 ], silk_resampler_frac_FIR_12[ 11 - table_index ][ 2 ] );
            res_Q15_c = silk_SMLABB( res_Q15_c, buf_ptr[ 6 ], silk_resampler_frac_FIR_12[ 11 - table_index ][ 1 ] );
            res_Q15_c = silk_SMLABB( res_Q15_c, buf_ptr[ 7 ], silk_resampler_frac_FIR_12[ 11 - table_index ][ 0 ] );
            celt_assert( res_Q15 == res_Q15_c );
        }
#endif
        *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q15, 15 ) );
    }
    return out;
}
