/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, (subject to the limitations in the disclaimer below)
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Skype Limited, nor the names of specific
contributors, may be used to endorse or promote products derived from
this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED
BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "SigProc_FIX.h"
#include "resampler_private.h"

static inline opus_int16 *silk_resampler_private_down_FIR_INTERPOL0(
    opus_int16          *out, 
    opus_int32          *buf2, 
    const opus_int16    *FIR_Coefs, 
    opus_int32          max_index_Q16, 
    opus_int32          index_increment_Q16
)
{
    opus_int32 index_Q16, res_Q6;
    opus_int32 *buf_ptr;

    for( index_Q16 = 0; index_Q16 < max_index_Q16; index_Q16 += index_increment_Q16 ) {
        /* Integer part gives pointer to buffered input */
        buf_ptr = buf2 + silk_RSHIFT( index_Q16, 16 );

        /* Inner product */
        res_Q6 = silk_SMULWB(         silk_ADD32( buf_ptr[ 0 ], buf_ptr[ 15 ] ), FIR_Coefs[ 0 ] );
        res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 1 ], buf_ptr[ 14 ] ), FIR_Coefs[ 1 ] );
        res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 2 ], buf_ptr[ 13 ] ), FIR_Coefs[ 2 ] );
        res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 3 ], buf_ptr[ 12 ] ), FIR_Coefs[ 3 ] );
        res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 4 ], buf_ptr[ 11 ] ), FIR_Coefs[ 4 ] );
        res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 5 ], buf_ptr[ 10 ] ), FIR_Coefs[ 5 ] );
        res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 6 ], buf_ptr[  9 ] ), FIR_Coefs[ 6 ] );
        res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 7 ], buf_ptr[  8 ] ), FIR_Coefs[ 7 ] );

        /* Scale down, saturate and store in output array */
        *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q6, 6 ) );
    }
    return out;
}

static inline opus_int16 *silk_resampler_private_down_FIR_INTERPOL1(
    opus_int16          *out, 
    opus_int32          *buf2, 
    const opus_int16    *FIR_Coefs, 
    opus_int32          max_index_Q16, 
    opus_int32          index_increment_Q16, 
    opus_int32          FIR_Fracs
)
{
    opus_int32 index_Q16, res_Q6;
    opus_int32 *buf_ptr;
    opus_int32 interpol_ind;
    const opus_int16 *interpol_ptr;

    for( index_Q16 = 0; index_Q16 < max_index_Q16; index_Q16 += index_increment_Q16 ) {
        /* Integer part gives pointer to buffered input */
        buf_ptr = buf2 + silk_RSHIFT( index_Q16, 16 );

        /* Fractional part gives interpolation coefficients */
        interpol_ind = silk_SMULWB( index_Q16 & 0xFFFF, FIR_Fracs );

        /* Inner product */
        interpol_ptr = &FIR_Coefs[ RESAMPLER_DOWN_ORDER_FIR / 2 * interpol_ind ];
        res_Q6 = silk_SMULWB(         buf_ptr[ 0 ], interpol_ptr[ 0 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 1 ], interpol_ptr[ 1 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 2 ], interpol_ptr[ 2 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 3 ], interpol_ptr[ 3 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 4 ], interpol_ptr[ 4 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 5 ], interpol_ptr[ 5 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 6 ], interpol_ptr[ 6 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 7 ], interpol_ptr[ 7 ] );
        interpol_ptr = &FIR_Coefs[ RESAMPLER_DOWN_ORDER_FIR / 2 * ( FIR_Fracs - 1 - interpol_ind ) ];
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 15 ], interpol_ptr[ 0 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 14 ], interpol_ptr[ 1 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 13 ], interpol_ptr[ 2 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 12 ], interpol_ptr[ 3 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 11 ], interpol_ptr[ 4 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 10 ], interpol_ptr[ 5 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[  9 ], interpol_ptr[ 6 ] );
        res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[  8 ], interpol_ptr[ 7 ] );

        /* Scale down, saturate and store in output array */
        *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q6, 6 ) );
    }
    return out;
}


/* Resample with a 2x downsampler (optional), a 2nd order AR filter followed by FIR interpolation */
void silk_resampler_private_down_FIR(
    void                            *SS,            /* I/O  Resampler state             */
    opus_int16                      out[],          /* O    Output signal               */
    const opus_int16                in[],           /* I    Input signal                */
    opus_int32                      inLen           /* I    Number of input samples     */
)
{
    silk_resampler_state_struct *S = (silk_resampler_state_struct *)SS;
    opus_int32 nSamplesIn;
    opus_int32 max_index_Q16, index_increment_Q16;
    opus_int16 buf1[ RESAMPLER_MAX_BATCH_SIZE_IN / 2 ];
    opus_int32 buf2[ RESAMPLER_MAX_BATCH_SIZE_IN + RESAMPLER_DOWN_ORDER_FIR ];
    const opus_int16 *FIR_Coefs;

    /* Copy buffered samples to start of buffer */
    silk_memcpy( buf2, S->sFIR, RESAMPLER_DOWN_ORDER_FIR * sizeof( opus_int32 ) );

    FIR_Coefs = &S->Coefs[ 2 ];

    /* Iterate over blocks of frameSizeIn input samples */
    index_increment_Q16 = S->invRatio_Q16;
    while( 1 ) {
        nSamplesIn = silk_min( inLen, S->batchSize );

        if( S->input2x == 1 ) {
            /* Downsample 2x */
            silk_resampler_down2( S->sDown2, buf1, in, nSamplesIn );

            nSamplesIn = silk_RSHIFT32( nSamplesIn, 1 );

            /* Second-order AR filter (output in Q8) */
            silk_resampler_private_AR2( S->sIIR, &buf2[ RESAMPLER_DOWN_ORDER_FIR ], buf1, S->Coefs, nSamplesIn );
        } else {
            /* Second-order AR filter (output in Q8) */
            silk_resampler_private_AR2( S->sIIR, &buf2[ RESAMPLER_DOWN_ORDER_FIR ], in, S->Coefs, nSamplesIn );
        }

        max_index_Q16 = silk_LSHIFT32( nSamplesIn, 16 );

        /* Interpolate filtered signal */
        if( S->FIR_Fracs == 1 ) {
            out = silk_resampler_private_down_FIR_INTERPOL0(out, buf2, FIR_Coefs, max_index_Q16, index_increment_Q16);
        } else {
            out = silk_resampler_private_down_FIR_INTERPOL1(out, buf2, FIR_Coefs, max_index_Q16, index_increment_Q16, S->FIR_Fracs);
        }

        in += nSamplesIn << S->input2x;
        inLen -= nSamplesIn << S->input2x;

        if( inLen > S->input2x ) {
            /* More iterations to do; copy last part of filtered signal to beginning of buffer */
            silk_memcpy( buf2, &buf2[ nSamplesIn ], RESAMPLER_DOWN_ORDER_FIR * sizeof( opus_int32 ) );
        } else {
            break;
        }
    }

    /* Copy last part of filtered signal to the state for the next call */
    silk_memcpy( S->sFIR, &buf2[ nSamplesIn ], RESAMPLER_DOWN_ORDER_FIR * sizeof( opus_int32 ) );
}

