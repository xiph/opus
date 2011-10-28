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

/* Matrix of resampling methods used:
 *                                 Fs_out (kHz)
 *                        8      12     16     24    48
 *
 *               8        C      UF     U      UF     UF
 *              12        AF     C      UF     U      UF
 * Fs_in (kHz)  16        D      AF     C      UF     UF
 *              24        AIF    D      AF     C      U
 *              48        DAF    DAF    AF     D      C
 *
 * default method: UF
 *
 * C   -> Copy (no resampling)
 * D   -> Allpass-based 2x downsampling
 * U   -> Allpass-based 2x upsampling
 * DAF -> Allpass-based 2x downsampling followed by AR2 filter followed by FIR interpolation
 * UF  -> Allpass-based 2x upsampling followed by FIR interpolation
 * AF  -> AR2 filter followed by FIR interpolation
 *
 * Signals sampled above 48 kHz are not supported.
 */

#include "resampler_private.h"

#define USE_silk_resampler_copy                     (0)
#define USE_silk_resampler_private_up2_HQ_wrapper   (1)
#define USE_silk_resampler_private_IIR_FIR          (2)
#define USE_silk_resampler_private_down_FIR         (3)

/* Initialize/reset the resampler state for a given pair of input/output sampling rates */
opus_int silk_resampler_init(
    silk_resampler_state_struct *S,                 /* I/O   Resampler state                                            */
    opus_int32                  Fs_Hz_in,           /* I     Input sampling rate (Hz)                                   */
    opus_int32                  Fs_Hz_out           /* I     Output sampling rate (Hz)                                  */
)
{
    opus_int32 up2 = 0, down2 = 0;

    /* Clear state */
    silk_memset( S, 0, sizeof( silk_resampler_state_struct ) );

    /* Input checking */
    if( ( Fs_Hz_in  != 8000 && Fs_Hz_in  != 12000 && Fs_Hz_in  != 16000 && Fs_Hz_in  != 24000 && Fs_Hz_in  != 48000 ) ||
        ( Fs_Hz_out != 8000 && Fs_Hz_out != 12000 && Fs_Hz_out != 16000 && Fs_Hz_out != 24000 && Fs_Hz_out != 48000 ) ) {
        silk_assert( 0 );
        return -1;
    }

    /* Number of samples processed per batch */
    S->batchSize = silk_DIV32_16( Fs_Hz_in, 100 );

    /* Find resampler with the right sampling ratio */
    if( Fs_Hz_out > Fs_Hz_in ) {
        /* Upsample */
        if( Fs_Hz_out == silk_MUL( Fs_Hz_in, 2 ) ) {                            /* Fs_out : Fs_in = 2 : 1 */
            /* Special case: directly use 2x upsampler */
            S->resampler_function = USE_silk_resampler_private_up2_HQ_wrapper;
        } else {
            /* Default resampler */
            S->resampler_function = USE_silk_resampler_private_IIR_FIR;
            up2 = 1;
        }
    } else if ( Fs_Hz_out < Fs_Hz_in ) {
        /* Downsample */
        if( silk_MUL( Fs_Hz_out, 4 ) == silk_MUL( Fs_Hz_in, 3 ) ) {             /* Fs_out : Fs_in = 3 : 4 */
            S->FIR_Fracs = 3;
            S->Coefs = silk_Resampler_3_4_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 3 ) == silk_MUL( Fs_Hz_in, 2 ) ) {      /* Fs_out : Fs_in = 2 : 3 */
            S->FIR_Fracs = 2;
            S->Coefs = silk_Resampler_2_3_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 2 ) == Fs_Hz_in ) {                     /* Fs_out : Fs_in = 1 : 2 */
            S->FIR_Fracs = 1;
            S->Coefs = silk_Resampler_1_2_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 3 ) == Fs_Hz_in ) {                     /* Fs_out : Fs_in = 1 : 3 */
            S->FIR_Fracs = 1;
            S->Coefs = silk_Resampler_1_3_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 4 ) == Fs_Hz_in ) {                     /* Fs_out : Fs_in = 1 : 4 */
            S->FIR_Fracs = 1;
            down2 = 1;
            S->Coefs = silk_Resampler_1_2_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 6 ) == Fs_Hz_in ) {                     /* Fs_out : Fs_in = 1 : 6 */
            S->FIR_Fracs = 1;
            down2 = 1;
            S->Coefs = silk_Resampler_1_3_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else {
            /* None available */
            silk_assert( 0 );
            return -1;
        }
    } else {
        /* Input and output sampling rates are equal: copy */
        S->resampler_function = USE_silk_resampler_copy;
    }

    S->input2x = up2 | down2;

    /* Ratio of input/output samples */
    S->invRatio_Q16 = silk_LSHIFT32( silk_DIV32( silk_LSHIFT32( Fs_Hz_in, 14 + up2 - down2 ), Fs_Hz_out ), 2 );
    /* Make sure the ratio is rounded up */
    while( silk_SMULWW( S->invRatio_Q16, silk_LSHIFT32( Fs_Hz_out, down2 ) ) < silk_LSHIFT32( Fs_Hz_in, up2 ) ) {
        S->invRatio_Q16++;
    }

    return 0;
}

/* Resampler: convert from one sampling rate to another */
opus_int silk_resampler(
    silk_resampler_state_struct *S,                 /* I/O   Resampler state                                            */
    opus_int16                  out[],              /* O     Output signal                                              */
    const opus_int16            in[],               /* I     Input signal                                               */
    opus_int32                  inLen               /* I     Number of input samples                                    */
)
{
    /* Input and output sampling rate are at most 48000 Hz */
    switch( S->resampler_function ) {
        case USE_silk_resampler_private_up2_HQ_wrapper:
            silk_resampler_private_up2_HQ_wrapper( S, out, in, inLen );
            break;
        case USE_silk_resampler_private_IIR_FIR:
            silk_resampler_private_IIR_FIR( S, out, in, inLen );
            break;
        case USE_silk_resampler_private_down_FIR:
            silk_resampler_private_down_FIR( S, out, in, inLen );
            break;
        default:
            silk_memcpy( out, in, inLen * sizeof( opus_int16 ) );
    }
    return 0;
}
