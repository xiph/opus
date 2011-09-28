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
 *                                        Fs_out (kHz)
 *                        8      12     16     24     32     44.1   48
 *
 *               8        C      UF     U      UF     UF     UF     UF
 *              12        AF     C      UF     U      UF     UF     UF
 *              16        D      AF     C      UF     U      UF     UF
 * Fs_in (kHz)  24        AIF    D      AF     C      UF     UF     U
 *              32        UF     AF     D      AF     C      UF     UF
 *              44.1      AMI    AMI    AMI    AMI    AMI    C      UF
 *              48        DAF    DAF    AF     D      AF     UF     C
 *
 * default method: UF
 *
 * C   -> Copy (no resampling)
 * D   -> Allpass-based 2x downsampling
 * U   -> Allpass-based 2x upsampling
 * DAF -> Allpass-based 2x downsampling followed by AR2 filter followed by FIR interpolation
 * UF  -> Allpass-based 2x upsampling followed by FIR interpolation
 * AMI -> ARMA4 filter followed by FIR interpolation
 * AF  -> AR2 filter followed by FIR interpolation
 *
 * Input signals sampled above 48 kHz are first downsampled to at most 48 kHz.
 * Output signals sampled above 48 kHz are upsampled from at most 48 kHz.
 */

#include "resampler_private.h"

/* Greatest common divisor */
static opus_int32 gcd(
    opus_int32 a,
    opus_int32 b
)
{
    opus_int32 tmp;
    while( b > 0 ) {
        tmp = a - b * silk_DIV32( a, b );
        a   = b;
        b   = tmp;
    }
    return a;
}

#define USE_silk_resampler_copy (0)
#define USE_silk_resampler_private_up2_HQ_wrapper (1)
#define USE_silk_resampler_private_IIR_FIR (2)
#define USE_silk_resampler_private_down_FIR (3)

/* Initialize/reset the resampler state for a given pair of input/output sampling rates */
opus_int silk_resampler_init(
    silk_resampler_state_struct    *S,                    /* I/O: Resampler state             */
    opus_int32                            Fs_Hz_in,    /* I:    Input sampling rate (Hz)    */
    opus_int32                            Fs_Hz_out    /* I:    Output sampling rate (Hz)    */
)
{
    opus_int32 cycleLen, cyclesPerBatch, up2 = 0, down2 = 0;

    /* Clear state */
    silk_memset( S, 0, sizeof( silk_resampler_state_struct ) );

    /* Input checking */
    if( Fs_Hz_in < 8000 || Fs_Hz_in >  48000 || Fs_Hz_out < 8000 || Fs_Hz_out >  48000 ) {
        silk_assert( 0 );
        return -1;
    }

    /* Number of samples processed per batch */
    /* First, try 10 ms frames */
    S->batchSize = silk_DIV32_16( Fs_Hz_in, 100 );
    if( ( silk_MUL( S->batchSize, 100 ) != Fs_Hz_in ) || ( Fs_Hz_in % 100 != 0 ) ) {
        /* No integer number of input or output samples with 10 ms frames, use greatest common divisor */
        cycleLen = silk_DIV32( Fs_Hz_in, gcd( Fs_Hz_in, Fs_Hz_out ) );
        cyclesPerBatch = silk_DIV32( RESAMPLER_MAX_BATCH_SIZE_IN, cycleLen );
        if( cyclesPerBatch == 0 ) {
            /* cycleLen too big, let's just use the maximum batch size. Some distortion will result. */
            S->batchSize = RESAMPLER_MAX_BATCH_SIZE_IN;
            silk_assert( 0 );
        } else {
            S->batchSize = silk_MUL( cyclesPerBatch, cycleLen );
        }
    }


    /* Find resampler with the right sampling ratio */
    if( Fs_Hz_out > Fs_Hz_in ) {
        /* Upsample */
        if( Fs_Hz_out == silk_MUL( Fs_Hz_in, 2 ) ) {                             /* Fs_out : Fs_in = 2 : 1 */
            /* Special case: directly use 2x upsampler */
            S->resampler_function = USE_silk_resampler_private_up2_HQ_wrapper;
        } else {
            /* Default resampler */
            S->resampler_function = USE_silk_resampler_private_IIR_FIR;
            up2 = 1;
            S->up2_hq = Fs_Hz_in <= 24000;
        }
    } else if ( Fs_Hz_out < Fs_Hz_in ) {
        /* Downsample */
        if( silk_MUL( Fs_Hz_out, 4 ) == silk_MUL( Fs_Hz_in, 3 ) ) {               /* Fs_out : Fs_in = 3 : 4 */
            S->FIR_Fracs = 3;
            S->Coefs = silk_Resampler_3_4_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 3 ) == silk_MUL( Fs_Hz_in, 2 ) ) {        /* Fs_out : Fs_in = 2 : 3 */
            S->FIR_Fracs = 2;
            S->Coefs = silk_Resampler_2_3_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 2 ) == Fs_Hz_in ) {                      /* Fs_out : Fs_in = 1 : 2 */
            S->FIR_Fracs = 1;
            S->Coefs = silk_Resampler_1_2_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 8 ) == silk_MUL( Fs_Hz_in, 3 ) ) {        /* Fs_out : Fs_in = 3 : 8 */
            S->FIR_Fracs = 3;
            S->Coefs = silk_Resampler_3_8_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 3 ) == Fs_Hz_in ) {                      /* Fs_out : Fs_in = 1 : 3 */
            S->FIR_Fracs = 1;
            S->Coefs = silk_Resampler_1_3_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 4 ) == Fs_Hz_in ) {                      /* Fs_out : Fs_in = 1 : 4 */
            S->FIR_Fracs = 1;
            down2 = 1;
            S->Coefs = silk_Resampler_1_2_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 6 ) == Fs_Hz_in ) {                      /* Fs_out : Fs_in = 1 : 6 */
            S->FIR_Fracs = 1;
            down2 = 1;
            S->Coefs = silk_Resampler_1_3_COEFS;
            S->resampler_function = USE_silk_resampler_private_down_FIR;
        } else if( silk_MUL( Fs_Hz_out, 441 ) == silk_MUL( Fs_Hz_in, 80 ) ) {     /* Fs_out : Fs_in = 80 : 441 */
            S->Coefs = silk_Resampler_80_441_ARMA4_COEFS;
            S->resampler_function = USE_silk_resampler_private_IIR_FIR;
        } else if( silk_MUL( Fs_Hz_out, 441 ) == silk_MUL( Fs_Hz_in, 120 ) ) {    /* Fs_out : Fs_in = 120 : 441 */
            S->Coefs = silk_Resampler_120_441_ARMA4_COEFS;
            S->resampler_function = USE_silk_resampler_private_IIR_FIR;
        } else if( silk_MUL( Fs_Hz_out, 441 ) == silk_MUL( Fs_Hz_in, 160 ) ) {    /* Fs_out : Fs_in = 160 : 441 */
            S->Coefs = silk_Resampler_160_441_ARMA4_COEFS;
            S->resampler_function = USE_silk_resampler_private_IIR_FIR;
        } else if( silk_MUL( Fs_Hz_out, 441 ) == silk_MUL( Fs_Hz_in, 240 ) ) {    /* Fs_out : Fs_in = 240 : 441 */
            S->Coefs = silk_Resampler_240_441_ARMA4_COEFS;
            S->resampler_function = USE_silk_resampler_private_IIR_FIR;
        } else if( silk_MUL( Fs_Hz_out, 441 ) == silk_MUL( Fs_Hz_in, 320 ) ) {    /* Fs_out : Fs_in = 320 : 441 */
            S->Coefs = silk_Resampler_320_441_ARMA4_COEFS;
            S->resampler_function = USE_silk_resampler_private_IIR_FIR;
        } else {
            /* Default resampler */
            S->resampler_function = USE_silk_resampler_private_IIR_FIR;
            up2 = 1;
            S->up2_hq = Fs_Hz_in <= 24000;
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

    S->magic_number = 123456789;

    return 0;
}

/* Clear the states of all resampling filters, without resetting sampling rate ratio */
opus_int silk_resampler_clear(
    silk_resampler_state_struct    *S            /* I/O: Resampler state             */
)
{
    /* Clear state */
    silk_memset( S->sDown2, 0, sizeof( S->sDown2 ) );
    silk_memset( S->sIIR,   0, sizeof( S->sIIR ) );
    silk_memset( S->sFIR,   0, sizeof( S->sFIR ) );
    return 0;
}

/* Resampler: convert from one sampling rate to another                                 */
opus_int silk_resampler(
    silk_resampler_state_struct    *S,                    /* I/O: Resampler state             */
    opus_int16                            out[],        /* O:    Output signal                 */
    const opus_int16                        in[],        /* I:    Input signal                */
    opus_int32                            inLen        /* I:    Number of input samples        */
)
{
    /* Verify that state was initialized and has not been corrupted */
    if( S->magic_number != 123456789 ) {
        silk_assert( 0 );
        return -1;
    }

    /* Input and output sampling rate are at most 48000 Hz */
    switch(S->resampler_function) {
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
