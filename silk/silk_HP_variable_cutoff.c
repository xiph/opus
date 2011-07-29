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
#ifdef FIXED_POINT
#include "silk_main_FIX.h"
#else
#include "silk_main_FLP.h"
#endif
#include "silk_tuning_parameters.h"

/* High-pass filter with cutoff frequency adaptation based on pitch lag statistics */
void silk_HP_variable_cutoff(
    silk_encoder_state_Fxx          state_Fxx[],        /* I/O  Encoder states                          */
    const opus_int                   nChannels           /* I    Number of channels                      */
)
{
    opus_int   quality_Q15, cutoff_Hz;
    opus_int32 B_Q28[ 3 ], A_Q28[ 2 ];
    opus_int32 Fc_Q19, r_Q28, r_Q22;
    opus_int32 pitch_freq_Hz_Q16, pitch_freq_log_Q7, delta_freq_Q7;
    silk_encoder_state *psEncC1 = &state_Fxx[ 0 ].sCmn;

    if( psEncC1->HP_cutoff_Hz == 0 ) {
        /* Adaptive cutoff frequency: estimate low end of pitch frequency range */
        if( psEncC1->prevSignalType == TYPE_VOICED ) {
            /* difference, in log domain */
            pitch_freq_Hz_Q16 = SKP_DIV32_16( SKP_LSHIFT( SKP_MUL( psEncC1->fs_kHz, 1000 ), 16 ), psEncC1->prevLag );
            pitch_freq_log_Q7 = silk_lin2log( pitch_freq_Hz_Q16 ) - ( 16 << 7 );

            /* adjustment based on quality */
            quality_Q15 = psEncC1->input_quality_bands_Q15[ 0 ];
            pitch_freq_log_Q7 = SKP_SMLAWB( pitch_freq_log_Q7, SKP_SMULWB( SKP_LSHIFT( -quality_Q15, 2 ), quality_Q15 ), 
                pitch_freq_log_Q7 - ( silk_lin2log( SILK_FIX_CONST( VARIABLE_HP_MIN_CUTOFF_HZ, 16 ) ) - ( 16 << 7 ) ) );

            /* delta_freq = pitch_freq_log - psEnc->variable_HP_smth1; */
            delta_freq_Q7 = pitch_freq_log_Q7 - SKP_RSHIFT( psEncC1->variable_HP_smth1_Q15, 8 );
            if( delta_freq_Q7 < 0 ) {
                /* less smoothing for decreasing pitch frequency, to track something close to the minimum */
                delta_freq_Q7 = SKP_MUL( delta_freq_Q7, 3 );
            }

            /* limit delta, to reduce impact of outliers in pitch estimation */
            delta_freq_Q7 = SKP_LIMIT_32( delta_freq_Q7, -SILK_FIX_CONST( VARIABLE_HP_MAX_DELTA_FREQ, 7 ), SILK_FIX_CONST( VARIABLE_HP_MAX_DELTA_FREQ, 7 ) );

            /* update smoother */
            psEncC1->variable_HP_smth1_Q15 = SKP_SMLAWB( psEncC1->variable_HP_smth1_Q15, 
                SKP_SMULBB( psEncC1->speech_activity_Q8, delta_freq_Q7 ), SILK_FIX_CONST( VARIABLE_HP_SMTH_COEF1, 16 ) );

            /* limit frequency range */
            psEncC1->variable_HP_smth1_Q15 = SKP_LIMIT_32( psEncC1->variable_HP_smth1_Q15, 
                SKP_LSHIFT( silk_lin2log( VARIABLE_HP_MIN_CUTOFF_HZ ), 8 ), 
                SKP_LSHIFT( silk_lin2log( VARIABLE_HP_MAX_CUTOFF_HZ ), 8 ) );
        }
    } else {
        /* Externally-controlled cutoff frequency */
        cutoff_Hz = SKP_LIMIT( psEncC1->HP_cutoff_Hz, 10, 500 );
        psEncC1->variable_HP_smth1_Q15 = SKP_LSHIFT( silk_lin2log( cutoff_Hz ), 8 );
    }

    /* second smoother */
    psEncC1->variable_HP_smth2_Q15 = SKP_SMLAWB( psEncC1->variable_HP_smth2_Q15, 
        psEncC1->variable_HP_smth1_Q15 - psEncC1->variable_HP_smth2_Q15, SILK_FIX_CONST( VARIABLE_HP_SMTH_COEF2, 16 ) );

    /* convert from log scale to Hertz */
    cutoff_Hz = silk_log2lin( SKP_RSHIFT( psEncC1->variable_HP_smth2_Q15, 8 ) );

    /********************************/
    /* Compute Filter Coefficients  */
    /********************************/
    /* compute cut-off frequency, in radians    */
    /* Fc_num   = 1.5 * 3.14159 * cutoff_Hz     */
    /* Fc_denom = 1e3f * psEncC1->fs_kHz        */
    SKP_assert( cutoff_Hz <= SKP_int32_MAX / SILK_FIX_CONST( 1.5 * 3.14159 / 1000, 19 ) );
    Fc_Q19 = SKP_DIV32_16( SKP_SMULBB( SILK_FIX_CONST( 1.5 * 3.14159 / 1000, 19 ), cutoff_Hz ), psEncC1->fs_kHz );
    SKP_assert( Fc_Q19 > 0 && Fc_Q19 < 32768 );

    r_Q28 = SILK_FIX_CONST( 1.0, 28 ) - SKP_MUL( SILK_FIX_CONST( 0.92, 9 ), Fc_Q19 );

    /* b = r * [ 1; -2; 1 ]; */
    /* a = [ 1; -2 * r * ( 1 - 0.5 * Fc^2 ); r^2 ]; */
    B_Q28[ 0 ] = r_Q28;
    B_Q28[ 1 ] = SKP_LSHIFT( -r_Q28, 1 );
    B_Q28[ 2 ] = r_Q28;
    
    /* -r * ( 2 - Fc * Fc ); */
    r_Q22  = SKP_RSHIFT( r_Q28, 6 );
    A_Q28[ 0 ] = SKP_SMULWW( r_Q22, SKP_SMULWW( Fc_Q19, Fc_Q19 ) - SILK_FIX_CONST( 2.0,  22 ) );
    A_Q28[ 1 ] = SKP_SMULWW( r_Q22, r_Q22 );

    /********************************/
    /* High-Pass Filter             */
    /********************************/
    silk_biquad_alt( psEncC1->inputBuf, B_Q28, A_Q28, psEncC1->In_HP_State, psEncC1->inputBuf, psEncC1->frame_length );
    if( nChannels == 2 ) {
        silk_biquad_alt( state_Fxx[ 1 ].sCmn.inputBuf, B_Q28, A_Q28, state_Fxx[ 1 ].sCmn.In_HP_State, 
            state_Fxx[ 1 ].sCmn.inputBuf, state_Fxx[ 1 ].sCmn.frame_length );
    }
}
