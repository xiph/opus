/***********************************************************************
Copyright (c) 2006-2010, Skype Limited. All rights reserved. 
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

#include "SKP_Silk_main_FLP.h"

#if HIGH_PASS_INPUT

/* High-pass filter with cutoff frequency adaptation based on pitch lag statistics */
void SKP_Silk_HP_variable_cutoff_FLP(
    SKP_Silk_encoder_state_FLP      *psEnc,             /* I/O  Encoder state FLP                       */
    SKP_Silk_encoder_control_FLP    *psEncCtrl,         /* I/O  Encoder control FLP                     */
          SKP_int16                 *out,               /* O    High-pass filtered output signal        */
    const SKP_int16                 *in                 /* I    Input signal                            */
)
{
    SKP_float pitch_freq_Hz, pitch_freq_log, quality, delta_freq, smth_coef, Fc, r;
    SKP_int32 B_Q28[ 3 ], A_Q28[ 2 ];

    /*********************************************/
    /* Estimate low end of pitch frequency range */
    /*********************************************/
    if( psEnc->sCmn.prev_sigtype == SIG_TYPE_VOICED ) {

        /* Difference, in log domain */
        pitch_freq_Hz  = 1e3f * psEnc->sCmn.fs_kHz / psEnc->sCmn.prevLag;
        pitch_freq_log = SKP_Silk_log2( pitch_freq_Hz );

        /* Adjustment based on quality */
        quality = psEncCtrl->input_quality_bands[ 0 ];
        pitch_freq_log -= quality * quality * ( pitch_freq_log - SKP_Silk_log2( VARIABLE_HP_MIN_FREQ ) );
        pitch_freq_log += 0.5f * ( 0.6f - quality );

        delta_freq = pitch_freq_log - psEnc->variable_HP_smth1;
        if( delta_freq < 0.0 ) {
            /* Less smoothing for decreasing pitch frequency, to track something close to the minimum */
            delta_freq *= 3.0f;
        }

        /* Limit delta, to reduce impact of outliers */
        delta_freq = SKP_LIMIT_float( delta_freq, -VARIABLE_HP_MAX_DELTA_FREQ, VARIABLE_HP_MAX_DELTA_FREQ );
    
        /* Update smoother */
        smth_coef = VARIABLE_HP_SMTH_COEF1 * psEnc->speech_activity;
        psEnc->variable_HP_smth1 += smth_coef * delta_freq;
    }

    /* Second smoother */
    psEnc->variable_HP_smth2 += VARIABLE_HP_SMTH_COEF2 * ( psEnc->variable_HP_smth1 - psEnc->variable_HP_smth2 );

    /* Convert from log scale to Hertz */
    psEncCtrl->pitch_freq_low_Hz = ( SKP_float )pow( 2.0, psEnc->variable_HP_smth2 );

    /* Limit frequency range */
    psEncCtrl->pitch_freq_low_Hz = SKP_LIMIT_float( psEncCtrl->pitch_freq_low_Hz, VARIABLE_HP_MIN_FREQ, VARIABLE_HP_MAX_FREQ );

    /*******************************/
    /* Compute filter coefficients */
    /*******************************/
    /* Compute cut-off frequency, in radians */
    Fc = ( SKP_float )( 0.45f * 2.0f * 3.14159265359 * psEncCtrl->pitch_freq_low_Hz / ( 1e3f * psEnc->sCmn.fs_kHz ) );

    /* 2nd order ARMA coefficients */
    r = 1.0f - 0.92f * Fc;

    /* b = r * [1; -2; 1]; */
    /* a = [1; -2 * r * (1 - 0.5 * Fc^2); r^2]; */
    B_Q28[ 0 ] = SKP_float2int( ( 1 << 28 ) * r );
    B_Q28[ 1 ] = SKP_float2int( ( 1 << 28 ) * -2.0f * r );
    B_Q28[ 2 ] = B_Q28[ 0 ];
    A_Q28[ 0 ] = SKP_float2int( ( 1 << 28 ) * -2.0f * r * ( 1.0f - 0.5f * Fc * Fc ) );
    A_Q28[ 1 ] = SKP_float2int( ( 1 << 28 ) * r * r );

    /********************/
    /* High-pass filter */
    /********************/
    SKP_Silk_biquad_alt( in, B_Q28, A_Q28, psEnc->sCmn.In_HP_State, out, psEnc->sCmn.frame_length );
}

#endif // HIGH_PASS_INPUT
