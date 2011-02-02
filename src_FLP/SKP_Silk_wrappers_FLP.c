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

/* Wrappers. Calls flp / fix code */

/* Convert AR filter coefficients to NLSF parameters */
void SKP_Silk_A2NLSF_FLP( 
          SKP_float                 *pNLSF,             /* O    NLSF vector      [ LPC_order ]          */
    const SKP_float                 *pAR,               /* I    LPC coefficients [ LPC_order ]          */
    const SKP_int                   LPC_order           /* I    LPC order                               */
)
{
    SKP_int   i;
    SKP_int   NLSF_fix[  MAX_LPC_ORDER ];
    SKP_int32 a_fix_Q16[ MAX_LPC_ORDER ];

    for( i = 0; i < LPC_order; i++ ) {
        a_fix_Q16[ i ] = SKP_float2int( pAR[ i ] * 65536.0f );
    }
    SKP_Silk_A2NLSF( NLSF_fix, a_fix_Q16, LPC_order );

    for( i = 0; i < LPC_order; i++ ) {
        pNLSF[ i ] = ( SKP_float )NLSF_fix[ i ] * ( 1.0f / 32768.0f );
    }
}

/* Convert LSF parameters to AR prediction filter coefficients */
void SKP_Silk_NLSF2A_stable_FLP( 
          SKP_float                 *pAR,               /* O    LPC coefficients [ LPC_order ]          */
    const SKP_float                 *pNLSF,             /* I    NLSF vector      [ LPC_order ]          */
    const SKP_int                   LPC_order           /* I    LPC order                               */
)
{
    SKP_int   i;
    SKP_int   NLSF_fix[  MAX_LPC_ORDER ];
    SKP_int16 a_fix_Q12[ MAX_LPC_ORDER ];

    for( i = 0; i < LPC_order; i++ ) {
        NLSF_fix[ i ] = ( SKP_int )SKP_CHECK_FIT16( SKP_float2int( pNLSF[ i ] * 32768.0f ) );
    }

    SKP_Silk_NLSF2A_stable( a_fix_Q12, NLSF_fix, LPC_order );

    for( i = 0; i < LPC_order; i++ ) {
        pAR[ i ] = ( SKP_float )a_fix_Q12[ i ] / 4096.0f;
    }
}


/* LSF stabilizer, for a single input data vector */
void SKP_Silk_NLSF_stabilize_FLP(
          SKP_float                 *pNLSF,             /* I/O  (Un)stable NLSF vector [ LPC_order ]    */
    const SKP_int                   *pNDelta_min_Q15,   /* I    Normalized delta min vector[LPC_order+1]*/
    const SKP_int                   LPC_order           /* I    LPC order                               */
)
{
    SKP_int   i;
    SKP_int   NLSF_Q15[ MAX_LPC_ORDER ], ndelta_min_Q15[ MAX_LPC_ORDER + 1 ];

    for( i = 0; i < LPC_order; i++ ) {
        NLSF_Q15[       i ] = ( SKP_int )SKP_float2int( pNLSF[ i ] * 32768.0f );
        ndelta_min_Q15[ i ] = ( SKP_int )SKP_float2int( pNDelta_min_Q15[ i ] );
    }
    ndelta_min_Q15[ LPC_order ] = ( SKP_int )SKP_float2int( pNDelta_min_Q15[ LPC_order ] );

    /* NLSF stabilizer, for a single input data vector */
    SKP_Silk_NLSF_stabilize( NLSF_Q15, ndelta_min_Q15, LPC_order );

    for( i = 0; i < LPC_order; i++ ) {
        pNLSF[ i ] = ( SKP_float )NLSF_Q15[ i ] * ( 1.0f / 32768.0f );
    }
}

/* Interpolation function with fixed point rounding */
void SKP_Silk_interpolate_wrapper_FLP(
          SKP_float                 xi[],               /* O    Interpolated vector                     */
    const SKP_float                 x0[],               /* I    First vector                            */
    const SKP_float                 x1[],               /* I    Second vector                           */
    const SKP_float                 ifact,              /* I    Interp. factor, weight on second vector */
    const SKP_int                   d                   /* I    Number of parameters                    */
)
{
    SKP_int x0_int[ MAX_LPC_ORDER ], x1_int[ MAX_LPC_ORDER ], xi_int[ MAX_LPC_ORDER ];
    SKP_int ifact_Q2 = ( SKP_int )( ifact * 4.0f );
    SKP_int i;

    /* Convert input from flp to fix */
    for( i = 0; i < d; i++ ) {
        x0_int[ i ] = SKP_float2int( x0[ i ] * 32768.0f );
        x1_int[ i ] = SKP_float2int( x1[ i ] * 32768.0f );
    }

    /* Interpolate two vectors */
    SKP_Silk_interpolate( xi_int, x0_int, x1_int, ifact_Q2, d );
    
    /* Convert output from fix to flp */
    for( i = 0; i < d; i++ ) {
        xi[ i ] = ( SKP_float )xi_int[ i ] * ( 1.0f / 32768.0f );
    }
}

/****************************************/
/* Floating-point Silk VAD wrapper      */
/****************************************/
SKP_int SKP_Silk_VAD_FLP(
    SKP_Silk_encoder_state_FLP      *psEnc,             /* I/O  Encoder state FLP                       */
    SKP_Silk_encoder_control_FLP    *psEncCtrl,         /* I/O  Encoder control FLP                     */
    const SKP_int16                 *pIn                /* I    Input signal                            */
)
{
    SKP_int i, ret, SA_Q8, SNR_dB_Q7, Tilt_Q15;
    SKP_int Quality_Bands_Q15[ VAD_N_BANDS ];

    ret = SKP_Silk_VAD_GetSA_Q8( &psEnc->sCmn.sVAD, &SA_Q8, &SNR_dB_Q7, Quality_Bands_Q15, &Tilt_Q15,
        pIn, psEnc->sCmn.frame_length, psEnc->sCmn.fs_kHz );

    psEnc->speech_activity = ( SKP_float )SA_Q8 / 256.0f;
    for( i = 0; i < VAD_N_BANDS; i++ ) {
        psEncCtrl->input_quality_bands[ i ] = ( SKP_float )Quality_Bands_Q15[ i ] / 32768.0f;
    }
    psEncCtrl->input_tilt = ( SKP_float )Tilt_Q15 / 32768.0f;

    return ret;
}

/****************************************/
/* Floating-point Silk NSQ wrapper      */
/****************************************/
void SKP_Silk_NSQ_wrapper_FLP(
    SKP_Silk_encoder_state_FLP      *psEnc,         /* I/O  Encoder state FLP                           */
    SKP_Silk_encoder_control_FLP    *psEncCtrl,     /* I/O  Encoder control FLP                         */
    const SKP_float                 x[],            /* I    Prefiltered input signal                    */
          SKP_int8                  q[],            /* O    Quantized pulse signal                      */
    const SKP_int                   useLBRR         /* I    LBRR flag                                   */
)
{
    SKP_int     i, j;
    SKP_float   tmp_float;
    SKP_int16   x_16[ MAX_FRAME_LENGTH ];
    SKP_int32   Gains_Q16[ MAX_NB_SUBFR ];
    SKP_DWORD_ALIGN SKP_int16 PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ];
    SKP_int16   LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ];
    SKP_int     LTP_scale_Q14;

    /* Noise shaping parameters */
    /* Testing */
    SKP_int16   AR2_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    SKP_int32   LF_shp_Q14[ MAX_NB_SUBFR ];         /* Packs two int16 coefficients per int32 value             */
    SKP_int     Lambda_Q10;
    SKP_int     Tilt_Q14[ MAX_NB_SUBFR ];
    SKP_int     HarmShapeGain_Q14[ MAX_NB_SUBFR ];

    /* Convert control struct to fix control struct */
    /* Noise shape parameters */
    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        for( j = 0; j < psEnc->sCmn.shapingLPCOrder; j++ ) {
            AR2_Q13[ i * MAX_SHAPE_LPC_ORDER + j ] = SKP_float2int( psEncCtrl->AR2[ i * MAX_SHAPE_LPC_ORDER + j ] * 8192.0f );
        }
    }

    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        LF_shp_Q14[ i ] =   SKP_LSHIFT32( SKP_float2int( psEncCtrl->LF_AR_shp[ i ]     * 16384.0f ), 16 ) |
                              (SKP_uint16)SKP_float2int( psEncCtrl->LF_MA_shp[ i ]     * 16384.0f );
        Tilt_Q14[ i ]   =        (SKP_int)SKP_float2int( psEncCtrl->Tilt[ i ]          * 16384.0f );
        HarmShapeGain_Q14[ i ] = (SKP_int)SKP_float2int( psEncCtrl->HarmShapeGain[ i ] * 16384.0f );    
    }
    Lambda_Q10 = ( SKP_int )SKP_float2int( psEncCtrl->Lambda * 1024.0f );

    /* prediction and coding parameters */
    for( i = 0; i < psEnc->sCmn.nb_subfr * LTP_ORDER; i++ ) {
        LTPCoef_Q14[ i ] = ( SKP_int16 )SKP_float2int( psEncCtrl->LTPCoef[ i ] * 16384.0f );
    }

    for( j = 0; j < 2; j++ ) {
        for( i = 0; i < psEnc->sCmn.predictLPCOrder; i++ ) {
            PredCoef_Q12[ j ][ i ] = ( SKP_int16 )SKP_float2int( psEncCtrl->PredCoef[ j ][ i ] * 4096.0f );
        }
    }

    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        tmp_float = SKP_LIMIT( ( psEncCtrl->Gains[ i ] * 65536.0f ), 2147483000.0f, -2147483000.0f );
        Gains_Q16[ i ] = SKP_float2int( tmp_float );
        if( psEncCtrl->Gains[ i ] > 0.0f ) {
            SKP_assert( tmp_float >= 0.0f );
            SKP_assert( Gains_Q16[ i ] >= 0 );
        }
    }

    if( psEncCtrl->sCmn.signalType == TYPE_VOICED ) {
        LTP_scale_Q14 = SKP_Silk_LTPScales_table_Q14[ psEncCtrl->sCmn.LTP_scaleIndex ];
    } else {
        LTP_scale_Q14 = 0;
    }

    /* Convert input to fix */
    SKP_float2short_array( x_16, x, psEnc->sCmn.frame_length );

    /* Call NSQ */
    if( useLBRR ) {
        if( psEnc->sCmn.nStatesDelayedDecision > 1 || psEnc->sCmn.warping_Q16 > 0 ) {
            SKP_Silk_NSQ_del_dec( &psEnc->sCmn, &psEncCtrl->sCmn, &psEnc->sNSQ_LBRR, 
                x_16, q, psEncCtrl->sCmn.NLSFInterpCoef_Q2, PredCoef_Q12[ 0 ], LTPCoef_Q14, AR2_Q13, 
                HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, Lambda_Q10, LTP_scale_Q14 );
        } else {
            SKP_Silk_NSQ( &psEnc->sCmn, &psEncCtrl->sCmn, &psEnc->sNSQ_LBRR, 
                x_16, q, psEncCtrl->sCmn.NLSFInterpCoef_Q2, PredCoef_Q12[ 0 ], LTPCoef_Q14, AR2_Q13, 
                HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, Lambda_Q10, LTP_scale_Q14 );
        }
    } else {
        if( psEnc->sCmn.nStatesDelayedDecision > 1 || psEnc->sCmn.warping_Q16 > 0 ) {
            SKP_Silk_NSQ_del_dec( &psEnc->sCmn, &psEncCtrl->sCmn, &psEnc->sNSQ, 
                x_16, q, psEncCtrl->sCmn.NLSFInterpCoef_Q2, PredCoef_Q12[ 0 ], LTPCoef_Q14, AR2_Q13, 
                HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, Lambda_Q10, LTP_scale_Q14 );
        } else {
            SKP_Silk_NSQ( &psEnc->sCmn, &psEncCtrl->sCmn, &psEnc->sNSQ, 
                x_16, q, psEncCtrl->sCmn.NLSFInterpCoef_Q2, PredCoef_Q12[ 0 ], LTPCoef_Q14, AR2_Q13, 
                HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, Lambda_Q10, LTP_scale_Q14 );
        }
    }
}

/***********************************************/
/* Floating-point Silk LTP quantiation wrapper */
/***********************************************/
void SKP_Silk_quant_LTP_gains_FLP(
          SKP_float B[ MAX_NB_SUBFR * LTP_ORDER ],              /* I/O  (Un-)quantized LTP gains                */
          SKP_int   cbk_index[ MAX_NB_SUBFR ],                  /* O    Codebook index                          */
          SKP_int   *periodicity_index,                         /* O    Periodicity index                       */
    const SKP_float W[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ],  /* I    Error weights                           */
    const SKP_int   mu_Q10,                                     /* I    Mu value (R/D tradeoff)     */
    const SKP_int   lowComplexity,                              /* I    Flag for low complexity                 */
    const SKP_int   nb_subfr                                    /* I    number of subframes                     */
)
{
    SKP_int   i;
    SKP_int16 B_Q14[ MAX_NB_SUBFR * LTP_ORDER ];
    SKP_int32 W_Q18[ MAX_NB_SUBFR*LTP_ORDER*LTP_ORDER ];

    for( i = 0; i < nb_subfr * LTP_ORDER; i++ ) {
        B_Q14[ i ] = (SKP_int16)SKP_float2int( B[ i ] * 16384.0f );
    }
    for( i = 0; i < nb_subfr * LTP_ORDER * LTP_ORDER; i++ ) {
        W_Q18[ i ] = (SKP_int32)SKP_float2int( W[ i ] * 262144.0f );
    }

    SKP_Silk_quant_LTP_gains( B_Q14, cbk_index, periodicity_index, W_Q18, mu_Q10, lowComplexity, nb_subfr );

    for( i = 0; i < nb_subfr * LTP_ORDER; i++ ) {
        B[ i ] = ( (SKP_float)B_Q14[ i ] ) / 16384.0f;
    }
}
