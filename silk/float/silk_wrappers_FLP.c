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

#include "silk_main_FLP.h"

/* Wrappers. Calls flp / fix code */

/* Convert AR filter coefficients to NLSF parameters */
void silk_A2NLSF_FLP( 
          SKP_int16                 *NLSF_Q15,          /* O    NLSF vector      [ LPC_order ]          */
    const SKP_float                 *pAR,               /* I    LPC coefficients [ LPC_order ]          */
    const SKP_int                   LPC_order           /* I    LPC order                               */
)
{
    SKP_int   i;
    SKP_int32 a_fix_Q16[ MAX_LPC_ORDER ];

    for( i = 0; i < LPC_order; i++ ) {
        a_fix_Q16[ i ] = SKP_float2int( pAR[ i ] * 65536.0f );
    }

    silk_A2NLSF( NLSF_Q15, a_fix_Q16, LPC_order );
}

/* Convert LSF parameters to AR prediction filter coefficients */
void silk_NLSF2A_stable_FLP( 
          SKP_float                 *pAR,               /* O    LPC coefficients [ LPC_order ]          */
    const SKP_int16                 *NLSF_Q15,          /* I    NLSF vector      [ LPC_order ]          */
    const SKP_int                   LPC_order           /* I    LPC order                               */
)
{
    SKP_int   i;
    SKP_int16 a_fix_Q12[ MAX_LPC_ORDER ];

    silk_NLSF2A_stable( a_fix_Q12, NLSF_Q15, LPC_order );

    for( i = 0; i < LPC_order; i++ ) {
        pAR[ i ] = ( SKP_float )a_fix_Q12[ i ] * ( 1.0f / 4096.0f );
    }
}

/******************************************/
/* Floating-point NLSF processing wrapper */
/******************************************/
void silk_process_NLSFs_FLP(
    silk_encoder_state              *psEncC,                            /* I/O  Encoder state                               */
    SKP_float                       PredCoef[ 2 ][ MAX_LPC_ORDER ],     /* O    Prediction coefficients                     */
    SKP_int16                       NLSF_Q15[      MAX_LPC_ORDER ],     /* I/O  Normalized LSFs (quant out) (0 - (2^15-1))  */
    const SKP_int16                 prev_NLSF_Q15[ MAX_LPC_ORDER ]      /* I    Previous Normalized LSFs (0 - (2^15-1))     */
)
{
    SKP_int     i, j;
    SKP_int16   PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ];

    silk_process_NLSFs( psEncC, PredCoef_Q12, NLSF_Q15, prev_NLSF_Q15);

    for( j = 0; j < 2; j++ ) {
        for( i = 0; i < psEncC->predictLPCOrder; i++ ) {
            PredCoef[ j ][ i ] = ( SKP_float )PredCoef_Q12[ j ][ i ] * ( 1.0f / 4096.0f );
        }
    }
}

/****************************************/
/* Floating-point Silk NSQ wrapper      */
/****************************************/
void silk_NSQ_wrapper_FLP(
    silk_encoder_state_FLP          *psEnc,         /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,     /* I/O  Encoder control FLP                         */
    SideInfoIndices                 *psIndices,     /* I/O  Quantization indices                        */
    silk_nsq_state                  *psNSQ,         /* I/O  Noise Shaping Quantzation state             */
          SKP_int8                  pulses[],       /* O    Quantized pulse signal                      */
    const SKP_float                 x[]             /* I    Prefiltered input signal                    */
)
{
    SKP_int     i, j;
    SKP_int16   x_16[ MAX_FRAME_LENGTH ];
    SKP_int32   Gains_Q16[ MAX_NB_SUBFR ];
    SKP_DWORD_ALIGN SKP_int16 PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ];
    SKP_int16   LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ];
    SKP_int     LTP_scale_Q14;

    /* Noise shaping parameters */
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
        Gains_Q16[ i ] = SKP_float2int( psEncCtrl->Gains[ i ] * 65536.0f );
        SKP_assert( Gains_Q16[ i ] > 0 );
    }

    if( psIndices->signalType == TYPE_VOICED ) {
        LTP_scale_Q14 = silk_LTPScales_table_Q14[ psIndices->LTP_scaleIndex ];
    } else {
        LTP_scale_Q14 = 0;
    }

    /* Convert input to fix */
    SKP_float2short_array( x_16, x, psEnc->sCmn.frame_length );

    /* Call NSQ */
    if( psEnc->sCmn.nStatesDelayedDecision > 1 || psEnc->sCmn.warping_Q16 > 0 ) {
        silk_NSQ_del_dec( &psEnc->sCmn, psNSQ, psIndices, x_16, pulses, PredCoef_Q12[ 0 ], LTPCoef_Q14, 
            AR2_Q13, HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, psEncCtrl->pitchL, Lambda_Q10, LTP_scale_Q14 );
    } else {
        silk_NSQ( &psEnc->sCmn, psNSQ, psIndices, x_16, pulses, PredCoef_Q12[ 0 ], LTPCoef_Q14, 
            AR2_Q13, HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, psEncCtrl->pitchL, Lambda_Q10, LTP_scale_Q14 );
    }
}

/***********************************************/
/* Floating-point Silk LTP quantiation wrapper */
/***********************************************/
void silk_quant_LTP_gains_FLP(
          SKP_float B[ MAX_NB_SUBFR * LTP_ORDER ],              /* I/O  (Un-)quantized LTP gains                */
          SKP_int8  cbk_index[ MAX_NB_SUBFR ],                  /* O    Codebook index                          */
          SKP_int8  *periodicity_index,                         /* O    Periodicity index                       */
    const SKP_float W[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ],  /* I    Error weights                           */
    const SKP_int   mu_Q10,                                     /* I    Mu value (R/D tradeoff)                 */
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

    silk_quant_LTP_gains( B_Q14, cbk_index, periodicity_index, W_Q18, mu_Q10, lowComplexity, nb_subfr );

    for( i = 0; i < nb_subfr * LTP_ORDER; i++ ) {
        B[ i ] = (SKP_float)B_Q14[ i ] * ( 1.0f / 16384.0f );
    }
}
