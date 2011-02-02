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

#include "SKP_Silk_main_FIX.h"

#ifdef SAVE_ALL_INTERNAL_DATA
#include <math.h>

void SKP_Silk_LTP_ana_core(
    SKP_float                       r_LPC[],        /* I    LPC residual            */
    SKP_float                       r_LTP[],        /* O    LTP residual            */
    const SKP_int                   pitchL[],       /* I    pitch lags              */
    const SKP_float                 LTPCoef[],      /* I    LTP Coeficients         */
    SKP_int                         subfr_length,   /* I    smpls in one sub frame  */
    SKP_int                         LTP_mem_length  /* I    Length of LTP state of input */
);

void SKP_Silk_LPC_analysis_filter_FLP(
    SKP_float                       r_LPC[],        /* O    LPC residual signal             */
    const SKP_float                 PredCoef[],     /* I    LPC coeficicnts                 */
    const SKP_float                 s[],            /* I    Input Signal                    */
    SKP_int                         length,         /* I    length of signal                */
    SKP_int                         Order           /* I    LPC order                       */
);

double SKP_Silk_energy_FLP( 
    const SKP_float     *data, 
    SKP_int             dataSize
);

/* integer to floating-point conversion */
SKP_INLINE void SKP_short2float_array(
    SKP_float       *out, 
    const SKP_int16 *in, 
    SKP_int32       length
) 
{
    SKP_int32 k;
    for (k = length-1; k >= 0; k--) {
        out[k] = (SKP_float)in[k];
    }
}

SKP_INLINE SKP_float SKP_Silk_log2( double x ) { return ( SKP_float )( 3.32192809488736 * log10( x ) ); }

#endif

void SKP_Silk_find_pred_coefs_FIX(
    SKP_Silk_encoder_state_FIX      *psEnc,         /* I/O  encoder state                               */
    SKP_Silk_encoder_control_FIX    *psEncCtrl,     /* I/O  encoder control                             */
    const SKP_int16                 res_pitch[],    /* I    Residual from pitch analysis                */
    const SKP_int16                 x[]             /* I    Speech signal                               */
)
{
    SKP_int         i;
    SKP_int32       WLTP[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ];
    SKP_int32       invGains_Q16[ MAX_NB_SUBFR ], local_gains[ MAX_NB_SUBFR ], Wght_Q15[ MAX_NB_SUBFR ];
    SKP_int         NLSF_Q15[ MAX_LPC_ORDER ];
    const SKP_int16 *x_ptr;
    SKP_int16       *x_pre_ptr, LPC_in_pre[ MAX_NB_SUBFR * MAX_LPC_ORDER + MAX_FRAME_LENGTH ];
    SKP_int32       tmp, min_gain_Q16;
    SKP_int         LTP_corrs_rshift[ MAX_NB_SUBFR ];

#ifdef SAVE_ALL_INTERNAL_DATA
    SKP_int16 uq_PredCoef_Q12[ MAX_NB_SUBFR >> 1 ][ MAX_LPC_ORDER ];
    SKP_float uq_PredCoef[     MAX_NB_SUBFR >> 1 ][ MAX_LPC_ORDER ];
    SKP_float uq_LTPCoef[ MAX_NB_SUBFR * LTP_ORDER ];
#endif

    /* weighting for weighted least squares */
    min_gain_Q16 = SKP_int32_MAX >> 6;
    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        min_gain_Q16 = SKP_min( min_gain_Q16, psEncCtrl->Gains_Q16[ i ] );
    }
    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        /* Divide to Q16 */
        SKP_assert( psEncCtrl->Gains_Q16[ i ] > 0 );
        /* Invert and normalize gains, and ensure that maximum invGains_Q16 is within range of a 16 bit int */
        invGains_Q16[ i ] = SKP_DIV32_varQ( min_gain_Q16, psEncCtrl->Gains_Q16[ i ], 16 - 2 );

        /* Ensure Wght_Q15 a minimum value 1 */
        invGains_Q16[ i ] = SKP_max( invGains_Q16[ i ], 363 ); 
        
        /* Square the inverted gains */
        SKP_assert( invGains_Q16[ i ] == SKP_SAT16( invGains_Q16[ i ] ) );
        tmp = SKP_SMULWB( invGains_Q16[ i ], invGains_Q16[ i ] );
        Wght_Q15[ i ] = SKP_RSHIFT( tmp, 1 );

        /* Invert the inverted and normalized gains */
        local_gains[ i ] = SKP_DIV32( ( 1 << 16 ), invGains_Q16[ i ] );
    }

    if( psEncCtrl->sCmn.signalType == TYPE_VOICED ) {
        /**********/
        /* VOICED */
        /**********/
        SKP_assert( psEnc->sCmn.ltp_mem_length - psEnc->sCmn.predictLPCOrder >= psEncCtrl->sCmn.pitchL[ 0 ] + LTP_ORDER / 2 );

        /* LTP analysis */
        SKP_Silk_find_LTP_FIX( psEncCtrl->LTPCoef_Q14, WLTP, &psEncCtrl->LTPredCodGain_Q7, 
            res_pitch, psEncCtrl->sCmn.pitchL, Wght_Q15, psEnc->sCmn.subfr_length, 
            psEnc->sCmn.nb_subfr, psEnc->sCmn.ltp_mem_length, LTP_corrs_rshift );

#ifdef SAVE_ALL_INTERNAL_DATA
        /* Save unquantized LTP coefficients */
        for( i = 0; i < LTP_ORDER * psEnc->sCmn.nb_subfr; i++ ) {
            uq_LTPCoef[ i ] = (SKP_float)psEncCtrl->LTPCoef_Q14[ i ] / 16384.0f;
        }
#endif

        /* Quantize LTP gain parameters */
        SKP_Silk_quant_LTP_gains( psEncCtrl->LTPCoef_Q14, psEncCtrl->sCmn.LTPIndex, &psEncCtrl->sCmn.PERIndex, 
            WLTP, psEnc->sCmn.mu_LTP_Q9, psEnc->sCmn.LTPQuantLowComplexity, psEnc->sCmn.nb_subfr);

        /* Control LTP scaling */
        SKP_Silk_LTP_scale_ctrl_FIX( psEnc, psEncCtrl );

        /* Create LTP residual */
        SKP_Silk_LTP_analysis_filter_FIX( LPC_in_pre, psEnc->x_buf + psEnc->sCmn.ltp_mem_length - psEnc->sCmn.predictLPCOrder, 
            psEncCtrl->LTPCoef_Q14, psEncCtrl->sCmn.pitchL, invGains_Q16, psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr, psEnc->sCmn.predictLPCOrder );

    } else {
        /************/
        /* UNVOICED */
        /************/
        /* Create signal with prepended subframes, scaled by inverse gains */
        x_ptr     = x - psEnc->sCmn.predictLPCOrder;
        x_pre_ptr = LPC_in_pre;
        for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
            SKP_Silk_scale_copy_vector16( x_pre_ptr, x_ptr, invGains_Q16[ i ], 
                psEnc->sCmn.subfr_length + psEnc->sCmn.predictLPCOrder );
            x_pre_ptr += psEnc->sCmn.subfr_length + psEnc->sCmn.predictLPCOrder;
            x_ptr     += psEnc->sCmn.subfr_length;
        }

        SKP_memset( psEncCtrl->LTPCoef_Q14, 0, psEnc->sCmn.nb_subfr * LTP_ORDER * sizeof( SKP_int16 ) );
        psEncCtrl->LTPredCodGain_Q7 = 0;
    }

    /* LPC_in_pre contains the LTP-filtered input for voiced, and the unfiltered input for unvoiced */
    TIC(FIND_LPC)
    SKP_Silk_find_LPC_FIX( NLSF_Q15, &psEncCtrl->sCmn.NLSFInterpCoef_Q2, psEnc->sPred.prev_NLSFq_Q15, 
        psEnc->sCmn.useInterpolatedNLSFs * ( 1 - psEnc->sCmn.first_frame_after_reset ), psEnc->sCmn.predictLPCOrder, 
        LPC_in_pre, psEnc->sCmn.subfr_length + psEnc->sCmn.predictLPCOrder, psEnc->sCmn.nb_subfr );
    TOC(FIND_LPC)

#ifdef SAVE_ALL_INTERNAL_DATA /* Save unquantized LPC's */
    if( psEnc->sCmn.useInterpolatedNLSFs == 0 ) {
        /* Convert back to filter representation */
        SKP_Silk_NLSF2A_stable( uq_PredCoef_Q12[ 0 ], NLSF_Q15, psEnc->sCmn.predictLPCOrder );
        SKP_memcpy( uq_PredCoef_Q12[ 1 ], uq_PredCoef_Q12[ 0 ], psEnc->sCmn.predictLPCOrder * sizeof( SKP_int16 ) );
    } else { /* i.e. if( psEnc->useInterpolatedLSFs != 0 ) */
        SKP_int iNLSF_Q15[ MAX_LPC_ORDER ];

        /* Update interpolated LSF0 coefficients taking quantization of LSF1 coefficients into account */
        SKP_Silk_interpolate( iNLSF_Q15, psEnc->sPred.prev_NLSFq_Q15, NLSF_Q15, 
            psEncCtrl->sCmn.NLSFInterpCoef_Q2, psEnc->sCmn.predictLPCOrder );

        /* Convert back to filter representation */
        SKP_Silk_NLSF2A_stable( uq_PredCoef_Q12[ 0 ], iNLSF_Q15, psEnc->sCmn.predictLPCOrder );

        /* Convert back to filter representation */
        SKP_Silk_NLSF2A_stable( uq_PredCoef_Q12[ 1 ], NLSF_Q15, psEnc->sCmn.predictLPCOrder );
    }
    
    /* Convert to FLP */
    for( i = 0; i < psEnc->sCmn.predictLPCOrder; i++ ) {
        uq_PredCoef[ 0 ][ i ] = (SKP_float)uq_PredCoef_Q12[ 0 ][ i ] / 4096.0f;
        uq_PredCoef[ 1 ][ i ] = (SKP_float)uq_PredCoef_Q12[ 1 ][ i ] / 4096.0f;
    }
#endif

    /* Quantize LSFs */
    TIC(PROCESS_LSFS)
        SKP_Silk_process_NLSFs_FIX( psEnc, psEncCtrl, NLSF_Q15 );
    TOC(PROCESS_LSFS)

    /* Calculate residual energy using quantized LPC coefficients */
    SKP_Silk_residual_energy_FIX( psEncCtrl->ResNrg, psEncCtrl->ResNrgQ, LPC_in_pre, psEncCtrl->PredCoef_Q12, local_gains,
        psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr, psEnc->sCmn.predictLPCOrder );

    /* Copy to prediction struct for use in next frame for fluctuation reduction */
    SKP_memcpy( psEnc->sPred.prev_NLSFq_Q15, NLSF_Q15, psEnc->sCmn.predictLPCOrder * sizeof( SKP_int ) );

#ifdef SAVE_ALL_INTERNAL_DATA
    {
        SKP_int   j, k;
        SKP_float in_nrg, *in_ptr;
        SKP_float LPC_res_nrg, qLPC_res_nrg, LTP_res_nrg, qLTP_res_nrg;
        SKP_float LPC_predCodGain, QLPC_predCodGain, QLTP_predCodGain, LTPredCodGain, predCodGain;
        SKP_float LPC_res[ MAX_FRAME_LENGTH << 1 ], LTP_res[ MAX_FRAME_LENGTH ];
        SKP_float SF_resNrg[ MAX_NB_SUBFR ];

        SKP_float x_flp[ 2 * MAX_FRAME_LENGTH ];
        SKP_float Wght[ MAX_NB_SUBFR ];
        SKP_float PredCoef[ 2 ][ MAX_LPC_ORDER ];
        SKP_float LTPCoef[ MAX_NB_SUBFR * LTP_ORDER ];

        /* Convert various FIX data to FLP */
        SKP_short2float_array( x_flp, psEnc->x_buf, psEnc->sCmn.ltp_mem_length + psEnc->sCmn.frame_length );
        for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
            Wght[ k ] = ( (SKP_float)Wght_Q15[ k ] / 32768.0f );
        }
        for( i = 0; i < psEnc->sCmn.predictLPCOrder; i++ ) {
            PredCoef[ 0 ][ i ] = (SKP_float)psEncCtrl->PredCoef_Q12[ 0 ][ i ] / 4096.0f;
            PredCoef[ 1 ][ i ] = (SKP_float)psEncCtrl->PredCoef_Q12[ 1 ][ i ] / 4096.0f;
        }
        for( i = 0; i < psEnc->sCmn.nb_subfr * LTP_ORDER; i++ ) {
            LTPCoef[ i ] = (SKP_float)psEncCtrl->LTPCoef_Q14[ i ] / 16384.0f;
        }

        /* Weighted input energy */
        in_ptr = &x_flp[ psEnc->sCmn.ltp_mem_length ];
        DEBUG_STORE_DATA( x_flp.dat,  x_flp,  psEnc->sCmn.frame_length * sizeof( SKP_float ) );
        DEBUG_STORE_DATA( in_ptr.dat, in_ptr, psEnc->sCmn.frame_length * sizeof( SKP_float ) );
        in_nrg = 0.0f;
        for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
            in_nrg += (SKP_float)SKP_Silk_energy_FLP( in_ptr, psEnc->sCmn.subfr_length ) * Wght[ k ];
            in_ptr += psEnc->sCmn.subfr_length;
        }

        if( psEnc->sCmn.useInterpolatedNLSFs == 0 ) {
            SKP_memcpy( PredCoef[ 0 ], PredCoef[ 1 ], psEnc->sCmn.predictLPCOrder * sizeof( SKP_float ) );
        }

        DEBUG_STORE_DATA( uq_PredCoef.dat, uq_PredCoef[0], psEnc->sCmn.predictLPCOrder * sizeof( SKP_float ) );
        DEBUG_STORE_DATA( PredCoef.dat,    PredCoef[0],    psEnc->sCmn.predictLPCOrder * sizeof( SKP_float ) );

        LPC_res_nrg  = 0.0f;
        LTP_res_nrg  = 0.0f;
        qLPC_res_nrg = 0.0f;
        qLTP_res_nrg = 0.0f;
        for( j = 0; j < psEnc->sCmn.nb_subfr; j += 2 ) {
            /* Calculate LPC residual with unquantized LPC */
            SKP_Silk_LPC_analysis_filter_FLP( LPC_res, uq_PredCoef[ j >> 1 ], x_flp + j * psEnc->sCmn.subfr_length,
                ( psEnc->sCmn.ltp_mem_length + ( psEnc->sCmn.subfr_length << 1 ) ), psEnc->sCmn.predictLPCOrder );

            /* Weighted energy */
            in_ptr = &LPC_res[ psEnc->sCmn.ltp_mem_length ];
            for( k = 0; k < 2; k++ ) {
                LPC_res_nrg += (SKP_float)SKP_Silk_energy_FLP( in_ptr, psEnc->sCmn.subfr_length ) * Wght[ j + k ];
                in_ptr      += psEnc->sCmn.subfr_length;
            }
                
            if( psEncCtrl->sCmn.signalType == TYPE_VOICED ) {
                /* Calculate LTP residual with unquantized LTP and unquantized LPC */
                SKP_Silk_LTP_ana_core( LPC_res, LTP_res, &psEncCtrl->sCmn.pitchL[ j ],
                    &uq_LTPCoef[ j * LTP_ORDER ], psEnc->sCmn.subfr_length, psEnc->sCmn.ltp_mem_length );

                /* Weighted energy */
                in_ptr = LTP_res;
                for( k = 0; k < 2; k++ ) {
                    LTP_res_nrg += (SKP_float)SKP_Silk_energy_FLP( in_ptr, psEnc->sCmn.subfr_length ) * Wght[ j + k ];
                    in_ptr      += psEnc->sCmn.subfr_length;
                }
            }

            /* Calculate LPC residual with quantized LPC */
            SKP_Silk_LPC_analysis_filter_FLP( LPC_res, PredCoef[ j >> 1 ], x_flp + j * psEnc->sCmn.subfr_length,
                ( psEnc->sCmn.ltp_mem_length + ( psEnc->sCmn.subfr_length << 1 ) ), psEnc->sCmn.predictLPCOrder );

            /* Weighted energy */
            in_ptr = &LPC_res[ psEnc->sCmn.ltp_mem_length ];
            for( k = 0; k < 2; k++ ) {
                SF_resNrg[ k + j ] = (SKP_float)SKP_Silk_energy_FLP( in_ptr, psEnc->sCmn.subfr_length );
                qLPC_res_nrg += SF_resNrg[ k + j ] * Wght[ j + k ];
                in_ptr       += psEnc->sCmn.subfr_length;
            }

            if( psEncCtrl->sCmn.signalType == TYPE_VOICED ) {
                /* Calculate LTP residual with unquantized LTP and unquantized LPC */
                SKP_Silk_LTP_ana_core( LPC_res, LTP_res, &psEncCtrl->sCmn.pitchL[ j ],
                    &LTPCoef[ j * LTP_ORDER ], psEnc->sCmn.subfr_length, psEnc->sCmn.ltp_mem_length );

                /* Weighted energy */
                in_ptr = LTP_res;
                for( k = 0; k < 2; k++ ) {
                    SF_resNrg[ k + j ] = (SKP_float)SKP_Silk_energy_FLP( in_ptr, psEnc->sCmn.subfr_length );
                    qLTP_res_nrg += SF_resNrg[ k + j ] * Wght[ j + k ];
                    in_ptr       += psEnc->sCmn.subfr_length;
                }
            } else {
                SKP_memcpy( LTP_res, &LPC_res[ psEnc->sCmn.ltp_mem_length ], ( psEnc->sCmn.subfr_length << 1 ) * sizeof( SKP_float ) );
            }
            /* Save residual */
            DEBUG_STORE_DATA( LPC_res.dat, &LPC_res[ psEnc->sCmn.ltp_mem_length ], ( psEnc->sCmn.subfr_length << 1 ) * sizeof( SKP_float ) );
            DEBUG_STORE_DATA( res.dat,     LTP_res,                                ( psEnc->sCmn.subfr_length << 1 ) * sizeof( SKP_float ) );
        }
        if( psEncCtrl->sCmn.signalType == TYPE_VOICED ) {
            LPC_predCodGain  = 3.0f * SKP_Silk_log2( in_nrg       / LPC_res_nrg  );
            QLPC_predCodGain = 3.0f * SKP_Silk_log2( in_nrg       / qLPC_res_nrg );
            LTPredCodGain    = 3.0f * SKP_Silk_log2( LPC_res_nrg  / LTP_res_nrg  );
            QLTP_predCodGain = 3.0f * SKP_Silk_log2( qLPC_res_nrg / qLTP_res_nrg );
        } else {
            LPC_predCodGain  = 3.0f * SKP_Silk_log2( in_nrg       / LPC_res_nrg  );
            QLPC_predCodGain = 3.0f * SKP_Silk_log2( in_nrg       / qLPC_res_nrg );
            LTPredCodGain    = 0.0f;
            QLTP_predCodGain = 0.0f;
        }
        predCodGain = QLPC_predCodGain + QLTP_predCodGain;

        DEBUG_STORE_DATA( LTPredCodGain.dat,    &LTPredCodGain,                                        sizeof( SKP_float ) );
        DEBUG_STORE_DATA( QLTP_predCodGain.dat, &QLTP_predCodGain,                                     sizeof( SKP_float ) ); 
        DEBUG_STORE_DATA( LPC_predCodGain.dat,  &LPC_predCodGain,                                      sizeof( SKP_float ) );
        DEBUG_STORE_DATA( QLPC_predCodGain.dat, &QLPC_predCodGain,                                     sizeof( SKP_float ) );
        DEBUG_STORE_DATA( predCodGain.dat,      &predCodGain,                                          sizeof( SKP_float ) ); 
        DEBUG_STORE_DATA( ResNrg.dat,           SF_resNrg,                      psEnc->sCmn.nb_subfr * sizeof( SKP_float ) );
    }
#endif
}

#ifdef SAVE_ALL_INTERNAL_DATA
/****************************************************/
/* LTP analysis filter. Filters two subframes       */
/****************************************************/
void SKP_Silk_LTP_ana_core(
    SKP_float                       r_LPC[],        /* I    LPC residual            */
    SKP_float                       r_LTP[],        /* O    LTP residual            */
    const SKP_int                   pitchL[],       /* I    pitch lags              */
    const SKP_float                 LTPCoef[],      /* I    LTP Coeficients         */
    SKP_int                         subfr_length,   /* I    smpls in one sub frame  */
    SKP_int                         LTP_mem_length  /* I    Length of LTP state of input */
)
{
    SKP_int   k, i;
    SKP_float LTP_pred;
    const SKP_float *r, *b_ptr, *lag_ptr;

    r = &r_LPC[ LTP_mem_length ];
    b_ptr = LTPCoef;
    for( k = 0; k < (MAX_NB_SUBFR >> 1); k++ ) {
        lag_ptr = r - pitchL[k];
        /* LTP analysis FIR filter */
        for( i = 0; i < subfr_length; i++ ) {
            /* long-term prediction */
            LTP_pred  = lag_ptr[LTP_ORDER/2]     * b_ptr[0];
            LTP_pred += lag_ptr[LTP_ORDER/2 - 1] * b_ptr[1];
            LTP_pred += lag_ptr[LTP_ORDER/2 - 2] * b_ptr[2];
            LTP_pred += lag_ptr[LTP_ORDER/2 - 3] * b_ptr[3];
            LTP_pred += lag_ptr[LTP_ORDER/2 - 4] * b_ptr[4];

            /* subtract prediction */
            r_LTP[i] = r[i] - LTP_pred;
            lag_ptr++;
        }
        r += subfr_length;
        r_LTP += subfr_length;
        b_ptr += LTP_ORDER;
    }
}

/*******************************************/
/* LPC analysis filter                     */
/* NB! State is kept internally and the    */
/* filter always starts with zero state    */
/* first Order output samples are not set  */
/*******************************************/
void SKP_Silk_LPC_analysis_filter_FLP(
    SKP_float                       r_LPC[],        /* O    LPC residual signal             */
    const SKP_float                 PredCoef[],     /* I    LPC coeficicnts                 */
    const SKP_float                 s[],            /* I    Input Signal                    */
    SKP_int                         length,         /* I    length of signal                */
    SKP_int                         Order           /* I    LPC order                       */
)
{
    SKP_int   i, j;
    SKP_float LPC_pred;
    const SKP_float *s_ptr;

    for ( i = Order; i < length; i++ ) {
        s_ptr = &s[i - 1];

        LPC_pred = 0;
        /* short-term prediction */
        for( j = 0; j < Order; j++ ) {
            LPC_pred += s_ptr[ -j ] * PredCoef[ j ];
        }

        /* prediction error */
        r_LPC[ i ] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* sum of squares of a SKP_float array, with result as double */
double SKP_Silk_energy_FLP( 
    const SKP_float     *data, 
    SKP_int             dataSize
)
{
    SKP_int  i, dataSize4;
    double   result;

    /* 4x unrolled loop */
    result = 0.0f;
    dataSize4 = dataSize & 0xFFFC;
    for( i = 0; i < dataSize4; i += 4 ) {
        result += data[ i + 0 ] * data[ i + 0 ] + 
                  data[ i + 1 ] * data[ i + 1 ] +
                  data[ i + 2 ] * data[ i + 2 ] +
                  data[ i + 3 ] * data[ i + 3 ];
    }

    /* add any remaining products */
    for( ; i < dataSize; i++ ) {
        result += data[ i ] * data[ i ];
    }

    SKP_assert( result >= 0.0 );
    return result;
}
#endif
