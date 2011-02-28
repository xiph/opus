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

#include "SKP_Silk_main.h"
#include "SKP_Silk_tuning_parameters.h"

SKP_INLINE SKP_int SKP_Silk_setup_complexity(
    SKP_Silk_encoder_state          *psEncC,            /* I/O                      */
    SKP_int                         Complexity          /* I                        */
)
{
    SKP_int ret = 0;

    /* Set encoding complexity */
    if( Complexity < 2 ) {
        psEncC->pitchEstimationComplexity       = SKP_Silk_PE_MIN_COMPLEX;
        psEncC->pitchEstimationThreshold_Q16    = SKP_FIX_CONST( 0.8, 16 );
        psEncC->pitchEstimationLPCOrder         = 6;
        psEncC->shapingLPCOrder                 = 8;
        psEncC->la_shape                        = 3 * psEncC->fs_kHz;
        psEncC->nStatesDelayedDecision          = 1;
        psEncC->useInterpolatedNLSFs            = 0;
        psEncC->LTPQuantLowComplexity           = 1;
        psEncC->NLSF_MSVQ_Survivors             = 2;
        psEncC->warping_Q16                     = 0;
    } else if( Complexity < 4 ) {
        psEncC->pitchEstimationComplexity       = SKP_Silk_PE_MID_COMPLEX;
        psEncC->pitchEstimationThreshold_Q16    = SKP_FIX_CONST( 0.76, 16 );
        psEncC->pitchEstimationLPCOrder         = 8;
        psEncC->shapingLPCOrder                 = 10;
        psEncC->la_shape                        = 5 * psEncC->fs_kHz;
        psEncC->nStatesDelayedDecision          = 1;
        psEncC->useInterpolatedNLSFs            = 1;
        psEncC->LTPQuantLowComplexity           = 0;
        psEncC->NLSF_MSVQ_Survivors             = 4;
        psEncC->warping_Q16                     = 0;
    } else if( Complexity < 6 ) {
        psEncC->pitchEstimationComplexity       = SKP_Silk_PE_MID_COMPLEX;
        psEncC->pitchEstimationThreshold_Q16    = SKP_FIX_CONST( 0.74, 16 );
        psEncC->pitchEstimationLPCOrder         = 10;
        psEncC->shapingLPCOrder                 = 12;
        psEncC->la_shape                        = 5 * psEncC->fs_kHz;
        psEncC->nStatesDelayedDecision          = 2;
        psEncC->useInterpolatedNLSFs            = 0;
        psEncC->LTPQuantLowComplexity           = 0;
        psEncC->NLSF_MSVQ_Survivors             = 6;
        psEncC->warping_Q16                     = psEncC->fs_kHz * SKP_FIX_CONST( WARPING_MULTIPLIER, 16 );
    } else if( Complexity < 8 ) {
        psEncC->pitchEstimationComplexity       = SKP_Silk_PE_MID_COMPLEX;
        psEncC->pitchEstimationThreshold_Q16    = SKP_FIX_CONST( 0.72, 16 );
        psEncC->pitchEstimationLPCOrder         = 12;
        psEncC->shapingLPCOrder                 = 14;
        psEncC->la_shape                        = 5 * psEncC->fs_kHz;
        psEncC->nStatesDelayedDecision          = 3;
        psEncC->useInterpolatedNLSFs            = 0;
        psEncC->LTPQuantLowComplexity           = 0;
        psEncC->NLSF_MSVQ_Survivors             = 8;
        psEncC->warping_Q16                     = psEncC->fs_kHz * SKP_FIX_CONST( WARPING_MULTIPLIER, 16 );
    } else if( Complexity <= 10 ) {
        psEncC->pitchEstimationComplexity       = SKP_Silk_PE_MAX_COMPLEX;
        psEncC->pitchEstimationThreshold_Q16    = SKP_FIX_CONST( 0.7, 16 );
        psEncC->pitchEstimationLPCOrder         = 16;
        psEncC->shapingLPCOrder                 = 16;
        psEncC->la_shape                        = 5 * psEncC->fs_kHz;
        psEncC->nStatesDelayedDecision          = MAX_DEL_DEC_STATES;
        psEncC->useInterpolatedNLSFs            = 1;
        psEncC->LTPQuantLowComplexity           = 0;
        psEncC->NLSF_MSVQ_Survivors             = 16;
        psEncC->warping_Q16                     = psEncC->fs_kHz * SKP_FIX_CONST( WARPING_MULTIPLIER, 16 );
    } else {
        ret = SKP_SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }

    /* Do not allow higher pitch estimation LPC order than predict LPC order */
    psEncC->pitchEstimationLPCOrder = SKP_min_int( psEncC->pitchEstimationLPCOrder, psEncC->predictLPCOrder );
    psEncC->shapeWinLength          = SUB_FRAME_LENGTH_MS * psEncC->fs_kHz + 2 * psEncC->la_shape;

    SKP_assert( psEncC->pitchEstimationLPCOrder <= MAX_FIND_PITCH_LPC_ORDER );
    SKP_assert( psEncC->shapingLPCOrder         <= MAX_SHAPE_LPC_ORDER      );
    SKP_assert( psEncC->nStatesDelayedDecision  <= MAX_DEL_DEC_STATES       );
    SKP_assert( psEncC->warping_Q16             <= 32767                    );
    SKP_assert( psEncC->la_shape                <= LA_SHAPE_MAX             );
    SKP_assert( psEncC->shapeWinLength          <= SHAPE_LPC_WIN_MAX        );
    SKP_assert( psEncC->NLSF_MSVQ_Survivors     <= NLSF_VQ_MAX_SURVIVORS    );

    return( ret );
}

SKP_INLINE SKP_int SKP_Silk_setup_LBRR(
    SKP_Silk_encoder_state          *psEncC            /* I/O                      */
)
{
    SKP_int   ret = SKP_SILK_NO_ERROR;
    SKP_int32 LBRRRate_thres_bps;

    if( psEncC->useInBandFEC < 0 || psEncC->useInBandFEC > 1 ) {
        ret = SKP_SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    
    if( psEncC->fs_kHz == 8 ) {
        LBRRRate_thres_bps = LBRR_NB_MIN_RATE_BPS;
    } else if( psEncC->fs_kHz == 12 ) {
        LBRRRate_thres_bps = LBRR_MB_MIN_RATE_BPS;
    } else if( psEncC->fs_kHz == 16 ) {
        LBRRRate_thres_bps = LBRR_WB_MIN_RATE_BPS;
    } else {
        SKP_assert( 0 );
    }

    LBRRRate_thres_bps = SKP_RSHIFT( SKP_SMULBB( LBRRRate_thres_bps, 7 - psEncC->PacketLoss_perc ), 2 );
    if( psEncC->useInBandFEC && psEncC->TargetRate_bps >= LBRRRate_thres_bps && psEncC->PacketLoss_perc > 0 ) {
        /* Set gain increase / rate reduction for LBRR usage */
        psEncC->LBRR_GainIncreases = SKP_max_int( 6 - SKP_SMULWB( psEncC->PacketLoss_perc, SKP_FIX_CONST( 0.4, 16 ) ), 2 );
        psEncC->LBRR_enabled = 1;
    } else {
        psEncC->LBRR_enabled = 0;
    }
    return ret;
}
