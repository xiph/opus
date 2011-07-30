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
#include "silk_tuning_parameters.h"

/****************/
/* Encode frame */
/****************/
opus_int silk_encode_frame_FLP(
    silk_encoder_state_FLP          *psEnc,             /* I/O  Encoder state FLP                       */
    opus_int32                       *pnBytesOut,        /*   O  Number of payload bytes                 */
    ec_enc                          *psRangeEnc         /* I/O  compressor data structure               */
)
{
    silk_encoder_control_FLP sEncCtrl;
    opus_int     i, ret = 0;
    SKP_float   *x_frame, *res_pitch_frame;
    SKP_float   xfw[ MAX_FRAME_LENGTH ];
    SKP_float   res_pitch[ 2 * MAX_FRAME_LENGTH + LA_PITCH_MAX ];

TIC(ENCODE_FRAME)

    psEnc->sCmn.indices.Seed = psEnc->sCmn.frameCounter++ & 3;

    /**************************************************************/
    /* Setup Input Pointers, and insert frame in input buffer    */
    /*************************************************************/
    /* pointers aligned with start of frame to encode */
    x_frame         = psEnc->x_buf + psEnc->sCmn.ltp_mem_length;    /* start of frame to encode */
    res_pitch_frame = res_pitch    + psEnc->sCmn.ltp_mem_length;    /* start of pitch LPC residual frame */

    /****************************/
    /* Voice Activity Detection */
    /****************************/
TIC(VAD)
    ret = silk_VAD_GetSA_Q8( &psEnc->sCmn, psEnc->sCmn.inputBuf - 1 );
TOC(VAD)

    /**************************************************/
    /* Convert speech activity into VAD and DTX flags */
    /**************************************************/
    if( psEnc->sCmn.nFramesEncoded == 0 ) {
        psEnc->sCmn.inDTX = psEnc->sCmn.useDTX;
    }
    if( psEnc->sCmn.speech_activity_Q8 < SILK_FIX_CONST( SPEECH_ACTIVITY_DTX_THRES, 8 ) ) {
        psEnc->sCmn.indices.signalType = TYPE_NO_VOICE_ACTIVITY;
        psEnc->sCmn.noSpeechCounter++;
        if( psEnc->sCmn.noSpeechCounter < NB_SPEECH_FRAMES_BEFORE_DTX ) {
            psEnc->sCmn.inDTX = 0;
        } else if( psEnc->sCmn.noSpeechCounter > MAX_CONSECUTIVE_DTX + NB_SPEECH_FRAMES_BEFORE_DTX ) {
            psEnc->sCmn.noSpeechCounter = NB_SPEECH_FRAMES_BEFORE_DTX;
            psEnc->sCmn.inDTX           = 0;
        }
        psEnc->sCmn.VAD_flags[ psEnc->sCmn.nFramesEncoded ] = 0;
    } else {
        psEnc->sCmn.noSpeechCounter    = 0;
        psEnc->sCmn.inDTX              = 0;
        psEnc->sCmn.indices.signalType = TYPE_UNVOICED;
        psEnc->sCmn.VAD_flags[ psEnc->sCmn.nFramesEncoded ] = 1;
    }

    /***************************************/
    /* Ensure smooth bandwidth transitions */
    /***************************************/
    silk_LP_variable_cutoff( &psEnc->sCmn.sLP, psEnc->sCmn.inputBuf - 1, psEnc->sCmn.frame_length );

    /*******************************************/
    /* Copy new frame to front of input buffer */
    /*******************************************/
    SKP_short2float_array( x_frame + LA_SHAPE_MS * psEnc->sCmn.fs_kHz, psEnc->sCmn.inputBuf - 1, psEnc->sCmn.frame_length );

    /* Add tiny signal to avoid high CPU load from denormalized floating point numbers */
    for( i = 0; i < 8; i++ ) {
        x_frame[ LA_SHAPE_MS * psEnc->sCmn.fs_kHz + i * ( psEnc->sCmn.frame_length >> 3 ) ] += ( 1 - ( i & 2 ) ) * 1e-6f;
    }

    /*****************************************/
    /* Find pitch lags, initial LPC analysis */
    /*****************************************/
TIC(FIND_PITCH)
    silk_find_pitch_lags_FLP( psEnc, &sEncCtrl, res_pitch, x_frame );
TOC(FIND_PITCH)

    /************************/
    /* Noise shape analysis */
    /************************/
TIC(NOISE_SHAPE_ANALYSIS)
    silk_noise_shape_analysis_FLP( psEnc, &sEncCtrl, res_pitch_frame, x_frame );
TOC(NOISE_SHAPE_ANALYSIS)

    /***************************************************/
    /* Find linear prediction coefficients (LPC + LTP) */
    /***************************************************/
TIC(FIND_PRED_COEF)
    silk_find_pred_coefs_FLP( psEnc, &sEncCtrl, res_pitch, x_frame );
TOC(FIND_PRED_COEF)

    /****************************************/
    /* Process gains                        */
    /****************************************/
TIC(PROCESS_GAINS)
    silk_process_gains_FLP( psEnc, &sEncCtrl );
TOC(PROCESS_GAINS)

    /*****************************************/
    /* Prefiltering for noise shaper         */
    /*****************************************/
TIC(PREFILTER)
    silk_prefilter_FLP( psEnc, &sEncCtrl, xfw, x_frame );
TOC(PREFILTER)

    /****************************************/
    /* Low Bitrate Redundant Encoding       */
    /****************************************/
TIC(LBRR)
    silk_LBRR_encode_FLP( psEnc, &sEncCtrl, xfw );
TOC(LBRR)

    /*****************************************/
    /* Noise shaping quantization            */
    /*****************************************/
TIC(NSQ)
    silk_NSQ_wrapper_FLP( psEnc, &sEncCtrl, &psEnc->sCmn.indices, &psEnc->sCmn.sNSQ, psEnc->sCmn.pulses, xfw );
TOC(NSQ)

    /* Update input buffer */
    SKP_memmove( psEnc->x_buf, &psEnc->x_buf[ psEnc->sCmn.frame_length ],
        ( psEnc->sCmn.ltp_mem_length + LA_SHAPE_MS * psEnc->sCmn.fs_kHz ) * sizeof( SKP_float ) );

    /* Parameters needed for next frame */
    psEnc->sCmn.prevLag        = sEncCtrl.pitchL[ psEnc->sCmn.nb_subfr - 1 ];
    psEnc->sCmn.prevSignalType = psEnc->sCmn.indices.signalType;

    /* Exit without entropy coding */
    if( psEnc->sCmn.prefillFlag ) {
        /* No payload */
        *pnBytesOut = 0;
        return ret;
    }

    /****************************************/
    /* Encode Parameters                    */
    /****************************************/
TIC(ENCODE_PARAMS)
    silk_encode_indices( &psEnc->sCmn, psRangeEnc, psEnc->sCmn.nFramesEncoded, 0 );
TOC(ENCODE_PARAMS)

    /****************************************/
    /* Encode Excitation Signal             */
    /****************************************/
TIC(ENCODE_PULSES)
    silk_encode_pulses( psRangeEnc, psEnc->sCmn.indices.signalType, psEnc->sCmn.indices.quantOffsetType,
        psEnc->sCmn.pulses, psEnc->sCmn.frame_length );
TOC(ENCODE_PULSES)

    /****************************************/
    /* Finalize payload                     */
    /****************************************/
    psEnc->sCmn.first_frame_after_reset = 0;
    if( ++psEnc->sCmn.nFramesEncoded >= psEnc->sCmn.nFramesPerPacket ) {
        /* Payload size */
        *pnBytesOut = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );

        /* Reset the number of frames in payload buffer */
        psEnc->sCmn.nFramesEncoded = 0;
    } else {
        /* No payload this time */
        *pnBytesOut = 0;
    }
TOC(ENCODE_FRAME)

#ifdef SAVE_ALL_INTERNAL_DATA
    //DEBUG_STORE_DATA( xf.dat,                   pIn_HP_LP,                           psEnc->sCmn.frame_length * sizeof( opus_int16 ) );
    //DEBUG_STORE_DATA( xfw.dat,                  xfw,                                 psEnc->sCmn.frame_length * sizeof( SKP_float ) );
    DEBUG_STORE_DATA( pitchL.dat,               sEncCtrl.pitchL,                                 MAX_NB_SUBFR * sizeof( opus_int   ) );
    DEBUG_STORE_DATA( pitchG_quantized.dat,     sEncCtrl.LTPCoef,            psEnc->sCmn.nb_subfr * LTP_ORDER * sizeof( SKP_float ) );
    DEBUG_STORE_DATA( LTPcorr.dat,              &psEnc->LTPCorr,                                                sizeof( SKP_float ) );
    DEBUG_STORE_DATA( gains.dat,                sEncCtrl.Gains,                          psEnc->sCmn.nb_subfr * sizeof( SKP_float ) );
    DEBUG_STORE_DATA( gains_indices.dat,        &psEnc->sCmn.indices.GainsIndices,       psEnc->sCmn.nb_subfr * sizeof( opus_int8  ) );
    DEBUG_STORE_DATA( quantOffsetType.dat,      &psEnc->sCmn.indices.quantOffsetType,                           sizeof( opus_int8  ) );
    DEBUG_STORE_DATA( speech_activity_q8.dat,   &psEnc->sCmn.speech_activity_Q8,                                sizeof( opus_int   ) );
    DEBUG_STORE_DATA( signalType.dat,           &psEnc->sCmn.indices.signalType,                                sizeof( opus_int8  ) );
    DEBUG_STORE_DATA( lag_index.dat,            &psEnc->sCmn.indices.lagIndex,                                  sizeof( opus_int16 ) );
    DEBUG_STORE_DATA( contour_index.dat,        &psEnc->sCmn.indices.contourIndex,                              sizeof( opus_int8  ) );
    DEBUG_STORE_DATA( per_index.dat,            &psEnc->sCmn.indices.PERIndex,                                  sizeof( opus_int8  ) );
    DEBUG_STORE_DATA( PredCoef.dat,             &sEncCtrl.PredCoef[ 1 ],          psEnc->sCmn.predictLPCOrder * sizeof( SKP_float ) );
    DEBUG_STORE_DATA( ltp_scale_idx.dat,        &psEnc->sCmn.indices.LTP_scaleIndex,                            sizeof( opus_int8   ) );
//  DEBUG_STORE_DATA( xq.dat,                   psEnc->sCmn.sNSQ.xqBuf,                psEnc->sCmn.frame_length * sizeof( SKP_float ) );
#endif
    return ret;
}

/* Low-Bitrate Redundancy (LBRR) encoding. Reuse all parameters but encode excitation at lower bitrate  */
void silk_LBRR_encode_FLP(
    silk_encoder_state_FLP          *psEnc,             /* I/O  Encoder state FLP                       */
    silk_encoder_control_FLP        *psEncCtrl,         /* I/O  Encoder control FLP                     */
    const SKP_float                 xfw[]               /* I    Input signal                            */
)
{
    opus_int     k;
    opus_int32   Gains_Q16[ MAX_NB_SUBFR ];
    SKP_float   TempGains[ MAX_NB_SUBFR ];
    SideInfoIndices *psIndices_LBRR = &psEnc->sCmn.indices_LBRR[ psEnc->sCmn.nFramesEncoded ];
    silk_nsq_state sNSQ_LBRR;

    /*******************************************/
    /* Control use of inband LBRR              */
    /*******************************************/
    if( psEnc->sCmn.LBRR_enabled && psEnc->sCmn.speech_activity_Q8 > SILK_FIX_CONST( LBRR_SPEECH_ACTIVITY_THRES, 8 ) ) {
        psEnc->sCmn.LBRR_flags[ psEnc->sCmn.nFramesEncoded ] = 1;

        /* Copy noise shaping quantizer state and quantization indices from regular encoding */
        SKP_memcpy( &sNSQ_LBRR, &psEnc->sCmn.sNSQ, sizeof( silk_nsq_state ) );
        SKP_memcpy( psIndices_LBRR, &psEnc->sCmn.indices, sizeof( SideInfoIndices ) );

        /* Save original gains */
        SKP_memcpy( TempGains, psEncCtrl->Gains, psEnc->sCmn.nb_subfr * sizeof( SKP_float ) );

        if( psEnc->sCmn.nFramesEncoded == 0 || psEnc->sCmn.LBRR_flags[ psEnc->sCmn.nFramesEncoded - 1 ] == 0 ) {
            /* First frame in packet or previous frame not LBRR coded */
            psEnc->sCmn.LBRRprevLastGainIndex = psEnc->sShape.LastGainIndex;

            /* Increase Gains to get target LBRR rate */
            psIndices_LBRR->GainsIndices[ 0 ] += psEnc->sCmn.LBRR_GainIncreases;
            psIndices_LBRR->GainsIndices[ 0 ] = SKP_min_int( psIndices_LBRR->GainsIndices[ 0 ], N_LEVELS_QGAIN - 1 );
        }

        /* Decode to get gains in sync with decoder */
        silk_gains_dequant( Gains_Q16, psIndices_LBRR->GainsIndices,
            &psEnc->sCmn.LBRRprevLastGainIndex, psEnc->sCmn.nFramesEncoded, psEnc->sCmn.nb_subfr );

        /* Overwrite unquantized gains with quantized gains and convert back to Q0 from Q16 */
        for( k = 0; k <  psEnc->sCmn.nb_subfr; k++ ) {
            psEncCtrl->Gains[ k ] = Gains_Q16[ k ] * ( 1.0f / 65536.0f );
        }

        /*****************************************/
        /* Noise shaping quantization            */
        /*****************************************/
        silk_NSQ_wrapper_FLP( psEnc, psEncCtrl, psIndices_LBRR, &sNSQ_LBRR,
            psEnc->sCmn.pulses_LBRR[ psEnc->sCmn.nFramesEncoded ], xfw );

        /* Restore original gains */
        SKP_memcpy( psEncCtrl->Gains, TempGains, psEnc->sCmn.nb_subfr * sizeof( SKP_float ) );
    }
}
