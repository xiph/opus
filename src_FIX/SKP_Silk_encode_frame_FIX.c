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

#include "SKP_Silk_main_FIX.h"
#include "SKP_Silk_tuning_parameters.h"

/****************/
/* Encode frame */
/****************/
SKP_int SKP_Silk_encode_frame_FIX( 
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O  Encoder state FIX                       */
    SKP_int32                       *pnBytesOut,        /*   O  Number of payload bytes                 */
    ec_enc                          *psRangeEnc         /* I/O  compressor data structure               */
)
{
    SKP_Silk_encoder_control_FIX sEncCtrl;
    SKP_int     i, nBits, ret = 0;
    SKP_uint8   flags;
    SKP_int16   *x_frame, *res_pitch_frame;
    SKP_int16   xfw[ MAX_FRAME_LENGTH ];
    SKP_int16   res_pitch[ 2 * MAX_FRAME_LENGTH + LA_PITCH_MAX ];

TIC(ENCODE_FRAME)

    if( psEnc->sCmn.nFramesAnalyzed == 0 && !psEnc->sCmn.prefillFlag && !( psEnc->sCmn.useDTX && psEnc->sCmn.inDTX ) ) {
        /* Create space at start of payload for VAD and FEC flags */
        SKP_uint8 iCDF[ 2 ] = { 0, 0 };
        iCDF[ 0 ] = 256 - SKP_RSHIFT( 256, psEnc->sCmn.nFramesPerPacket + 1 );
        ec_enc_icdf( psRangeEnc, 0, iCDF, 8 );

        /* Encode any LBRR data from previous packet */
        SKP_Silk_LBRR_embed( &psEnc->sCmn, psRangeEnc );

        /* Reduce coding SNR depending on how many bits used by LBRR */
        nBits = ec_tell( psRangeEnc );
        psEnc->inBandFEC_SNR_comp_Q7 = SKP_DIV32_16( SKP_SMULBB( SKP_FIX_CONST( 6.0f, 7 ), nBits ), 
            SKP_SMULBB( psEnc->sCmn.nFramesPerPacket, psEnc->sCmn.frame_length ) );

        /* Reset LBRR flags */
        SKP_memset( psEnc->sCmn.LBRR_flags, 0, sizeof( psEnc->sCmn.LBRR_flags ) );
    }

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
    ret = SKP_Silk_VAD_GetSA_Q8( &psEnc->sCmn, psEnc->sCmn.inputBuf );
TOC(VAD)

    /**************************************************/
    /* Convert speech activity into VAD and DTX flags */
    /**************************************************/
    if( psEnc->sCmn.speech_activity_Q8 < SKP_FIX_CONST( SPEECH_ACTIVITY_DTX_THRES, 8 ) ) {
        psEnc->sCmn.indices.signalType = TYPE_NO_VOICE_ACTIVITY;
        psEnc->sCmn.noSpeechCounter++;
        if( psEnc->sCmn.noSpeechCounter > NO_SPEECH_FRAMES_BEFORE_DTX ) {
            psEnc->sCmn.inDTX = 1;
        }
        if( psEnc->sCmn.noSpeechCounter > MAX_CONSECUTIVE_DTX ) {
            psEnc->sCmn.noSpeechCounter = 0;
            psEnc->sCmn.inDTX           = 0;
        }
        psEnc->sCmn.VAD_flags[ psEnc->sCmn.nFramesAnalyzed ] = 0;
    } else {
        psEnc->sCmn.noSpeechCounter    = 0;
        psEnc->sCmn.inDTX              = 0;
        psEnc->sCmn.indices.signalType = TYPE_UNVOICED;
        psEnc->sCmn.VAD_flags[ psEnc->sCmn.nFramesAnalyzed ] = 1;
    }

    /*******************************************/
    /* High-pass filtering of the input signal */
    /*******************************************/
TIC(HP_IN)
    SKP_Silk_HP_variable_cutoff( &psEnc->sCmn, x_frame + LA_SHAPE_MS * psEnc->sCmn.fs_kHz, psEnc->sCmn.inputBuf, psEnc->sCmn.frame_length );
TOC(HP_IN)

#if SWITCH_TRANSITION_FILTERING
    /* Ensure smooth bandwidth transitions */
    SKP_Silk_LP_variable_cutoff( &psEnc->sCmn.sLP, x_frame + LA_SHAPE_MS * psEnc->sCmn.fs_kHz, psEnc->sCmn.frame_length );
#endif
    
    /*****************************************/
    /* Find pitch lags, initial LPC analysis */
    /*****************************************/
TIC(FIND_PITCH)
    SKP_Silk_find_pitch_lags_FIX( psEnc, &sEncCtrl, res_pitch, x_frame );
TOC(FIND_PITCH)

    /************************/
    /* Noise shape analysis */
    /************************/
TIC(NOISE_SHAPE_ANALYSIS)
    SKP_Silk_noise_shape_analysis_FIX( psEnc, &sEncCtrl, res_pitch_frame, x_frame );
TOC(NOISE_SHAPE_ANALYSIS)

    /*****************************************/
    /* Prefiltering for noise shaper         */
    /*****************************************/
TIC(PREFILTER)
    SKP_Silk_prefilter_FIX( psEnc, &sEncCtrl, xfw, x_frame );
TOC(PREFILTER)

    /***************************************************/
    /* Find linear prediction coefficients (LPC + LTP) */
    /***************************************************/
TIC(FIND_PRED_COEF)
    SKP_Silk_find_pred_coefs_FIX( psEnc, &sEncCtrl, res_pitch, x_frame );
TOC(FIND_PRED_COEF)

    /****************************************/
    /* Process gains                        */
    /****************************************/
TIC(PROCESS_GAINS)
    SKP_Silk_process_gains_FIX( psEnc, &sEncCtrl );
TOC(PROCESS_GAINS)
    
    /****************************************/
    /* Low Bitrate Redundant Encoding       */
    /****************************************/
TIC(LBRR)
    SKP_Silk_LBRR_encode_FIX( psEnc, &sEncCtrl, xfw );
TOC(LBRR)

    /*****************************************/
    /* Noise shaping quantization            */
    /*****************************************/
TIC(NSQ)
    if( psEnc->sCmn.nStatesDelayedDecision > 1 || psEnc->sCmn.warping_Q16 > 0 ) {
        SKP_Silk_NSQ_del_dec( &psEnc->sCmn, &psEnc->sCmn.sNSQ, &psEnc->sCmn.indices, xfw, psEnc->sCmn.pulses, 
            sEncCtrl.PredCoef_Q12[ 0 ], sEncCtrl.LTPCoef_Q14, sEncCtrl.AR2_Q13, sEncCtrl.HarmShapeGain_Q14, 
            sEncCtrl.Tilt_Q14, sEncCtrl.LF_shp_Q14, sEncCtrl.Gains_Q16, sEncCtrl.pitchL, sEncCtrl.Lambda_Q10, sEncCtrl.LTP_scale_Q14 );
    } else {
        SKP_Silk_NSQ( &psEnc->sCmn, &psEnc->sCmn.sNSQ, &psEnc->sCmn.indices, xfw, psEnc->sCmn.pulses, 
            sEncCtrl.PredCoef_Q12[ 0 ], sEncCtrl.LTPCoef_Q14, sEncCtrl.AR2_Q13, sEncCtrl.HarmShapeGain_Q14, 
            sEncCtrl.Tilt_Q14, sEncCtrl.LF_shp_Q14, sEncCtrl.Gains_Q16, sEncCtrl.pitchL, sEncCtrl.Lambda_Q10, sEncCtrl.LTP_scale_Q14 );
    }
TOC(NSQ)

    /* Update input buffer */
    SKP_memmove( psEnc->x_buf, &psEnc->x_buf[ psEnc->sCmn.frame_length ], 
        ( psEnc->sCmn.ltp_mem_length + LA_SHAPE_MS * psEnc->sCmn.fs_kHz ) * sizeof( SKP_int16 ) );

    /* Parameters needed for next frame */
    psEnc->sCmn.prevLag        = sEncCtrl.pitchL[ psEnc->sCmn.nb_subfr - 1 ];
    psEnc->sCmn.prevSignalType = psEnc->sCmn.indices.signalType;

    /* Exit without entropy coding */
    if( psEnc->sCmn.prefillFlag || ( psEnc->sCmn.useDTX && psEnc->sCmn.inDTX ) ) {
        /* No payload */
        *pnBytesOut = 0;
        return ret;
    }

    /****************************************/
    /* Encode Parameters                    */
    /****************************************/
TIC(ENCODE_PARAMS)
    SKP_Silk_encode_indices( &psEnc->sCmn, psRangeEnc, psEnc->sCmn.nFramesAnalyzed, 0 );
TOC(ENCODE_PARAMS)

    /****************************************/
    /* Encode Excitation Signal             */
    /****************************************/
TIC(ENCODE_PULSES)
    SKP_Silk_encode_pulses( psRangeEnc, psEnc->sCmn.indices.signalType, psEnc->sCmn.indices.quantOffsetType, 
        psEnc->sCmn.pulses, psEnc->sCmn.frame_length );
TOC(ENCODE_PULSES)

    /****************************************/
    /* Simulate network buffer delay caused */
    /* by exceeding TargetRate              */
    /****************************************/
    nBits = ec_tell( psRangeEnc );
    psEnc->BufferedInChannel_ms += SKP_DIV32( SKP_SMULBB( 1000, nBits - psEnc->sCmn.prev_nBits ), psEnc->sCmn.TargetRate_bps );
    psEnc->BufferedInChannel_ms -= psEnc->sCmn.PacketSize_ms;
    psEnc->BufferedInChannel_ms  = SKP_LIMIT_int( psEnc->BufferedInChannel_ms, 0, 100 );
    psEnc->sCmn.prev_nBits = nBits;
    psEnc->sCmn.first_frame_after_reset = 0;

    /****************************************/
    /* Finalize payload                     */
    /****************************************/
    if( ++psEnc->sCmn.nFramesAnalyzed >= psEnc->sCmn.nFramesPerPacket ) {
        /* Insert VAD flags and FEC flag at beginning of bitstream */
        flags = 0;
        for( i = 0; i < psEnc->sCmn.nFramesPerPacket; i++ ) {
            flags |= psEnc->sCmn.VAD_flags[i];
            flags  = SKP_LSHIFT( flags, 1 );
        }
        flags |= psEnc->sCmn.LBRR_flag;
        ec_enc_patch_initial_bits( psRangeEnc, flags, psEnc->sCmn.nFramesPerPacket + 1 );

        /* Payload size */
        nBits = ec_tell( psRangeEnc );
        *pnBytesOut = SKP_RSHIFT( nBits + 7, 3 );

        /* Reset the number of frames in payload buffer */
        psEnc->sCmn.nFramesAnalyzed = 0;
        psEnc->sCmn.prev_nBits = 0;
    } else {
        /* No payload this time */
        *pnBytesOut = 0;
    }
TOC(ENCODE_FRAME)

#ifdef SAVE_ALL_INTERNAL_DATA
    {
        SKP_float tmp[ MAX_NB_SUBFR * LTP_ORDER ];
        int i;
        DEBUG_STORE_DATA( xf.dat,                   x_frame + LA_SHAPE_MS * psEnc->sCmn.fs_kHz, psEnc->sCmn.frame_length    * sizeof( SKP_int16 ) );
        DEBUG_STORE_DATA( xfw.dat,                  xfw,                            psEnc->sCmn.frame_length    * sizeof( SKP_int16 ) );
        DEBUG_STORE_DATA( pitchL.dat,               sEncCtrl.pitchL,                psEnc->sCmn.nb_subfr            * sizeof( SKP_int ) );
        for( i = 0; i < psEnc->sCmn.nb_subfr * LTP_ORDER; i++ ) {
            tmp[ i ] = (SKP_float)sEncCtrl.LTPCoef_Q14[ i ] / 16384.0f;
        }
        DEBUG_STORE_DATA( pitchG_quantized.dat,     tmp,                            psEnc->sCmn.nb_subfr * LTP_ORDER * sizeof( SKP_float ) );
        for( i = 0; i <psEnc->sCmn.predictLPCOrder; i++ ) {
            tmp[ i ] = (SKP_float)sEncCtrl.PredCoef_Q12[ 1 ][ i ] / 4096.0f;
        }
        DEBUG_STORE_DATA( PredCoef.dat,             tmp,                            psEnc->sCmn.predictLPCOrder * sizeof( SKP_float ) );
        
        tmp[ 0 ] = (SKP_float)sEncCtrl.LTPredCodGain_Q7 / 128.0f;
        DEBUG_STORE_DATA( LTPredCodGain.dat,        tmp,                            sizeof( SKP_float ) );
        tmp[ 0 ] = (SKP_float)psEnc->LTPCorr_Q15 / 32768.0f;
        DEBUG_STORE_DATA( LTPcorr.dat,              tmp,                            sizeof( SKP_float ) );
        tmp[ 0 ] = (SKP_float)psEnc->sCmn.input_tilt_Q15 / 32768.0f;
        DEBUG_STORE_DATA( tilt.dat,                 tmp,                            sizeof( SKP_float ) );
        for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
            tmp[ i ] = (SKP_float)sEncCtrl.Gains_Q16[ i ] / 65536.0f;
        }
        DEBUG_STORE_DATA( gains.dat,                tmp,                            psEnc->sCmn.nb_subfr * sizeof( SKP_float ) );
        DEBUG_STORE_DATA( gains_indices.dat,        &psEnc->sCmn.indices.GainsIndices,       psEnc->sCmn.nb_subfr * sizeof( SKP_int ) );
        DEBUG_STORE_DATA( nBits.dat,                &nBits,                         sizeof( SKP_int ) );
        tmp[ 0 ] = (SKP_float)sEncCtrl.current_SNR_dB_Q7 / 128.0f;
        DEBUG_STORE_DATA( current_SNR_db.dat,       tmp,                            sizeof( SKP_float ) );
        DEBUG_STORE_DATA( quantOffsetType.dat,      &psEnc->sCmn.indices.quantOffsetType,    sizeof( SKP_int ) );
        tmp[ 0 ] = (SKP_float)psEnc->sCmn.speech_activity_Q8 / 256.0f;
        DEBUG_STORE_DATA( speech_activity.dat,      tmp,                            sizeof( SKP_float ) );
        for( i = 0; i < VAD_N_BANDS; i++ ) {
            tmp[ i ] = (SKP_float)psEnc->sCmn.input_quality_bands_Q15[ i ] / 32768.0f;
        }
        DEBUG_STORE_DATA( input_quality_bands.dat,  tmp,                            VAD_N_BANDS * sizeof( SKP_float ) );
        DEBUG_STORE_DATA( signalType.dat,           &psEnc->sCmn.indices.signalType,         sizeof( SKP_int8) ); 
        DEBUG_STORE_DATA( lag_index.dat,            &psEnc->sCmn.indices.lagIndex,           sizeof( SKP_int16 ) ); 
        DEBUG_STORE_DATA( contour_index.dat,        &psEnc->sCmn.indices.contourIndex,       sizeof( SKP_int8 ) ); 
        DEBUG_STORE_DATA( per_index.dat,            &psEnc->sCmn.indices.PERIndex,           sizeof( SKP_int8) ); 
    }
#endif
    return ret;
}

/* Low-Bitrate Redundancy (LBRR) encoding. Reuse all parameters but encode excitation at lower bitrate  */
void SKP_Silk_LBRR_encode_FIX(
    SKP_Silk_encoder_state_FIX      *psEnc,         /* I/O  Pointer to Silk FIX encoder state           */
    SKP_Silk_encoder_control_FIX    *psEncCtrl,     /* I/O  Pointer to Silk FIX encoder control struct  */
    const SKP_int16                 xfw[]           /* I    Input signal                                */
)
{
    SKP_int32   TempGains_Q16[ MAX_NB_SUBFR ];
    SideInfoIndices *psIndices_LBRR = &psEnc->sCmn.indices_LBRR[ psEnc->sCmn.nFramesAnalyzed ];
    SKP_Silk_nsq_state sNSQ_LBRR;

    /*******************************************/
    /* Control use of inband LBRR              */
    /*******************************************/
    if( psEnc->sCmn.LBRR_enabled && psEnc->sCmn.speech_activity_Q8 > SKP_FIX_CONST( LBRR_SPEECH_ACTIVITY_THRES, 8 ) ) {
        psEnc->sCmn.LBRR_flags[ psEnc->sCmn.nFramesAnalyzed ] = 1;

        /* Copy noise shaping quantizer state and quantization indices from regular encoding */
        SKP_memcpy( &sNSQ_LBRR, &psEnc->sCmn.sNSQ, sizeof( SKP_Silk_nsq_state ) );
        SKP_memcpy( psIndices_LBRR, &psEnc->sCmn.indices, sizeof( SideInfoIndices ) );

        /* Save original gains */
        SKP_memcpy( TempGains_Q16, psEncCtrl->Gains_Q16, psEnc->sCmn.nb_subfr * sizeof( SKP_int32 ) );

        if( psEnc->sCmn.nFramesAnalyzed == 0 || psEnc->sCmn.LBRR_flags[ psEnc->sCmn.nFramesAnalyzed - 1 ] == 0 ) {
            /* First frame in packet or previous frame not LBRR coded */
            psEnc->sCmn.LBRRprevLastGainIndex = psEnc->sShape.LastGainIndex;

            /* Increase Gains to get target LBRR rate */
            psIndices_LBRR->GainsIndices[ 0 ] = psIndices_LBRR->GainsIndices[ 0 ] + psEnc->sCmn.LBRR_GainIncreases;
            psIndices_LBRR->GainsIndices[ 0 ] = SKP_min_int( psIndices_LBRR->GainsIndices[ 0 ], N_LEVELS_QGAIN - 1 );
        }

        /* Decode to get gains in sync with decoder         */
        /* Overwrite unquantized gains with quantized gains */
        SKP_Silk_gains_dequant( psEncCtrl->Gains_Q16, psIndices_LBRR->GainsIndices, 
            &psEnc->sCmn.LBRRprevLastGainIndex, psEnc->sCmn.nFramesAnalyzed, psEnc->sCmn.nb_subfr );

        /*****************************************/
        /* Noise shaping quantization            */
        /*****************************************/
        if( psEnc->sCmn.nStatesDelayedDecision > 1 || psEnc->sCmn.warping_Q16 > 0 ) {
            SKP_Silk_NSQ_del_dec( &psEnc->sCmn, &sNSQ_LBRR, psIndices_LBRR, xfw, 
                psEnc->sCmn.pulses_LBRR[ psEnc->sCmn.nFramesAnalyzed ], psEncCtrl->PredCoef_Q12[ 0 ], psEncCtrl->LTPCoef_Q14, 
                psEncCtrl->AR2_Q13, psEncCtrl->HarmShapeGain_Q14, psEncCtrl->Tilt_Q14, psEncCtrl->LF_shp_Q14, 
                psEncCtrl->Gains_Q16, psEncCtrl->pitchL, psEncCtrl->Lambda_Q10, psEncCtrl->LTP_scale_Q14 );
        } else {
            SKP_Silk_NSQ( &psEnc->sCmn, &sNSQ_LBRR, psIndices_LBRR, xfw, 
                psEnc->sCmn.pulses_LBRR[ psEnc->sCmn.nFramesAnalyzed ], psEncCtrl->PredCoef_Q12[ 0 ], psEncCtrl->LTPCoef_Q14, 
                psEncCtrl->AR2_Q13, psEncCtrl->HarmShapeGain_Q14, psEncCtrl->Tilt_Q14, psEncCtrl->LF_shp_Q14, 
                psEncCtrl->Gains_Q16, psEncCtrl->pitchL, psEncCtrl->Lambda_Q10, psEncCtrl->LTP_scale_Q14 );
        }

        /* Restore original Gains */
        SKP_memcpy( psEncCtrl->Gains_Q16, TempGains_Q16, psEnc->sCmn.nb_subfr * sizeof( SKP_int32 ) );
    }
}
