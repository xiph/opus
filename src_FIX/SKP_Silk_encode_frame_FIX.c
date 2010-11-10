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
#include "SKP_Silk_tuning_parameters.h"

/****************/
/* Encode frame */
/****************/
SKP_int SKP_Silk_encode_frame_FIX( 
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O  Encoder state FIX                       */
    SKP_int32                       *pnBytesOut,        /* I/O  Number of payload bytes                 */
                                                        /*      input: max length; output: used         */
    ec_enc                          *psRangeEnc,        /* I/O  compressor data structure               */
    const SKP_int16                 *pIn                /* I    Input speech frame                      */
)
{
    SKP_Silk_encoder_control_FIX sEncCtrl;
    SKP_int     i, nBytes, ret = 0;
    SKP_int16   *x_frame, *res_pitch_frame;
    SKP_int16   xfw[ MAX_FRAME_LENGTH ];
    SKP_int16   pIn_HP[ MAX_FRAME_LENGTH ];
    SKP_int16   res_pitch[ 2 * MAX_FRAME_LENGTH + LA_PITCH_MAX ];
    SKP_int     LBRR_idx, frame_terminator, SNR_dB_Q7;
    const SKP_uint16 *FrameTermination_CDF;

    /* Low bitrate redundancy parameters */
    SKP_uint8   LBRRpayload[ MAX_ARITHM_BYTES ];
    SKP_int32   nBytesLBRR;

TIC(ENCODE_FRAME)

    sEncCtrl.sCmn.Seed = psEnc->sCmn.frameCounter++ & 3;
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
    ret = SKP_Silk_VAD_GetSA_Q8( &psEnc->sCmn.sVAD, &psEnc->speech_activity_Q8, &SNR_dB_Q7, 
                                 sEncCtrl.input_quality_bands_Q15, &sEncCtrl.input_tilt_Q15,
                                 pIn, psEnc->sCmn.frame_length, psEnc->sCmn.fs_kHz );
TOC(VAD)

    /*******************************************/
    /* High-pass filtering of the input signal */
    /*******************************************/
TIC(HP_IN)
#if HIGH_PASS_INPUT
    /* Variable high-pass filter */
    SKP_Silk_HP_variable_cutoff_FIX( psEnc, &sEncCtrl, pIn_HP, pIn );
#else
    SKP_memcpy( pIn_HP, pIn, psEnc->sCmn.frame_length * sizeof( SKP_int16 ) );
#endif
TOC(HP_IN)

#if SWITCH_TRANSITION_FILTERING
    /* Ensure smooth bandwidth transitions */
    SKP_Silk_LP_variable_cutoff( &psEnc->sCmn.sLP, x_frame + LA_SHAPE_MS * psEnc->sCmn.fs_kHz, pIn_HP, psEnc->sCmn.frame_length );
#else
    SKP_memcpy( x_frame + LA_SHAPE_MS * psEnc->sCmn.fs_kHz, pIn_HP,psEnc->sCmn.frame_length * sizeof( SKP_int16 ) );
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
    SKP_Silk_find_pred_coefs_FIX( psEnc, &sEncCtrl, res_pitch );
TOC(FIND_PRED_COEF)

    /****************************************/
    /* Process gains                        */
    /****************************************/
TIC(PROCESS_GAINS)
    SKP_Silk_process_gains_FIX( psEnc, &sEncCtrl );
TOC(PROCESS_GAINS)
    
    psEnc->sCmn.sigtype[         psEnc->sCmn.nFramesInPayloadBuf ] = sEncCtrl.sCmn.sigtype;
    psEnc->sCmn.QuantOffsetType[ psEnc->sCmn.nFramesInPayloadBuf ] = sEncCtrl.sCmn.QuantOffsetType;

    /****************************************/
    /* Low Bitrate Redundant Encoding       */
    /****************************************/
    nBytesLBRR = MAX_ARITHM_BYTES;
TIC(LBRR)
    SKP_Silk_LBRR_encode_FIX( psEnc, &sEncCtrl, LBRRpayload, &nBytesLBRR, xfw );
TOC(LBRR)

    /*****************************************/
    /* Noise shaping quantization            */
    /*****************************************/
TIC(NSQ)
    if( psEnc->sCmn.nStatesDelayedDecision > 1 || psEnc->sCmn.warping_Q16 > 0 ) {
        SKP_Silk_NSQ_del_dec( &psEnc->sCmn, &sEncCtrl.sCmn, &psEnc->sNSQ, xfw,
            psEnc->sCmn.q, sEncCtrl.sCmn.NLSFInterpCoef_Q2, 
            sEncCtrl.PredCoef_Q12[ 0 ], sEncCtrl.LTPCoef_Q14, sEncCtrl.AR2_Q13, sEncCtrl.HarmShapeGain_Q14, 
            sEncCtrl.Tilt_Q14, sEncCtrl.LF_shp_Q14, sEncCtrl.Gains_Q16, sEncCtrl.Lambda_Q10, 
            sEncCtrl.LTP_scale_Q14 );
    } else {
        SKP_Silk_NSQ( &psEnc->sCmn, &sEncCtrl.sCmn, &psEnc->sNSQ, xfw, 
            psEnc->sCmn.q, sEncCtrl.sCmn.NLSFInterpCoef_Q2, 
            sEncCtrl.PredCoef_Q12[ 0 ], sEncCtrl.LTPCoef_Q14, sEncCtrl.AR2_Q13, sEncCtrl.HarmShapeGain_Q14, 
            sEncCtrl.Tilt_Q14, sEncCtrl.LF_shp_Q14, sEncCtrl.Gains_Q16, sEncCtrl.Lambda_Q10, 
            sEncCtrl.LTP_scale_Q14 );
    }
TOC(NSQ)

    /**************************************************/
    /* Convert speech activity into VAD and DTX flags */
    /**************************************************/
    if( psEnc->speech_activity_Q8 < SKP_FIX_CONST( SPEECH_ACTIVITY_DTX_THRES, 8 ) ) {
        psEnc->sCmn.vadFlag = NO_VOICE_ACTIVITY;
        psEnc->sCmn.noSpeechCounter++;
        if( psEnc->sCmn.noSpeechCounter > NO_SPEECH_FRAMES_BEFORE_DTX ) {
            psEnc->sCmn.inDTX = 1;
        }
        if( psEnc->sCmn.noSpeechCounter > MAX_CONSECUTIVE_DTX ) {
            psEnc->sCmn.noSpeechCounter = 0;
            psEnc->sCmn.inDTX           = 0;
        }
    } else {
        psEnc->sCmn.noSpeechCounter = 0;
        psEnc->sCmn.inDTX           = 0;
        psEnc->sCmn.vadFlag         = VOICE_ACTIVITY;
    }

    /****************************************/
    /* Initialize range coder               */
    /****************************************/
    if( psEnc->sCmn.nFramesInPayloadBuf == 0 ) {
        psEnc->sCmn.nBytesInPayloadBuf = 0;
    }

    /****************************************/
    /* Encode Parameters                    */
    /****************************************/
TIC(ENCODE_PARAMS)
    SKP_Silk_encode_parameters( &psEnc->sCmn, &sEncCtrl.sCmn, psRangeEnc );
    FrameTermination_CDF = SKP_Silk_FrameTermination_CDF;
TOC(ENCODE_PARAMS)

    /****************************************/
    /* Update Buffers and State             */
    /****************************************/
    /* Update input buffer */
    SKP_memmove( psEnc->x_buf, &psEnc->x_buf[ psEnc->sCmn.frame_length ], 
        ( psEnc->sCmn.ltp_mem_length + LA_SHAPE_MS * psEnc->sCmn.fs_kHz ) * sizeof( SKP_int16 ) );
    
    /* Parameters needed for next frame */
    psEnc->sCmn.prev_sigtype            = sEncCtrl.sCmn.sigtype;
    psEnc->sCmn.prevLag                 = sEncCtrl.sCmn.pitchL[ psEnc->sCmn.nb_subfr - 1 ];
    psEnc->sCmn.first_frame_after_reset = 0;

    if( 0 ) { //psEnc->sCmn.sRC.error ) {
        /* Encoder returned error: clear payload buffer */
        psEnc->sCmn.nFramesInPayloadBuf = 0;
    } else {
        psEnc->sCmn.nFramesInPayloadBuf++;
    }

    /****************************************/
    /* Finalize payload and copy to output  */
    /****************************************/
    if( psEnc->sCmn.nFramesInPayloadBuf * SUB_FRAME_LENGTH_MS * psEnc->sCmn.nb_subfr >= psEnc->sCmn.PacketSize_ms ) {

        LBRR_idx = ( psEnc->sCmn.oldest_LBRR_idx + 1 ) & LBRR_IDX_MASK;

        /* Check if FEC information should be added */
        //frame_terminator = psEnc->sCmn.LBRR_buffer[ LBRR_idx ].usage;
        frame_terminator = SKP_SILK_NO_LBRR;

        /* Add the frame termination info to stream */
        ec_encode_bin( psRangeEnc, FrameTermination_CDF[ frame_terminator ], 
            FrameTermination_CDF[ frame_terminator + 1 ], 16 );

        /* Code excitation signal */
        for( i = 0; i < psEnc->sCmn.nFramesInPayloadBuf; i++ ) {
            SKP_Silk_encode_pulses( psRangeEnc, psEnc->sCmn.sigtype[ i ], psEnc->sCmn.QuantOffsetType[ i ], 
                &psEnc->sCmn.q[ i * psEnc->sCmn.frame_length ], psEnc->sCmn.frame_length );
        }

        /* Payload length so far */
        nBytes = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );

        /* Check that there is enough space in external output buffer, and move data */
        if( *pnBytesOut >= nBytes ) {
            //SKP_int bits_in_stream, mask;
            //bits_in_stream = ec_enc_tell( psRangeEnc, 0 );
            //ec_enc_done( psRangeEnc );
            
#if 0
            /* Fill up any remaining bits in the last byte with 1s */
            if( bits_in_stream & 7 ) {
                mask = SKP_RSHIFT( 0xFF, bits_in_stream & 7 );
                if( nBytes - 1 < *pnBytesOut ) {
                    psEnc->sCmn.sRC.range_enc_celt_state.buf->buf[ nBytes - 1 ] |= mask;
                }
            }
            SKP_memcpy( pCode, psEnc->sCmn.sRC.range_enc_celt_state.buf->buf, nBytes * sizeof( SKP_uint8 ) );
#endif

#if 0
            if( frame_terminator > SKP_SILK_MORE_FRAMES && 
                    *pnBytesOut >= nBytes + psEnc->sCmn.LBRR_buffer[ LBRR_idx ].nBytes ) {
                /* Get old packet and add to payload. */
                SKP_memcpy( &pCode[ nBytes ],
                    psEnc->sCmn.LBRR_buffer[ LBRR_idx ].payload,
                    psEnc->sCmn.LBRR_buffer[ LBRR_idx ].nBytes * sizeof( SKP_uint8 ) );
                nBytes += psEnc->sCmn.LBRR_buffer[ LBRR_idx ].nBytes;
            }
#endif
            *pnBytesOut = nBytes;

            /* Update FEC buffer */
            SKP_memcpy( psEnc->sCmn.LBRR_buffer[ psEnc->sCmn.oldest_LBRR_idx ].payload, LBRRpayload, 
                nBytesLBRR * sizeof( SKP_uint8 ) );
            psEnc->sCmn.LBRR_buffer[ psEnc->sCmn.oldest_LBRR_idx ].nBytes = nBytesLBRR;
            /* The line below describes how FEC should be used */
            psEnc->sCmn.LBRR_buffer[ psEnc->sCmn.oldest_LBRR_idx ].usage = sEncCtrl.sCmn.LBRR_usage;
            psEnc->sCmn.oldest_LBRR_idx = ( psEnc->sCmn.oldest_LBRR_idx + 1 ) & LBRR_IDX_MASK;

        } else {
            /* Not enough space: Payload will be discarded */
            *pnBytesOut = 0;
            nBytes      = 0;
            ret = SKP_SILK_ENC_PAYLOAD_BUF_TOO_SHORT;
        }

        /* Reset the number of frames in payload buffer */
        psEnc->sCmn.nFramesInPayloadBuf = 0;
    } else {
        /* No payload this time */
        *pnBytesOut = 0;

        /* Payload length so far */
        nBytes = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );

        /* Take into account the q signal that isn't in the bitstream yet */
        nBytes += SKP_Silk_pulses_to_bytes( &psEnc->sCmn, 
            &psEnc->sCmn.q[ psEnc->sCmn.nFramesInPayloadBuf * psEnc->sCmn.frame_length ] );
    }

    /* Check for arithmetic coder errors */
    if( 0 ) { //psEnc->sCmn.sRC.error ) {
        ret = SKP_SILK_ENC_INTERNAL_ERROR;
    }

    /* Simulate number of ms buffered in channel because of exceeding TargetRate */
    SKP_assert(  ( 8 * 1000 * ( (SKP_int64)nBytes - (SKP_int64)psEnc->sCmn.nBytesInPayloadBuf ) ) == 
        SKP_SAT32( 8 * 1000 * ( (SKP_int64)nBytes - (SKP_int64)psEnc->sCmn.nBytesInPayloadBuf ) ) );
    SKP_assert( psEnc->sCmn.TargetRate_bps > 0 );
    psEnc->BufferedInChannel_ms   += SKP_DIV32( 8 * 1000 * ( nBytes - psEnc->sCmn.nBytesInPayloadBuf ), psEnc->sCmn.TargetRate_bps );
    psEnc->BufferedInChannel_ms   -= SKP_SMULBB( SUB_FRAME_LENGTH_MS, psEnc->sCmn.nb_subfr );
    psEnc->BufferedInChannel_ms    = SKP_LIMIT_int( psEnc->BufferedInChannel_ms, 0, 100 );
    psEnc->sCmn.nBytesInPayloadBuf = nBytes;

    if( psEnc->speech_activity_Q8 > SKP_FIX_CONST( WB_DETECT_ACTIVE_SPEECH_LEVEL_THRES, 8 ) ) {
        psEnc->sCmn.sSWBdetect.ActiveSpeech_ms = SKP_ADD_POS_SAT32( psEnc->sCmn.sSWBdetect.ActiveSpeech_ms, SKP_SMULBB( SUB_FRAME_LENGTH_MS, psEnc->sCmn.nb_subfr ) ); 
    }

TOC(ENCODE_FRAME)

#ifdef SAVE_ALL_INTERNAL_DATA
    {
        SKP_float tmp[ MAX_NB_SUBFR * LTP_ORDER ];
        int i;
        DEBUG_STORE_DATA( xf.dat,                   x_frame + LA_SHAPE_MS * psEnc->sCmn.fs_kHz, psEnc->sCmn.frame_length    * sizeof( SKP_int16 ) );
        DEBUG_STORE_DATA( xfw.dat,                  xfw,                            psEnc->sCmn.frame_length    * sizeof( SKP_int16 ) );
        //	DEBUG_STORE_DATA( q.dat,                    &psEnc->sCmn.q[ ( psEnc->sCmn.nFramesInPayloadBuf - 1)*psEnc->sCmn.frame_length ],  psEnc->sCmn.frame_length    * sizeof( SKP_int8 ) );
        DEBUG_STORE_DATA( pitchL.dat,               sEncCtrl.sCmn.pitchL,           psEnc->sCmn.nb_subfr            * sizeof( SKP_int ) );
        for( i = 0; i < psEnc->sCmn.nb_subfr * LTP_ORDER; i++ ) {
            tmp[ i ] = (SKP_float)sEncCtrl.LTPCoef_Q14[ i ] / 16384.0f;
        }
        DEBUG_STORE_DATA( pitchG_quantized.dat,     tmp,                            psEnc->sCmn.nb_subfr * LTP_ORDER * sizeof( SKP_float ) );
        for( i = 0; i <psEnc->sCmn.predictLPCOrder; i++ ) {
            tmp[ i ] = (SKP_float)sEncCtrl.PredCoef_Q12[ 1 ][ i ] / 4096.0f;
        }
        DEBUG_STORE_DATA( PredCoef.dat,             tmp,                            psEnc->sCmn.predictLPCOrder * sizeof( SKP_float ) );
        
        tmp[ 0 ] = (SKP_float)sEncCtrl.pitch_freq_low_Hz;
        DEBUG_STORE_DATA( pitch_freq_low_Hz.dat,    tmp,                            sizeof( SKP_float ) );
        tmp[ 0 ] = (SKP_float)sEncCtrl.LTPredCodGain_Q7 / 128.0f;
        DEBUG_STORE_DATA( LTPredCodGain.dat,        tmp,                            sizeof( SKP_float ) );
        tmp[ 0 ] = (SKP_float)psEnc->LTPCorr_Q15 / 32768.0f;
        DEBUG_STORE_DATA( LTPcorr.dat,              tmp,                            sizeof( SKP_float ) );
        tmp[ 0 ] = (SKP_float)sEncCtrl.input_tilt_Q15 / 32768.0f;
        DEBUG_STORE_DATA( tilt.dat,                 tmp,                            sizeof( SKP_float ) );
        for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
            tmp[ i ] = (SKP_float)sEncCtrl.Gains_Q16[ i ] / 65536.0f;
        }
        DEBUG_STORE_DATA( gains.dat,                tmp,                            psEnc->sCmn.nb_subfr * sizeof( SKP_float ) );
        DEBUG_STORE_DATA( gains_indices.dat,        &sEncCtrl.sCmn.GainsIndices,    psEnc->sCmn.nb_subfr * sizeof( SKP_int ) );
        DEBUG_STORE_DATA( nBytes.dat,               &nBytes,                        sizeof( SKP_int ) );
        tmp[ 0 ] = (SKP_float)sEncCtrl.current_SNR_dB_Q7 / 128.0f;
        DEBUG_STORE_DATA( current_SNR_db.dat,       tmp,                            sizeof( SKP_float ) );
        DEBUG_STORE_DATA( QuantOffsetType.dat,      &sEncCtrl.sCmn.QuantOffsetType, sizeof( SKP_int ) );
        tmp[ 0 ] = (SKP_float)psEnc->speech_activity_Q8 / 256.0f;
        DEBUG_STORE_DATA( speech_activity.dat,      tmp,                            sizeof( SKP_float ) );
        for( i = 0; i < VAD_N_BANDS; i++ ) {
            tmp[ i ] = (SKP_float)sEncCtrl.input_quality_bands_Q15[ i ] / 32768.0f;
        }
        DEBUG_STORE_DATA( input_quality_bands.dat,  tmp,                            VAD_N_BANDS * sizeof( SKP_float ) );
        DEBUG_STORE_DATA( sigtype.dat,              &sEncCtrl.sCmn.sigtype,         sizeof( SKP_int ) ); 
        DEBUG_STORE_DATA( ratelevel.dat,            &sEncCtrl.sCmn.RateLevelIndex,  sizeof( SKP_int ) ); 
        DEBUG_STORE_DATA( lag_index.dat,            &sEncCtrl.sCmn.lagIndex,        sizeof( SKP_int ) ); 
        DEBUG_STORE_DATA( contour_index.dat,        &sEncCtrl.sCmn.contourIndex,    sizeof( SKP_int ) ); 
        DEBUG_STORE_DATA( per_index.dat,            &sEncCtrl.sCmn.PERIndex,        sizeof( SKP_int ) ); 
    }
#endif
    return( ret );
}

#if 0  //tmp
/* Low BitRate Redundancy encoding functionality. Reuse all parameters but encode residual with lower bitrate */
void SKP_Silk_LBRR_encode_FIX(
    SKP_Silk_encoder_state_FIX      *psEnc,         /* I/O  Pointer to Silk encoder state           */
    SKP_Silk_encoder_control_FIX    *psEncCtrl,     /* I/O  Pointer to Silk encoder control struct  */
    SKP_uint8                       *pCode,         /* O    Pointer to payload                      */
    SKP_int16                       *pnBytesOut,    /* I/O  Pointer to number of payload bytes      */
    SKP_int16                       xfw[]           /* I    Input signal                            */
)
{
    SKP_int     i, TempGainsIndices[ MAX_NB_SUBFR ], frame_terminator;
    SKP_int     nBytes, nFramesInPayloadBuf;
    SKP_int32   TempGains_Q16[ MAX_NB_SUBFR ];
    SKP_int     typeOffset, LTP_scaleIndex, Rate_only_parameters = 0;
    ec_byte_buffer range_enc_celt_buf;

    /*******************************************/
    /* Control use of inband LBRR              */
    /*******************************************/
    SKP_Silk_LBRR_ctrl_FIX( psEnc, &psEncCtrl->sCmn );

    if( psEnc->sCmn.LBRR_enabled ) {
        /* Save original gains */
        SKP_memcpy( TempGainsIndices, psEncCtrl->sCmn.GainsIndices, MAX_NB_SUBFR * sizeof( SKP_int   ) );
        SKP_memcpy( TempGains_Q16,    psEncCtrl->Gains_Q16,         MAX_NB_SUBFR * sizeof( SKP_int32 ) );

        typeOffset     = psEnc->sCmn.typeOffsetPrev; // Temp save as cannot be overwritten
        LTP_scaleIndex = psEncCtrl->sCmn.LTP_scaleIndex;

        /* Set max rate where quant signal is encoded */
        if( psEnc->sCmn.fs_kHz == 8 ) {
            Rate_only_parameters = 13500;
        } else if( psEnc->sCmn.fs_kHz == 12 ) {
            Rate_only_parameters = 15500;
        } else if( psEnc->sCmn.fs_kHz == 16 ) {
            Rate_only_parameters = 17500;
        } else if( psEnc->sCmn.fs_kHz == 24 ) {
            Rate_only_parameters = 19500;
        } else {
            SKP_assert( 0 );
        }

        if( psEnc->sCmn.Complexity > 0 && psEnc->sCmn.TargetRate_bps > Rate_only_parameters ) {
            if( psEnc->sCmn.nFramesInPayloadBuf == 0 ) {
                /* First frame in packet; copy everything */
                SKP_memcpy( &psEnc->sNSQ_LBRR, &psEnc->sNSQ, sizeof( SKP_Silk_nsq_state ) );

                psEnc->sCmn.LBRRprevLastGainIndex = psEnc->sShape.LastGainIndex;
                /* Increase Gains to get target LBRR rate */
                psEncCtrl->sCmn.GainsIndices[ 0 ] = psEncCtrl->sCmn.GainsIndices[ 0 ] + psEnc->sCmn.LBRR_GainIncreases;
                psEncCtrl->sCmn.GainsIndices[ 0 ] = SKP_LIMIT_int( psEncCtrl->sCmn.GainsIndices[ 0 ], 0, N_LEVELS_QGAIN - 1 );
            }
            /* Decode to get gains in sync with decoder         */
            /* Overwrite unquantized gains with quantized gains */
            SKP_Silk_gains_dequant( psEncCtrl->Gains_Q16, psEncCtrl->sCmn.GainsIndices, 
                &psEnc->sCmn.LBRRprevLastGainIndex, psEnc->sCmn.nFramesInPayloadBuf, psEnc->sCmn.nb_subfr );

            /*****************************************/
            /* Noise shaping quantization            */
            /*****************************************/
            if( psEnc->sCmn.nStatesDelayedDecision > 1 || psEnc->sCmn.warping_Q16 > 0 ) {
                SKP_Silk_NSQ_del_dec( &psEnc->sCmn, &psEncCtrl->sCmn, &psEnc->sNSQ_LBRR, xfw, psEnc->sCmn.q_LBRR, 
                    psEncCtrl->sCmn.NLSFInterpCoef_Q2, psEncCtrl->PredCoef_Q12[ 0 ], psEncCtrl->LTPCoef_Q14, 
                    psEncCtrl->AR2_Q13, psEncCtrl->HarmShapeGain_Q14, psEncCtrl->Tilt_Q14, psEncCtrl->LF_shp_Q14, 
                    psEncCtrl->Gains_Q16, psEncCtrl->Lambda_Q10, psEncCtrl->LTP_scale_Q14 );
            } else {
                SKP_Silk_NSQ( &psEnc->sCmn, &psEncCtrl->sCmn, &psEnc->sNSQ_LBRR, xfw, psEnc->sCmn.q_LBRR, 
                    psEncCtrl->sCmn.NLSFInterpCoef_Q2, psEncCtrl->PredCoef_Q12[ 0 ], psEncCtrl->LTPCoef_Q14, 
                    psEncCtrl->AR2_Q13, psEncCtrl->HarmShapeGain_Q14, psEncCtrl->Tilt_Q14, psEncCtrl->LF_shp_Q14, 
                    psEncCtrl->Gains_Q16, psEncCtrl->Lambda_Q10, psEncCtrl->LTP_scale_Q14 );
            }
        } else {
            SKP_memset( psEnc->sCmn.q_LBRR, 0, psEnc->sCmn.frame_length * sizeof( SKP_int8 ) );
            psEncCtrl->sCmn.LTP_scaleIndex = 0;
        }
        /****************************************/
        /* Initialize arithmetic coder          */
        /****************************************/
        if( psEnc->sCmn.nFramesInPayloadBuf == 0 ) {
            ec_byte_writeinit_buffer( &range_enc_celt_buf, psEnc->sCmn.sRC_LBRR.buffer, MAX_ARITHM_BYTES );
            ec_enc_init( &psEnc->sCmn.sRC_LBRR.range_enc_celt_state, &range_enc_celt_buf );

            SKP_Silk_range_enc_init( &psEnc->sCmn.sRC_LBRR );
            psEnc->sCmn.nBytesInPayloadBuf = 0;
        }

        /****************************************/
        /* Encode Parameters                    */
        /****************************************/
        SKP_Silk_encode_parameters( &psEnc->sCmn, &psEncCtrl->sCmn, 
            &psEnc->sCmn.sRC_LBRR );

        /****************************************/
        /* Encode Parameters                    */
        /****************************************/
        if( psEnc->sCmn.sRC_LBRR.error ) {
            /* Encoder returned error: clear payload buffer */
            nFramesInPayloadBuf = 0;
        } else {
            nFramesInPayloadBuf = psEnc->sCmn.nFramesInPayloadBuf + 1;
        }

        /****************************************/
        /* Finalize payload and copy to output  */
        /****************************************/
        if( SKP_SMULBB( nFramesInPayloadBuf, SKP_SMULBB( SUB_FRAME_LENGTH_MS, psEnc->sCmn.nb_subfr ) ) >= psEnc->sCmn.PacketSize_ms ) {

            /* Check if FEC information should be added */
            frame_terminator = SKP_SILK_LAST_FRAME;

            /* Add the frame termination info to stream */
            ec_encode_bin( psRangeEnc_LBRR, FrameTermination_CDF[ frame_terminator ], 
                FrameTermination_CDF[ frame_terminator + 1 ], 16 );

            /*********************************************/
            /* Encode quantization indices of excitation */
            /*********************************************/
            for( i = 0; i < nFramesInPayloadBuf; i++ ) {
                SKP_Silk_encode_pulses( &psEnc->sCmn.sRC_LBRR, psEnc->sCmn.sigtype[ i ], psEnc->sCmn.QuantOffsetType[ i ],
                    &psEnc->sCmn.q_LBRR[ i * psEnc->sCmn.frame_length ], psEnc->sCmn.frame_length );
            }

            /* Payload length so far */
            nBytes = SKP_RSHIFT( ec_enc_tell( psRangeEnc_LBRR, 0 ) + 7, 3 );

            /* Check that there is enough space in external output buffer and move data */
            if( *pnBytesOut >= nBytes ) {
                SKP_int bits_in_stream, mask;
                bits_in_stream = ec_enc_tell( &psEnc->sCmn.sRC_LBRR.range_enc_celt_state, 0 );
                ec_enc_done( &psEnc->sCmn.sRC_LBRR.range_enc_celt_state );

                /* Fill up any remaining bits in the last byte with 1s */
                if( bits_in_stream & 7 ) {
                    mask = SKP_RSHIFT( 0xFF, bits_in_stream & 7 );
                    if( nBytes - 1 < *pnBytesOut ) {
                        psEnc->sCmn.sRC_LBRR.range_enc_celt_state.buf->buf[ nBytes - 1 ] |= mask;
                    }
                }
                SKP_memcpy( pCode, psEnc->sCmn.sRC_LBRR.range_enc_celt_state.buf->buf, nBytes * sizeof( SKP_uint8 ) );

                *pnBytesOut = nBytes;
            } else {
                /* Not enough space: payload will be discarded */
                *pnBytesOut = 0;
                SKP_assert( 0 );
            }
        } else {
            /* No payload this time */
            *pnBytesOut = 0;

            /* Encode that more frames follows */
            frame_terminator = SKP_SILK_MORE_FRAMES;
            ec_encode_bin( psRangeEnc_LBRR, FrameTermination_CDF[ frame_terminator ], 
                FrameTermination_CDF[ frame_terminator + 1 ], 16 );
        }

        /* Restore original Gains */
        SKP_memcpy( psEncCtrl->sCmn.GainsIndices, TempGainsIndices, psEnc->sCmn.nb_subfr * sizeof( SKP_int   ) );
        SKP_memcpy( psEncCtrl->Gains_Q16,         TempGains_Q16,    psEnc->sCmn.nb_subfr * sizeof( SKP_int32 ) );
    
        /* Restore LTP scale index and typeoffset */
        psEncCtrl->sCmn.LTP_scaleIndex = LTP_scaleIndex;
        psEnc->sCmn.typeOffsetPrev     = typeOffset;
    }
}
#endif
