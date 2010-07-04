#include <stdlib.h>
#include "SKP_Silk_main_FLP.h"

/****************/
/* Encode frame */
/****************/
SKP_int SKP_Silk_encode_frame_FLP( 
    SKP_Silk_encoder_state_FLP      *psEnc,             /* I/O  Encoder state FLP                       */
    SKP_int16                       *pnBytesOut,        /* I/O  Number of payload bytes;                */
                                                        /*      input: max length; output: used         */
    ec_enc                          *psRangeEnc,        /* I/O  compressor data structure                */
    const SKP_int16                 *pIn                /* I    Input speech frame                      */
)
{
    SKP_Silk_encoder_control_FLP sEncCtrl;
    SKP_int     k, i, nBytes, ret = 0;
    SKP_float   *x_frame, *res_pitch_frame;
    SKP_int16   pIn_HP[    MAX_FRAME_LENGTH ];
    SKP_int16   pIn_HP_LP[ MAX_FRAME_LENGTH ];
    SKP_float   xfw[       MAX_FRAME_LENGTH ];
    SKP_float   res_pitch[ 2 * MAX_FRAME_LENGTH + LA_PITCH_MAX ];
    SKP_int     LBRR_idx, frame_terminator;

    /* Low bitrate redundancy parameters */
    SKP_uint8   LBRRpayload[ MAX_ARITHM_BYTES ];
    SKP_int16   nBytesLBRR;

    const SKP_uint16 *FrameTermination_CDF;

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
    SKP_Silk_VAD_FLP( psEnc, &sEncCtrl, pIn );
TOC(VAD)

    /*******************************************/
    /* High-pass filtering of the input signal */
    /*******************************************/
TIC(HP_IN)
#if HIGH_PASS_INPUT
    /* Variable high-pass filter */
    SKP_Silk_HP_variable_cutoff_FLP( psEnc, &sEncCtrl, pIn_HP, pIn );
#else
    SKP_memcpy( pIn_HP, pIn, psEnc->sCmn.frame_length * sizeof( SKP_int16 ) );
#endif
TOC(HP_IN)

#if SWITCH_TRANSITION_FILTERING
    /* Ensure smooth bandwidth transitions */
    SKP_Silk_LP_variable_cutoff( &psEnc->sCmn.sLP, pIn_HP_LP, pIn_HP, psEnc->sCmn.frame_length );
#else
    SKP_memcpy( pIn_HP_LP, pIn_HP, psEnc->sCmn.frame_length * sizeof( SKP_int16 ) );
#endif

    /*******************************************/
    /* Copy new frame to front of input buffer */
    /*******************************************/
    SKP_short2float_array( x_frame + psEnc->sCmn.la_shape, pIn_HP_LP, psEnc->sCmn.frame_length );

    /* Add tiny signal to avoid high CPU load from denormalized floating point numbers */
    for( k = 0; k < 8; k++ ) {
        x_frame[ psEnc->sCmn.la_shape + k * ( psEnc->sCmn.frame_length >> 3 ) ] += ( 1 - ( k & 2 ) ) * 1e-6f;
    }

    /*****************************************/
    /* Find pitch lags, initial LPC analysis */
    /*****************************************/
TIC(FIND_PITCH)
    SKP_Silk_find_pitch_lags_FLP( psEnc, &sEncCtrl, res_pitch, x_frame );
TOC(FIND_PITCH)

    /************************/
    /* Noise shape analysis */
    /************************/
TIC(NOISE_SHAPE_ANALYSIS)
    SKP_Silk_noise_shape_analysis_FLP( psEnc, &sEncCtrl, res_pitch_frame, x_frame );
TOC(NOISE_SHAPE_ANALYSIS)

    /*****************************************/
    /* Prefiltering for noise shaper         */
    /*****************************************/
TIC(PREFILTER)
    SKP_Silk_prefilter_FLP( psEnc, &sEncCtrl, xfw, x_frame );
TOC(PREFILTER)

    /***************************************************/
    /* Find linear prediction coefficients (LPC + LTP) */
    /***************************************************/
TIC(FIND_PRED_COEF)
    SKP_Silk_find_pred_coefs_FLP( psEnc, &sEncCtrl, res_pitch );
TOC(FIND_PRED_COEF)

    /****************************************/
    /* Process gains                        */
    /****************************************/
TIC(PROCESS_GAINS)
    SKP_Silk_process_gains_FLP( psEnc, &sEncCtrl );
TOC(PROCESS_GAINS)
    
    psEnc->sCmn.sigtype[         psEnc->sCmn.nFramesInPayloadBuf ] = sEncCtrl.sCmn.sigtype;
    psEnc->sCmn.QuantOffsetType[ psEnc->sCmn.nFramesInPayloadBuf ] = sEncCtrl.sCmn.QuantOffsetType;
    /****************************************/
    /* Low Bitrate Redundant Encoding       */
    /****************************************/
    nBytesLBRR = MAX_ARITHM_BYTES;
TIC(LBRR)
    //SKP_Silk_LBRR_encode_FLP( psEnc, &sEncCtrl, LBRRpayload, &nBytesLBRR, xfw );
TOC(LBRR)

    /*****************************************/
    /* Noise shaping quantization            */
    /*****************************************/
TIC(NSQ)
    SKP_Silk_NSQ_wrapper_FLP( psEnc, &sEncCtrl, xfw,
        &psEnc->sCmn.q[ psEnc->sCmn.nFramesInPayloadBuf * psEnc->sCmn.frame_length ], 0 );
TOC(NSQ)

    /**************************************************/
    /* Convert speech activity into VAD and DTX flags */
    /**************************************************/
    if( psEnc->speech_activity < SPEECH_ACTIVITY_DTX_THRES ) {
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
        ( psEnc->sCmn.ltp_mem_length + psEnc->sCmn.la_shape ) * sizeof( SKP_float ) );
    
    /* Parameters needed for next frame */
    psEnc->sCmn.prev_sigtype = sEncCtrl.sCmn.sigtype;
    psEnc->sCmn.prevLag      = sEncCtrl.sCmn.pitchL[ psEnc->sCmn.nb_subfr - 1 ];
    psEnc->sCmn.first_frame_after_reset = 0;

    if( 0 ) { //psEnc->sCmn.sRC.error ) {
        /* Encoder returned error: Clear payload buffer */
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
        frame_terminator = SKP_SILK_LAST_FRAME;
        if( psEnc->sCmn.LBRR_buffer[ LBRR_idx ].usage == SKP_SILK_ADD_LBRR_TO_PLUS1 ) {
            frame_terminator = SKP_SILK_LBRR_VER1;
        }
        if( psEnc->sCmn.LBRR_buffer[ psEnc->sCmn.oldest_LBRR_idx ].usage == SKP_SILK_ADD_LBRR_TO_PLUS2 ) {
            frame_terminator = SKP_SILK_LBRR_VER2;
            LBRR_idx = psEnc->sCmn.oldest_LBRR_idx;
        }

        /* Add the frame termination info to stream */
        ec_encode_bin( psRangeEnc, FrameTermination_CDF[ frame_terminator ], 
            FrameTermination_CDF[ frame_terminator + 1 ], 16 );
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
            ec_enc_done( psRangeEnc );
            
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
            /* The below line describes how FEC should be used */ 
            psEnc->sCmn.LBRR_buffer[ psEnc->sCmn.oldest_LBRR_idx ].usage = sEncCtrl.sCmn.LBRR_usage;
            psEnc->sCmn.oldest_LBRR_idx = ( ( psEnc->sCmn.oldest_LBRR_idx + 1 ) & LBRR_IDX_MASK );

        } else {
            /* Not enough space: Payload will be discarded */
            *pnBytesOut = 0;
            nBytes      = 0;
            ret = SKP_SILK_ENC_PAYLOAD_BUF_TOO_SHORT;
        }

        /* Reset the number of frames in payload buffer */         
        psEnc->sCmn.nFramesInPayloadBuf = 0;
    } else {
        /* No payload for you this time */
        *pnBytesOut = 0;

        /* Encode that more frames follows */
        frame_terminator = SKP_SILK_MORE_FRAMES;
        ec_encode_bin( psRangeEnc, FrameTermination_CDF[ frame_terminator ], 
            FrameTermination_CDF[ frame_terminator + 1 ], 16 );

        /* Payload length so far */
        nBytes = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );

        /* Take into account the q signal that isn't in the bitstream yet */
        nBytes += SKP_Silk_pulses_to_bytes( &psEnc->sCmn, 
            &psEnc->sCmn.q[ psEnc->sCmn.nFramesInPayloadBuf * psEnc->sCmn.frame_length ] );
    }

    /* simulate number of ms buffered in channel because of exceeding TargetRate */
    psEnc->BufferedInChannel_ms   += ( 8.0f * 1000.0f * ( nBytes - psEnc->sCmn.nBytesInPayloadBuf ) ) / psEnc->sCmn.TargetRate_bps;
    psEnc->BufferedInChannel_ms   -= SUB_FRAME_LENGTH_MS * psEnc->sCmn.nb_subfr;
    psEnc->BufferedInChannel_ms    = SKP_LIMIT_float( psEnc->BufferedInChannel_ms, 0.0f, 100.0f );
    psEnc->sCmn.nBytesInPayloadBuf = nBytes;

    if( psEnc->speech_activity > WB_DETECT_ACTIVE_SPEECH_LEVEL_THRES ) {
        psEnc->sCmn.sSWBdetect.ActiveSpeech_ms = SKP_ADD_POS_SAT32( psEnc->sCmn.sSWBdetect.ActiveSpeech_ms, SUB_FRAME_LENGTH_MS * psEnc->sCmn.nb_subfr ); 
    }

TOC(ENCODE_FRAME)
#ifdef SAVE_ALL_INTERNAL_DATA
    DEBUG_STORE_DATA( xf.dat,                   pIn_HP_LP,                           psEnc->sCmn.frame_length * sizeof( SKP_int16 ) );
    DEBUG_STORE_DATA( xfw.dat,                  xfw,                                 psEnc->sCmn.frame_length * sizeof( SKP_float ) );
//  DEBUG_STORE_DATA( q.dat,                    q,                                   psEnc->sCmn.frame_length * sizeof( SKP_int   ) );
    DEBUG_STORE_DATA( pitchL.dat,               sEncCtrl.sCmn.pitchL,                            MAX_NB_SUBFR * sizeof( SKP_int   ) );
    DEBUG_STORE_DATA( pitchG_quantized.dat,     sEncCtrl.LTPCoef,            psEnc->sCmn.nb_subfr * LTP_ORDER * sizeof( SKP_float ) );
    DEBUG_STORE_DATA( pitch_freq_low_Hz.dat,    &sEncCtrl.pitch_freq_low_Hz,                                    sizeof( SKP_float ) );
    DEBUG_STORE_DATA( LTPcorr.dat,              &psEnc->LTPCorr,                                                sizeof( SKP_float ) );
    DEBUG_STORE_DATA( tilt.dat,                 &sEncCtrl.input_tilt,                                           sizeof( SKP_float ) );
    DEBUG_STORE_DATA( gains.dat,                sEncCtrl.Gains,                          psEnc->sCmn.nb_subfr * sizeof( SKP_float ) );
//    DEBUG_STORE_DATA( gains_indices.dat,        sEncCtrl.sCmn.GainsIndices, MAX_LAYERS * psEnc->sCmn.nb_subfr * sizeof( SKP_int   ) );
    DEBUG_STORE_DATA( nBytes.dat,               &nBytes,                                                        sizeof( SKP_int   ) );
    DEBUG_STORE_DATA( current_SNR_db.dat,       &sEncCtrl.current_SNR_dB,                                       sizeof( SKP_float ) );
    DEBUG_STORE_DATA( QuantOffsetType.dat,      &sEncCtrl.sCmn.QuantOffsetType,                                 sizeof( SKP_int   ) );
    DEBUG_STORE_DATA( speech_activity.dat,      &psEnc->speech_activity,                                        sizeof( SKP_float ) );
    DEBUG_STORE_DATA( input_quality_bands.dat,  sEncCtrl.input_quality_bands,                     VAD_N_BANDS * sizeof( SKP_float ) );
    DEBUG_STORE_DATA( sigtype.dat,              &sEncCtrl.sCmn.sigtype,                                         sizeof( SKP_int   ) ); 
    DEBUG_STORE_DATA( ratelevel.dat,            &sEncCtrl.sCmn.RateLevelIndex,                                  sizeof( SKP_int   ) ); 
    DEBUG_STORE_DATA( lag_index.dat,            &sEncCtrl.sCmn.lagIndex,                                        sizeof( SKP_int   ) ); 
    DEBUG_STORE_DATA( contour_index.dat,        &sEncCtrl.sCmn.contourIndex,                                    sizeof( SKP_int   ) ); 
    DEBUG_STORE_DATA( per_index.dat,            &sEncCtrl.sCmn.PERIndex,                                        sizeof( SKP_int   ) );
    DEBUG_STORE_DATA( PredCoef.dat,             &sEncCtrl.PredCoef[ 1 ],          psEnc->sCmn.predictLPCOrder * sizeof( SKP_float ) );
    DEBUG_STORE_DATA( ltp_scale_idx.dat,        &sEncCtrl.sCmn.LTP_scaleIndex,                                  sizeof( SKP_int   ) );
//  DEBUG_STORE_DATA( xq.dat,                   psEnc->sNSQ.xqBuf,                   psEnc->sCmn.frame_length * sizeof( SKP_float ) );
#endif
    return( ret );
}

#if 0  //tmp
/* Low Bitrate Redundancy (LBRR) encoding. Reuse all parameters but encode with lower bitrate           */
void SKP_Silk_LBRR_encode_FLP(
    SKP_Silk_encoder_state_FLP      *psEnc,             /* I/O  Encoder state FLP                       */
    SKP_Silk_encoder_control_FLP    *psEncCtrl,         /* I/O  Encoder control FLP                     */
          SKP_uint8                 *pCode,             /* O    Payload                                 */
          SKP_int16                 *pnBytesOut,        /* I/O  Payload bytes; in: max; out: used       */
    const SKP_float                 xfw[]               /* I    Input signal                            */
)
{
    SKP_int32   Gains_Q16[ MAX_NB_SUBFR ];
    SKP_int     i, k, TempGainsIndices[ MAX_NB_SUBFR ], frame_terminator;
    SKP_int     nBytes, nFramesInPayloadBuf;
    SKP_float   TempGains[ MAX_NB_SUBFR ];
    SKP_int     typeOffset, LTP_scaleIndex, Rate_only_parameters = 0;
    ec_byte_buffer range_enc_celt_buf;

    /* Control use of inband LBRR */
    SKP_Silk_LBRR_ctrl_FLP( psEnc, &psEncCtrl->sCmn );

    if( psEnc->sCmn.LBRR_enabled ) {
        /* Save original gains */
        SKP_memcpy( TempGainsIndices, psEncCtrl->sCmn.GainsIndices, MAX_NB_SUBFR * sizeof( SKP_int   ) );
        SKP_memcpy( TempGains,        psEncCtrl->Gains,             MAX_NB_SUBFR * sizeof( SKP_float ) );

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
                /* First frame in packet copy everything */
                SKP_memcpy( &psEnc->sNSQ_LBRR, &psEnc->sNSQ, sizeof( SKP_Silk_nsq_state ) );
                psEnc->sCmn.LBRRprevLastGainIndex = psEnc->sShape.LastGainIndex;
                /* Increase Gains to get target LBRR rate */
                psEncCtrl->sCmn.GainsIndices[ 0 ] += psEnc->sCmn.LBRR_GainIncreases;
                psEncCtrl->sCmn.GainsIndices[ 0 ]  = SKP_LIMIT( psEncCtrl->sCmn.GainsIndices[ 0 ], 0, N_LEVELS_QGAIN - 1 );
            }
            /* Decode to get Gains in sync with decoder */
            SKP_Silk_gains_dequant( Gains_Q16, psEncCtrl->sCmn.GainsIndices, 
                &psEnc->sCmn.LBRRprevLastGainIndex, psEnc->sCmn.nFramesInPayloadBuf, psEnc->sCmn.nb_subfr );

            /* Overwrite unquantized gains with quantized gains and convert back to Q0 from Q16 */
            for( k = 0; k <  psEnc->sCmn.nb_subfr; k++ ) {
                psEncCtrl->Gains[ k ] = Gains_Q16[ k ] / 65536.0f;
            }

            /*****************************************/
            /* Noise shaping quantization            */
            /*****************************************/
            SKP_Silk_NSQ_wrapper_FLP( psEnc, psEncCtrl, xfw, 
                &psEnc->sCmn.q_LBRR[ psEnc->sCmn.nFramesInPayloadBuf * psEnc->sCmn.frame_length ], 1 );
        } else {
            SKP_memset( &psEnc->sCmn.q_LBRR[ psEnc->sCmn.nFramesInPayloadBuf * psEnc->sCmn.frame_length ], 0, 
                psEnc->sCmn.frame_length * sizeof( SKP_int ) );

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
        SKP_Silk_encode_parameters( &psEnc->sCmn, &psEncCtrl->sCmn, &psEnc->sCmn.sRC_LBRR );
        
        /****************************************/
        /* Encode Parameters                    */
        /****************************************/
        if( psEnc->sCmn.sRC_LBRR.error ) {
            /* Encoder returned error: Clear payload buffer */
            nFramesInPayloadBuf = 0;
        } else {
            nFramesInPayloadBuf = psEnc->sCmn.nFramesInPayloadBuf + 1;
        }

        /****************************************/
        /* Finalize payload and copy to output  */
        /****************************************/
        if( psEnc->sCmn.nFramesInPayloadBuf * SUB_FRAME_LENGTH_MS * psEnc->sCmn.nb_subfr >= psEnc->sCmn.PacketSize_ms ) {

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
                /* Not enough space: Payload will be discarded */
                *pnBytesOut = 0;
                SKP_assert( 0 );
            }
        } else {
            /* No payload for you this time */
            *pnBytesOut = 0;

            /* Encode that more frames follows */
            frame_terminator = SKP_SILK_MORE_FRAMES;
            ec_encode_bin( psRangeEnc_LBRR, FrameTermination_CDF[ frame_terminator ], 
                FrameTermination_CDF[ frame_terminator + 1 ], 16 );
        }

        /* Restore original Gains */
        SKP_memcpy( psEncCtrl->sCmn.GainsIndices, TempGainsIndices,  psEnc->sCmn.nb_subfr * sizeof( SKP_int   ) );
        SKP_memcpy( psEncCtrl->Gains,             TempGains,         psEnc->sCmn.nb_subfr * sizeof( SKP_float ) );
    
        /* Restore LTP scale index and typeoffset */
        psEncCtrl->sCmn.LTP_scaleIndex = LTP_scaleIndex;
        psEnc->sCmn.typeOffsetPrev     = typeOffset;
    }
}
#endif