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

/* ToDo: Move the functions belowto common to be able to use them in FLP control codec also */
SKP_INLINE SKP_int SKP_Silk_setup_resamplers(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         API_fs_Hz,          /* I                        */
    SKP_int                         fs_kHz              /* I                        */
);

SKP_INLINE SKP_int SKP_Silk_setup_packetsize(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         PacketSize_ms,      /* I                        */
    SKP_int                         fs_kHz              /* I                        */
);

SKP_INLINE SKP_int SKP_Silk_setup_fs(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         fs_kHz              /* I                        */
);

SKP_INLINE SKP_int SKP_Silk_setup_complexity(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         Complexity          /* I                        */
);

SKP_INLINE SKP_int SKP_Silk_setup_rate(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         TargetRate_bps      /* I                        */
);

SKP_INLINE SKP_int SKP_Silk_setup_LBRR(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         INBandFEC_enabled   /* I                        */
);

/* Control encoder SNR */
SKP_int SKP_Silk_control_encoder_FIX( 
    SKP_Silk_encoder_state_FIX  *psEnc,             /* I/O  Pointer to Silk encoder state                   */
    const SKP_int32             API_fs_Hz,          /* I    External (API) sampling rate (Hz)               */
    const SKP_int               max_internal_fs_kHz,/* I    Maximum internal sampling rate (kHz)            */
    const SKP_int               PacketSize_ms,      /* I    Packet length (ms)                              */
    SKP_int32                   TargetRate_bps,     /* I    Target max bitrate (bps) (used if SNR_dB == 0)  */
    const SKP_int               PacketLoss_perc,    /* I    Packet loss rate (in percent)                   */
    const SKP_int               INBandFEC_enabled,  /* I    Enable (1) / disable (0) inband FEC             */
    const SKP_int               DTX_enabled,        /* I    Enable / disable DTX                            */
    const SKP_int               InputFramesize_ms,  /* I    Inputframe in ms                                */
    const SKP_int               Complexity          /* I    Complexity (0->low; 1->medium; 2->high)         */
)
{
    SKP_int   fs_kHz, ret = 0;

    /* State machine for the SWB/WB switching */
    fs_kHz = psEnc->sCmn.fs_kHz;
    
    /* Only switch during low speech activity, when no frames are sitting in the payload buffer */
    if( API_fs_Hz == 8000 || fs_kHz == 0 || API_fs_Hz < SKP_SMULBB( fs_kHz, 1000 ) || fs_kHz > max_internal_fs_kHz ) {
        /* Switching is not possible, encoder just initialized, internal mode higher than external, */
        /* or internal mode higher than maximum allowed internal mode                               */
        fs_kHz = SKP_min( SKP_DIV32_16( API_fs_Hz, 1000 ), max_internal_fs_kHz );
    } else {
        /* Accumulate the difference between the target rate and limit for switching down */
        psEnc->sCmn.bitrateDiff += SKP_MUL( InputFramesize_ms, TargetRate_bps - psEnc->sCmn.bitrate_threshold_down );
        psEnc->sCmn.bitrateDiff  = SKP_min( psEnc->sCmn.bitrateDiff, 0 );

        if( psEnc->speech_activity_Q8 < 128 && psEnc->sCmn.nFramesInPayloadBuf == 0 ) { /* Low speech activity and payload buffer empty */
            /* Check if we should switch down */
#if SWITCH_TRANSITION_FILTERING 
            if( ( psEnc->sCmn.sLP.transition_frame_no == 0 ) &&                         /* Transition phase not active */
                ( psEnc->sCmn.bitrateDiff <= -ACCUM_BITS_DIFF_THRESHOLD ||              /* Bitrate threshold is met */
                ( psEnc->sCmn.sSWBdetect.WB_detected * psEnc->sCmn.fs_kHz == 24 ) ) ) { /* Forced down-switching due to WB input */
                psEnc->sCmn.sLP.transition_frame_no = 1;                                /* Begin transition phase */
                psEnc->sCmn.sLP.mode                = 0;                                /* Switch down */
            } else if( 
                ( psEnc->sCmn.sLP.transition_frame_no >= TRANSITION_FRAMES_DOWN ) &&    /* Transition phase complete */
                ( psEnc->sCmn.sLP.mode == 0 ) ) {                                       /* Ready to switch down */
                psEnc->sCmn.sLP.transition_frame_no = 0;                                /* Ready for new transition phase */
#else
            if( psEnc->sCmn.bitrateDiff <= -ACCUM_BITS_DIFF_THRESHOLD ) {               /* Bitrate threshold is met */ 
#endif            
                psEnc->sCmn.bitrateDiff = 0;

                /* Switch to a lower sample frequency */
                if( psEnc->sCmn.fs_kHz == 24 ) {
                    fs_kHz = 16;
                } else if( psEnc->sCmn.fs_kHz == 16 ) {
                    fs_kHz = 12;
                } else {
                    SKP_assert( psEnc->sCmn.fs_kHz == 12 );
                    fs_kHz = 8;
                }
            }

            /* Check if we should switch up */
            if( ( ( SKP_SMULBB( psEnc->sCmn.fs_kHz, 1000 ) < API_fs_Hz ) &&
                ( TargetRate_bps >= psEnc->sCmn.bitrate_threshold_up ) && 
                ( psEnc->sCmn.sSWBdetect.WB_detected * psEnc->sCmn.fs_kHz != 16 ) ) && 
                ( ( psEnc->sCmn.fs_kHz == 16 ) && ( max_internal_fs_kHz >= 24 ) || 
                  ( psEnc->sCmn.fs_kHz == 12 ) && ( max_internal_fs_kHz >= 16 ) ||
                  ( psEnc->sCmn.fs_kHz ==  8 ) && ( max_internal_fs_kHz >= 12 ) ) 
#if SWITCH_TRANSITION_FILTERING
                  && ( psEnc->sCmn.sLP.transition_frame_no == 0 ) ) { /* No transition phase running, ready to switch */
                    psEnc->sCmn.sLP.mode = 1; /* Switch up */
#else
                ) {
#endif
                psEnc->sCmn.bitrateDiff = 0;

                /* Switch to a higher sample frequency */
                if( psEnc->sCmn.fs_kHz == 8 ) {
                    fs_kHz = 12;
                } else if( psEnc->sCmn.fs_kHz == 12 ) {
                    fs_kHz = 16;
                } else {
                    SKP_assert( psEnc->sCmn.fs_kHz == 16 );
                    fs_kHz = 24;
                } 
            }
        }
    }

#if SWITCH_TRANSITION_FILTERING
    /* After switching up, stop transition filter during speech inactivity */
    if( ( psEnc->sCmn.sLP.mode == 1 ) &&
        ( psEnc->sCmn.sLP.transition_frame_no >= TRANSITION_FRAMES_UP ) && 
        ( psEnc->speech_activity_Q8 < 128 ) && 
        ( psEnc->sCmn.nFramesInPayloadBuf == 0 ) ) {
        
        psEnc->sCmn.sLP.transition_frame_no = 0;

        /* Reset transition filter state */
        SKP_memset( psEnc->sCmn.sLP.In_LP_State, 0, 2 * sizeof( SKP_int32 ) );
    }
#endif

#ifdef FORCE_FS_KHZ
    SKP_assert( FORCE_FS_KHZ == 8 || FORCE_FS_KHZ == 12 || FORCE_FS_KHZ == 16 || FORCE_FS_KHZ == 24 );
    if( psEnc->sCmn.fs_kHz != 0 ) {
        /* Force except during init */
        fs_kHz = FORCE_FS_KHZ;
    }
#endif

#ifdef SAVE_ALL_INTERNAL_DATA
    DEBUG_STORE_DATA( bitrate.dat,  &TargetRate_bps,          1 * sizeof( SKP_int32 ) ); 
    DEBUG_STORE_DATA( fs.dat,       &fs_kHz,                  1 * sizeof( SKP_int32 ) ); 
    DEBUG_STORE_DATA( diff.dat,     &psEnc->sCmn.bitrateDiff, 1 * sizeof( SKP_int32 ) ); 
    DEBUG_STORE_DATA( lashape.dat,  &psEnc->sCmn.la_shape,    1 * sizeof( SKP_int32 ) ); 
#endif
    /********************************************/
    /* Prepare resampler and buffered data      */    
    /********************************************/
    SKP_Silk_setup_resamplers( psEnc, API_fs_Hz, fs_kHz );

    /********************************************/
    /* Set packet size                          */
    /********************************************/
    ret += SKP_Silk_setup_packetsize( psEnc, PacketSize_ms, fs_kHz );

    /********************************************/
    /* Set internal sampling frequency          */
    /********************************************/
    ret += SKP_Silk_setup_fs( psEnc, fs_kHz );

    /********************************************/
    /* Set encoding complexity                  */
    /********************************************/
    ret += SKP_Silk_setup_complexity( psEnc, Complexity );

    /********************************************/
    /* Set bitrate/coding quality               */
    /********************************************/
    ret += SKP_Silk_setup_rate( psEnc, TargetRate_bps );

    /********************************************/
    /* Set packet loss rate measured by farend  */
    /********************************************/
    if( ( PacketLoss_perc < 0 ) || ( PacketLoss_perc > 100 ) ) {
        ret = SKP_SILK_ENC_INVALID_LOSS_RATE;
    }
    psEnc->sCmn.PacketLoss_perc = PacketLoss_perc;

    /********************************************/
    /* Set LBRR usage                           */
    /********************************************/
    ret += SKP_Silk_setup_LBRR( psEnc, INBandFEC_enabled );

    /********************************************/
    /* Set DTX mode                             */
    /********************************************/
    if( DTX_enabled < 0 || DTX_enabled > 1 ) {
        ret = SKP_SILK_ENC_INVALID_DTX_SETTING;
    }
    psEnc->sCmn.useDTX = DTX_enabled;

    return ret;
}

/* Control low bitrate redundancy usage */
void SKP_Silk_LBRR_ctrl_FIX(
    SKP_Silk_encoder_state_FIX      *psEnc,     /* I/O  encoder state                               */
    SKP_Silk_encoder_control_FIX    *psEncCtrl  /* I/O  encoder control                             */
)
{
    SKP_int LBRR_usage;

    if( psEnc->sCmn.LBRR_enabled ) {
        /* Control LBRR */

        /* Usage Control based on sensitivity and packet loss caracteristics */
        /* For now only enable adding to next for active frames. Make more complex later */
        LBRR_usage = SKP_SILK_NO_LBRR;
        if( psEnc->speech_activity_Q8 > LBRR_SPEECH_ACTIVITY_THRES_Q8 && psEnc->sCmn.PacketLoss_perc > LBRR_LOSS_THRES ) { // nb! maybe multiply loss prob and speech activity 
            LBRR_usage = SKP_SILK_ADD_LBRR_TO_PLUS1;
        }
        psEncCtrl->sCmn.LBRR_usage = LBRR_usage;
    } else {
        psEncCtrl->sCmn.LBRR_usage = SKP_SILK_NO_LBRR;
    }
}

SKP_INLINE SKP_int SKP_Silk_setup_packetsize(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         PacketSize_ms,      /* I                        */
    SKP_int                         fs_kHz              /* I                        */
)
{
    SKP_int ret = SKP_SILK_NO_ERROR;
    if( ( PacketSize_ms !=  10 ) &&
        ( PacketSize_ms !=  20 ) &&
        ( PacketSize_ms !=  40 ) && 
        ( PacketSize_ms !=  60 ) && 
        ( PacketSize_ms !=  80 ) && 
        ( PacketSize_ms != 100 ) ) {
        ret = SKP_SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
    } else {
        if( PacketSize_ms != psEnc->sCmn.PacketSize_ms ) {
            if( PacketSize_ms == 10 ) {
                if( psEnc->sCmn.nFramesInPayloadBuf == 0 ) {
                    /* Only allowed when the payload buffer is empty */
                    psEnc->sCmn.nb_subfr      = MAX_NB_SUBFR >> 1;
                    psEnc->sCmn.frame_length  = SKP_SMULBB( psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr );
                    psEnc->sCmn.PacketSize_ms = PacketSize_ms;
                    psEnc->sPred.pitch_LPC_win_length = SKP_SMULBB( FIND_PITCH_LPC_WIN_MS_2_SF, fs_kHz );
                    /* Packet length changes. Reset LBRR buffer */
                    SKP_Silk_LBRR_reset( &psEnc->sCmn );
                }
            } else{
                psEnc->sCmn.nb_subfr      = MAX_NB_SUBFR;
                psEnc->sCmn.frame_length  = SKP_SMULBB( psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr );
                psEnc->sCmn.PacketSize_ms = PacketSize_ms;
                psEnc->sPred.pitch_LPC_win_length = SKP_SMULBB( FIND_PITCH_LPC_WIN_MS, fs_kHz );
                /* Packet length changes. Reset LBRR buffer */
                SKP_Silk_LBRR_reset( &psEnc->sCmn );
            }
        }
    }
    return(ret);
}

SKP_INLINE SKP_int SKP_Silk_setup_resamplers(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         API_fs_Hz,          /* I                        */
    SKP_int                         fs_kHz              /* I                        */
)
{
    SKP_int ret = SKP_SILK_NO_ERROR;
    
    if( psEnc->sCmn.fs_kHz != fs_kHz || psEnc->sCmn.prev_API_fs_Hz != API_fs_Hz ) {

        /* Allocate space for worst case temporary upsampling, 8 to 48 kHz, so a factor 6 */
        SKP_int16 x_buf_API_fs_Hz[ ( MAX_API_FS_KHZ / 8 ) * ( 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX ) ];

        SKP_int32 nSamples_temp = SKP_LSHIFT( psEnc->sCmn.frame_length, 1 ) + psEnc->sCmn.la_shape;

        if( SKP_SMULBB( fs_kHz, 1000 ) < API_fs_Hz && psEnc->sCmn.fs_kHz != 0 ) {
            /* Resample buffered data in x_buf to API_fs_Hz */

            SKP_Silk_resampler_state_struct  temp_resampler_state;

            /* Initialize resampler for temporary resampling of x_buf data to API_fs_Hz */
            ret += SKP_Silk_resampler_init( &temp_resampler_state, SKP_SMULBB( psEnc->sCmn.fs_kHz, 1000 ), API_fs_Hz );

            /* Temporary resampling of x_buf data to API_fs_Hz */
            ret += SKP_Silk_resampler( &temp_resampler_state, x_buf_API_fs_Hz, psEnc->x_buf, nSamples_temp );

            /* Calculate number of samples that has been temporarily upsampled */
            nSamples_temp = SKP_DIV32_16( nSamples_temp * API_fs_Hz, SKP_SMULBB( psEnc->sCmn.fs_kHz, 1000 ) );

            /* Initialize the resampler for enc_API.c preparing resampling from API_fs_Hz to fs_kHz */
            ret += SKP_Silk_resampler_init( &psEnc->sCmn.resampler_state, API_fs_Hz, SKP_SMULBB( fs_kHz, 1000 ) );

        } else {
            /* Copy data */
            SKP_memcpy( x_buf_API_fs_Hz, psEnc->x_buf, nSamples_temp * sizeof( SKP_int16 ) );
        }

        if( 1000 * fs_kHz != API_fs_Hz ) {
            /* Correct resampler state (unless resampling by a factor 1) by resampling buffered data from API_fs_Hz to fs_kHz */
            ret += SKP_Silk_resampler( &psEnc->sCmn.resampler_state, psEnc->x_buf, x_buf_API_fs_Hz, nSamples_temp );
        }
    }
    psEnc->sCmn.prev_API_fs_Hz = API_fs_Hz;
 
    return(ret);
}

SKP_INLINE SKP_int SKP_Silk_setup_fs(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         fs_kHz              /* I                        */
)
{
    SKP_int ret = 0;

    if( psEnc->sCmn.fs_kHz != fs_kHz ) {
        /* reset part of the state */
        SKP_memset( &psEnc->sShape,          0,                            sizeof( SKP_Silk_shape_state_FIX ) );
        SKP_memset( &psEnc->sPrefilt,        0,                            sizeof( SKP_Silk_prefilter_state_FIX ) );
        SKP_memset( &psEnc->sNSQ,            0,                            sizeof( SKP_Silk_nsq_state ) );
        SKP_memset( &psEnc->sPred,           0,                            sizeof( SKP_Silk_predict_state_FIX ) );
        SKP_memset( psEnc->sNSQ.xq,          0, ( 2 * MAX_FRAME_LENGTH ) * sizeof( SKP_int16 ) );
        SKP_memset( psEnc->sNSQ_LBRR.xq,     0, ( 2 * MAX_FRAME_LENGTH ) * sizeof( SKP_int16 ) );
        SKP_memset( psEnc->sCmn.LBRR_buffer, 0,           MAX_LBRR_DELAY * sizeof( SKP_SILK_LBRR_struct ) );
#if SWITCH_TRANSITION_FILTERING
        SKP_memset( psEnc->sCmn.sLP.In_LP_State, 0, 2 * sizeof( SKP_int32 ) );
        if( psEnc->sCmn.sLP.mode == 1 ) {
            /* Begin transition phase */
            psEnc->sCmn.sLP.transition_frame_no = 1;
        } else {
            /* End transition phase */
            psEnc->sCmn.sLP.transition_frame_no = 0;
        }
#endif
        psEnc->sCmn.inputBufIx          = 0;
        psEnc->sCmn.nFramesInPayloadBuf = 0;
        psEnc->sCmn.nBytesInPayloadBuf  = 0;
        psEnc->sCmn.oldest_LBRR_idx     = 0;
        psEnc->sCmn.TargetRate_bps      = 0; /* Ensures that psEnc->SNR_dB is recomputed */

        SKP_memset( psEnc->sPred.prev_NLSFq_Q15, 0, MAX_LPC_ORDER * sizeof( SKP_int ) );

        /* Initialize non-zero parameters */
        psEnc->sCmn.prevLag                 = 100;
        psEnc->sCmn.prev_sigtype            = SIG_TYPE_UNVOICED;
        psEnc->sCmn.first_frame_after_reset = 1;
        psEnc->sPrefilt.lagPrev             = 100;
        psEnc->sShape.LastGainIndex         = 1;
        psEnc->sNSQ.lagPrev                 = 100;
        psEnc->sNSQ.prev_inv_gain_Q16       = 65536;
        psEnc->sNSQ_LBRR.prev_inv_gain_Q16  = 65536;
        psEnc->sCmn.fs_kHz = fs_kHz;
        if( psEnc->sCmn.fs_kHz == 8 ) {
            psEnc->sCmn.predictLPCOrder = MIN_LPC_ORDER;
            psEnc->sCmn.psNLSF_CB[ 0 ]  = &SKP_Silk_NLSF_CB0_10;
            psEnc->sCmn.psNLSF_CB[ 1 ]  = &SKP_Silk_NLSF_CB1_10;
        } else {
            psEnc->sCmn.predictLPCOrder = MAX_LPC_ORDER;
            psEnc->sCmn.psNLSF_CB[ 0 ]  = &SKP_Silk_NLSF_CB0_16;
            psEnc->sCmn.psNLSF_CB[ 1 ]  = &SKP_Silk_NLSF_CB1_16;
        }
        psEnc->sCmn.subfr_length   = SKP_SMULBB( SUB_FRAME_LENGTH_MS, fs_kHz );
        psEnc->sCmn.frame_length   = SKP_SMULBB( psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr );
        psEnc->sCmn.ltp_mem_length = SKP_SMULBB( LTP_MEM_LENGTH_MS, fs_kHz ); 
        psEnc->sCmn.la_pitch       = SKP_SMULBB( LA_PITCH_MS, fs_kHz );
        psEnc->sCmn.la_shape       = SKP_SMULBB( LA_SHAPE_MS, fs_kHz );
        psEnc->sPred.min_pitch_lag = SKP_SMULBB(  3, fs_kHz );
        psEnc->sPred.max_pitch_lag = SKP_SMULBB( 18, fs_kHz );
        if( psEnc->sCmn.nb_subfr == MAX_NB_SUBFR ){
            psEnc->sPred.pitch_LPC_win_length = SKP_SMULBB( FIND_PITCH_LPC_WIN_MS, fs_kHz );
        } else {
            psEnc->sPred.pitch_LPC_win_length = SKP_SMULBB( FIND_PITCH_LPC_WIN_MS_2_SF, fs_kHz );
        }
        if( psEnc->sCmn.fs_kHz == 24 ) {
            psEnc->mu_LTP_Q8 = MU_LTP_QUANT_SWB_Q8;
            psEnc->sCmn.bitrate_threshold_up   = SKP_int32_MAX;
            psEnc->sCmn.bitrate_threshold_down = SWB2WB_BITRATE_BPS; 
        } else if( psEnc->sCmn.fs_kHz == 16 ) {
            psEnc->mu_LTP_Q8 = MU_LTP_QUANT_WB_Q8;
            psEnc->sCmn.bitrate_threshold_up   = WB2SWB_BITRATE_BPS;
            psEnc->sCmn.bitrate_threshold_down = WB2MB_BITRATE_BPS; 
        } else if( psEnc->sCmn.fs_kHz == 12 ) {
            psEnc->mu_LTP_Q8 = MU_LTP_QUANT_MB_Q8;
            psEnc->sCmn.bitrate_threshold_up   = MB2WB_BITRATE_BPS;
            psEnc->sCmn.bitrate_threshold_down = MB2NB_BITRATE_BPS;
        } else {
            psEnc->mu_LTP_Q8 = MU_LTP_QUANT_NB_Q8;
            psEnc->sCmn.bitrate_threshold_up   = NB2MB_BITRATE_BPS;
            psEnc->sCmn.bitrate_threshold_down = 0;
        }
        psEnc->sCmn.fs_kHz_changed = 1;

        /* Check that settings are valid */
        SKP_assert( ( psEnc->sCmn.subfr_length * psEnc->sCmn.nb_subfr ) == psEnc->sCmn.frame_length );
    }
    return( ret );
}

SKP_INLINE SKP_int SKP_Silk_setup_complexity(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         Complexity          /* I                        */
)
{
    SKP_int ret = 0;

    /* Check that settings are valid */
    if( LOW_COMPLEXITY_ONLY && Complexity != 0 ) { 
        ret = SKP_SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }

    /* Set encoding complexity */
    if( Complexity == 0 || LOW_COMPLEXITY_ONLY ) {
        /* Low complexity */
        psEnc->sCmn.Complexity                  = 0;
        psEnc->sCmn.pitchEstimationComplexity   = PITCH_EST_COMPLEXITY_LC_MODE;
        psEnc->pitchEstimationThreshold_Q16     = FIND_PITCH_CORRELATION_THRESHOLD_Q16_LC_MODE;
        psEnc->sCmn.pitchEstimationLPCOrder     = 8;
        psEnc->sCmn.shapingLPCOrder             = 8;
        psEnc->sCmn.nStatesDelayedDecision      = 1;
        psEnc->sCmn.useInterpolatedNLSFs        = 0;
        psEnc->sCmn.LTPQuantLowComplexity       = 1;
        psEnc->sCmn.NLSF_MSVQ_Survivors         = MAX_NLSF_MSVQ_SURVIVORS_LC_MODE;
    } else if( Complexity == 1 ) {
        /* Medium complexity */
        psEnc->sCmn.Complexity                  = 1;
        psEnc->sCmn.pitchEstimationComplexity   = PITCH_EST_COMPLEXITY_MC_MODE;
        psEnc->pitchEstimationThreshold_Q16     = FIND_PITCH_CORRELATION_THRESHOLD_Q16_MC_MODE;
        psEnc->sCmn.pitchEstimationLPCOrder     = 12;
        psEnc->sCmn.shapingLPCOrder             = 12;
        psEnc->sCmn.nStatesDelayedDecision      = 2;
        psEnc->sCmn.useInterpolatedNLSFs        = 0;
        psEnc->sCmn.LTPQuantLowComplexity       = 0;
        psEnc->sCmn.NLSF_MSVQ_Survivors         = MAX_NLSF_MSVQ_SURVIVORS_MC_MODE;
    } else if( Complexity == 2 ) {
        /* High complexity */
        psEnc->sCmn.Complexity                  = 2;
        psEnc->sCmn.pitchEstimationComplexity   = PITCH_EST_COMPLEXITY_HC_MODE;
        psEnc->pitchEstimationThreshold_Q16     = FIND_PITCH_CORRELATION_THRESHOLD_Q16_HC_MODE;
        psEnc->sCmn.pitchEstimationLPCOrder     = 16;
        psEnc->sCmn.shapingLPCOrder             = 16;
        psEnc->sCmn.nStatesDelayedDecision      = 4;
        psEnc->sCmn.useInterpolatedNLSFs        = 1;
        psEnc->sCmn.LTPQuantLowComplexity       = 0;
        psEnc->sCmn.NLSF_MSVQ_Survivors         = MAX_NLSF_MSVQ_SURVIVORS;
    } else {
        ret = SKP_SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }

    /* Do not allow higher pitch estimation LPC order than predict LPC order */
    psEnc->sCmn.pitchEstimationLPCOrder = SKP_min_int( psEnc->sCmn.pitchEstimationLPCOrder, psEnc->sCmn.predictLPCOrder );

    SKP_assert( psEnc->sCmn.pitchEstimationLPCOrder <= MAX_FIND_PITCH_LPC_ORDER );
    SKP_assert( psEnc->sCmn.shapingLPCOrder         <= MAX_SHAPE_LPC_ORDER      );
    SKP_assert( psEnc->sCmn.nStatesDelayedDecision  <= MAX_DEL_DEC_STATES       );

    return( ret );
}

SKP_INLINE SKP_int SKP_Silk_setup_rate(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         TargetRate_bps      /* I                        */
)
{
    SKP_int k, ret = 0;
    SKP_int32 frac_Q6;
    const SKP_int32 *rateTable;

    TargetRate_bps = SKP_min( TargetRate_bps, 100000 );
    if( psEnc->sCmn.fs_kHz == 8 ) {
        TargetRate_bps = SKP_max( TargetRate_bps, MIN_TARGET_RATE_NB_BPS );
    } else if( psEnc->sCmn.fs_kHz == 12 ) {
        TargetRate_bps = SKP_max( TargetRate_bps, MIN_TARGET_RATE_MB_BPS );
    } else if( psEnc->sCmn.fs_kHz == 16 ) {
        TargetRate_bps = SKP_max( TargetRate_bps, MIN_TARGET_RATE_WB_BPS );
    } else {
        TargetRate_bps = SKP_max( TargetRate_bps, MIN_TARGET_RATE_SWB_BPS );
    }
    if( TargetRate_bps != psEnc->sCmn.TargetRate_bps ) {
        psEnc->sCmn.TargetRate_bps = TargetRate_bps;

        /* If new TargetRate_bps, translate to SNR_dB value */
        if( psEnc->sCmn.fs_kHz == 8 ) {
            rateTable = TargetRate_table_NB;
        } else if( psEnc->sCmn.fs_kHz == 12 ) {
            rateTable = TargetRate_table_MB;
        } else if( psEnc->sCmn.fs_kHz == 16 ) {
            rateTable = TargetRate_table_WB;
        } else {
            rateTable = TargetRate_table_SWB;
        }
        for( k = 1; k < TARGET_RATE_TAB_SZ; k++ ) {
            /* Find bitrate interval in table and interpolate */
            if( TargetRate_bps < rateTable[ k ] ) {
                frac_Q6 = SKP_DIV32( SKP_LSHIFT( TargetRate_bps - rateTable[ k - 1 ], 6 ), 
                                                 rateTable[ k ] - rateTable[ k - 1 ] );
                psEnc->SNR_dB_Q7 = SKP_LSHIFT( SNR_table_Q1[ k - 1 ], 6 ) + SKP_MUL( frac_Q6, SNR_table_Q1[ k ] - SNR_table_Q1[ k - 1 ] );
                break;
            }
        }
    }
    return( ret );
}

SKP_INLINE SKP_int SKP_Silk_setup_LBRR(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         INBandFEC_enabled   /* I                        */
)
{
    SKP_int   ret = 0;
    SKP_int32 LBRRRate_thres_bps;

#if USE_LBRR
    if( INBandFEC_enabled < 0 || INBandFEC_enabled > 1 ) {
        ret = SKP_SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    
    /* Only change settings if first frame in packet */
    if( psEnc->sCmn.nFramesInPayloadBuf == 0 ) {
        
        psEnc->sCmn.LBRR_enabled = INBandFEC_enabled;
        if( psEnc->sCmn.fs_kHz == 8 ) {
            LBRRRate_thres_bps = INBAND_FEC_MIN_RATE_BPS - 9000;
        } else if( psEnc->sCmn.fs_kHz == 12 ) {
            LBRRRate_thres_bps = INBAND_FEC_MIN_RATE_BPS - 6000;;
        } else if( psEnc->sCmn.fs_kHz == 16 ) {
            LBRRRate_thres_bps = INBAND_FEC_MIN_RATE_BPS - 3000;
        } else {
            LBRRRate_thres_bps = INBAND_FEC_MIN_RATE_BPS;
        }

        if( psEnc->sCmn.TargetRate_bps >= LBRRRate_thres_bps ) {
            /* Set gain increase / rate reduction for LBRR usage */
            /* Coarsely tuned with PESQ for now. */
            /* Linear regression coefs G = 8 - 0.5 * loss */
            /* Meaning that at 16% loss main rate and redundant rate is the same, -> G = 0 */
            psEnc->sCmn.LBRR_GainIncreases = SKP_max_int( 8 - SKP_RSHIFT( psEnc->sCmn.PacketLoss_perc, 1 ), 0 );

            /* Set main stream rate compensation */
            if( psEnc->sCmn.LBRR_enabled && psEnc->sCmn.PacketLoss_perc > LBRR_LOSS_THRES ) {
                /* Tuned to give aprox same mean / weighted bitrate as no inband FEC */
                psEnc->inBandFEC_SNR_comp_Q8 = ( 6 << 8 ) - SKP_LSHIFT( psEnc->sCmn.LBRR_GainIncreases, 7 );
            } else {
                psEnc->inBandFEC_SNR_comp_Q8 = 0;
                psEnc->sCmn.LBRR_enabled     = 0;
            }
        } else {
            psEnc->inBandFEC_SNR_comp_Q8     = 0;
            psEnc->sCmn.LBRR_enabled         = 0;
        }
    }
#else
    if( INBandFEC_enabled != 0 ) {
        ret = SKP_SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    psEnc->sCmn.LBRR_enabled = 0;
#endif
    return( ret );
}