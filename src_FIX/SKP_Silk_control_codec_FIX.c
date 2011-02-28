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
#include "SKP_Silk_setup.h"

SKP_INLINE SKP_int SKP_Silk_setup_resamplers(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         fs_kHz              /* I                        */
);

SKP_INLINE SKP_int SKP_Silk_setup_fs(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         fs_kHz,             /* I                        */
    SKP_int                         PacketSize_ms       /* I                        */
);

SKP_INLINE SKP_int SKP_Silk_setup_rate(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int32                       TargetRate_bps      /* I                        */
);

/* Control encoder SNR */
SKP_int SKP_Silk_control_encoder_FIX( 
    SKP_Silk_encoder_state_FIX  *psEnc,                 /* I/O  Pointer to Silk encoder state           */
    const SKP_int               PacketSize_ms,          /* I    Packet length (ms)                      */
    const SKP_int32             TargetRate_bps,         /* I    Target max bitrate (bps)                */
    const SKP_int               PacketLoss_perc,        /* I    Packet loss rate (in percent)           */
    const SKP_int               Complexity              /* I    Complexity (0-10)                       */
)
{
    SKP_int   fs_kHz, ret = 0;

    if( psEnc->sCmn.controlled_since_last_payload != 0 && psEnc->sCmn.prefillFlag == 0 ) {
        if( psEnc->sCmn.API_fs_Hz != psEnc->sCmn.prev_API_fs_Hz && psEnc->sCmn.fs_kHz > 0 ) {
            /* Change in API sampling rate in the middle of encoding a packet */
            ret += SKP_Silk_setup_resamplers( psEnc, psEnc->sCmn.fs_kHz );
        }
        return ret;
    }

    /* Beyond this point we know that there are no previously coded frames in the payload buffer */

    /********************************************/
    /* Determine internal sampling rate         */
    /********************************************/
    fs_kHz = SKP_Silk_control_audio_bandwidth( &psEnc->sCmn, TargetRate_bps );

    /********************************************/
    /* Prepare resampler and buffered data      */
    /********************************************/
    ret += SKP_Silk_setup_resamplers( psEnc, fs_kHz );

    /********************************************/
    /* Set internal sampling frequency          */
    /********************************************/
    ret += SKP_Silk_setup_fs( psEnc, fs_kHz, PacketSize_ms );

    /********************************************/
    /* Set encoding complexity                  */
    /********************************************/
    ret += SKP_Silk_setup_complexity( &psEnc->sCmn, Complexity );

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
    ret += SKP_Silk_setup_LBRR( &psEnc->sCmn );

    psEnc->sCmn.controlled_since_last_payload = 1;

    return ret;
}

SKP_INLINE SKP_int SKP_Silk_setup_resamplers(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         fs_kHz              /* I                        */
)
{
    SKP_int ret = SKP_SILK_NO_ERROR;
    
    if( psEnc->sCmn.fs_kHz != fs_kHz || psEnc->sCmn.prev_API_fs_Hz != psEnc->sCmn.API_fs_Hz ) {

        if( psEnc->sCmn.fs_kHz == 0 ) {
            /* Initialize the resampler for enc_API.c preparing resampling from API_fs_Hz to fs_kHz */
            ret += SKP_Silk_resampler_init( &psEnc->sCmn.resampler_state, psEnc->sCmn.API_fs_Hz, fs_kHz * 1000 );
        } else {
            /* Allocate space for worst case temporary upsampling, 8 to 48 kHz, so a factor 6 */
            SKP_int16 x_buf_API_fs_Hz[ ( 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX ) * ( MAX_API_FS_KHZ / 8 ) ];

            SKP_int32 nSamples_temp = SKP_LSHIFT( psEnc->sCmn.frame_length, 1 ) + LA_SHAPE_MS * psEnc->sCmn.fs_kHz;

            if( SKP_SMULBB( fs_kHz, 1000 ) < psEnc->sCmn.API_fs_Hz && psEnc->sCmn.fs_kHz != 0 ) {
                /* Resample buffered data in x_buf to API_fs_Hz */

                SKP_Silk_resampler_state_struct  temp_resampler_state;

                /* Initialize resampler for temporary resampling of x_buf data to API_fs_Hz */
                ret += SKP_Silk_resampler_init( &temp_resampler_state, SKP_SMULBB( psEnc->sCmn.fs_kHz, 1000 ), psEnc->sCmn.API_fs_Hz );

                /* Temporary resampling of x_buf data to API_fs_Hz */
                ret += SKP_Silk_resampler( &temp_resampler_state, x_buf_API_fs_Hz, psEnc->x_buf, nSamples_temp );

                /* Calculate number of samples that has been temporarily upsampled */
                nSamples_temp = SKP_DIV32_16( nSamples_temp * psEnc->sCmn.API_fs_Hz, SKP_SMULBB( psEnc->sCmn.fs_kHz, 1000 ) );

                /* Initialize the resampler for enc_API.c preparing resampling from API_fs_Hz to fs_kHz */
                ret += SKP_Silk_resampler_init( &psEnc->sCmn.resampler_state, psEnc->sCmn.API_fs_Hz, SKP_SMULBB( fs_kHz, 1000 ) );

            } else {
                /* Copy data */
                SKP_memcpy( x_buf_API_fs_Hz, psEnc->x_buf, nSamples_temp * sizeof( SKP_int16 ) );
            }

            if( 1000 * fs_kHz != psEnc->sCmn.API_fs_Hz ) {
                /* Correct resampler state (unless resampling by a factor 1) by resampling buffered data from API_fs_Hz to fs_kHz */
                ret += SKP_Silk_resampler( &psEnc->sCmn.resampler_state, psEnc->x_buf, x_buf_API_fs_Hz, nSamples_temp );
            }
        }
    }

    psEnc->sCmn.prev_API_fs_Hz = psEnc->sCmn.API_fs_Hz;

    return(ret);
}

SKP_INLINE SKP_int SKP_Silk_setup_fs(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int                         fs_kHz,             /* I                        */
    SKP_int                         PacketSize_ms       /* I                        */
)
{
    SKP_int ret = SKP_SILK_NO_ERROR;

    /* Set packet size */
    if( PacketSize_ms != psEnc->sCmn.PacketSize_ms ) {
        if( ( PacketSize_ms !=  10 ) &&
            ( PacketSize_ms !=  20 ) &&
            ( PacketSize_ms !=  40 ) && 
            ( PacketSize_ms !=  60 ) ) {
            ret = SKP_SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
        }
        if( PacketSize_ms <= 10 ) {
            psEnc->sCmn.nFramesPerPacket = 1;
            psEnc->sCmn.nb_subfr = PacketSize_ms == 10 ? 2 : 1;
            psEnc->sCmn.frame_length = SKP_SMULBB( PacketSize_ms, fs_kHz );
            psEnc->sCmn.pitch_LPC_win_length = SKP_SMULBB( FIND_PITCH_LPC_WIN_MS_2_SF, fs_kHz );
            if( psEnc->sCmn.fs_kHz == 8 ) {
                psEnc->sCmn.pitch_contour_iCDF = SKP_Silk_pitch_contour_10_ms_NB_iCDF;
            } else {
                psEnc->sCmn.pitch_contour_iCDF = SKP_Silk_pitch_contour_10_ms_iCDF;
            }
        } else {
            psEnc->sCmn.nFramesPerPacket = SKP_DIV32_16( PacketSize_ms, MAX_FRAME_LENGTH_MS );
            psEnc->sCmn.nb_subfr = MAX_NB_SUBFR;
            psEnc->sCmn.frame_length = SKP_SMULBB( 20, fs_kHz );
            psEnc->sCmn.pitch_LPC_win_length = SKP_SMULBB( FIND_PITCH_LPC_WIN_MS, fs_kHz );
            if( psEnc->sCmn.fs_kHz == 8 ) {
                psEnc->sCmn.pitch_contour_iCDF = SKP_Silk_pitch_contour_NB_iCDF;
            } else {
                psEnc->sCmn.pitch_contour_iCDF = SKP_Silk_pitch_contour_iCDF; 
            }
        }
        psEnc->sCmn.PacketSize_ms  = PacketSize_ms;
        psEnc->sCmn.TargetRate_bps = 0;         /* trigger new SNR computation */
    }

    /* Set internal sampling frequency */
    SKP_assert( fs_kHz == 8 || fs_kHz == 12 || fs_kHz == 16 );
    SKP_assert( psEnc->sCmn.nb_subfr == 2 || psEnc->sCmn.nb_subfr == 4 );
    if( psEnc->sCmn.fs_kHz != fs_kHz ) {
        /* reset part of the state */
        SKP_memset( &psEnc->sShape,               0, sizeof( SKP_Silk_shape_state_FIX ) );
        SKP_memset( &psEnc->sPrefilt,             0, sizeof( SKP_Silk_prefilter_state_FIX ) );
        SKP_memset( &psEnc->sCmn.sNSQ,            0, sizeof( SKP_Silk_nsq_state ) );
        SKP_memset( psEnc->sCmn.prev_NLSFq_Q15,   0, sizeof( psEnc->sCmn.prev_NLSFq_Q15 ) );
        SKP_memset( &psEnc->sCmn.sLP.In_LP_State, 0, sizeof( psEnc->sCmn.sLP.In_LP_State ) );
        psEnc->sCmn.inputBufIx                  = 0;
        psEnc->sCmn.nFramesAnalyzed             = 0;
        psEnc->sCmn.TargetRate_bps              = 0; /* Ensures that psEnc->SNR_dB is recomputed */

        /* Initialize non-zero parameters */
        psEnc->sCmn.prevLag                     = 100;
        psEnc->sCmn.first_frame_after_reset     = 1;
        psEnc->sPrefilt.lagPrev                 = 100;
        psEnc->sShape.LastGainIndex             = 10;
        psEnc->sCmn.sNSQ.lagPrev                = 100;
        psEnc->sCmn.sNSQ.prev_inv_gain_Q16      = 65536;

        psEnc->sCmn.fs_kHz = fs_kHz;
        if( psEnc->sCmn.fs_kHz == 8 ) {
            if( psEnc->sCmn.nb_subfr == MAX_NB_SUBFR ) {
                psEnc->sCmn.pitch_contour_iCDF = SKP_Silk_pitch_contour_NB_iCDF; 
            } else {
                psEnc->sCmn.pitch_contour_iCDF = SKP_Silk_pitch_contour_10_ms_NB_iCDF;
            }
        } else {
            if( psEnc->sCmn.nb_subfr == MAX_NB_SUBFR ) {
                psEnc->sCmn.pitch_contour_iCDF = SKP_Silk_pitch_contour_iCDF; 
            } else {
                psEnc->sCmn.pitch_contour_iCDF = SKP_Silk_pitch_contour_10_ms_iCDF;
            }
        }
        if( psEnc->sCmn.fs_kHz == 8 || psEnc->sCmn.fs_kHz == 12 ) {
            psEnc->sCmn.predictLPCOrder = MIN_LPC_ORDER;
            psEnc->sCmn.psNLSF_CB  = &SKP_Silk_NLSF_CB_NB_MB;
        } else {
            psEnc->sCmn.predictLPCOrder = MAX_LPC_ORDER;
            psEnc->sCmn.psNLSF_CB  = &SKP_Silk_NLSF_CB_WB;
        }
        psEnc->sCmn.subfr_length   = SUB_FRAME_LENGTH_MS * fs_kHz;
        psEnc->sCmn.frame_length   = SKP_SMULBB( psEnc->sCmn.subfr_length, psEnc->sCmn.nb_subfr );
        psEnc->sCmn.ltp_mem_length = SKP_SMULBB( LTP_MEM_LENGTH_MS, fs_kHz ); 
        psEnc->sCmn.la_pitch       = SKP_SMULBB( LA_PITCH_MS, fs_kHz );
        psEnc->sCmn.max_pitch_lag  = SKP_SMULBB( 18, fs_kHz );
        if( psEnc->sCmn.nb_subfr == MAX_NB_SUBFR ) {
            psEnc->sCmn.pitch_LPC_win_length = SKP_SMULBB( FIND_PITCH_LPC_WIN_MS, fs_kHz );
        } else {
            psEnc->sCmn.pitch_LPC_win_length = SKP_SMULBB( FIND_PITCH_LPC_WIN_MS_2_SF, fs_kHz );
        }
        if( psEnc->sCmn.fs_kHz == 16 ) {
            psEnc->sCmn.mu_LTP_Q9 = SKP_FIX_CONST( MU_LTP_QUANT_WB, 9 );
            psEnc->sCmn.pitch_lag_low_bits_iCDF = SKP_Silk_uniform8_iCDF;
        } else if( psEnc->sCmn.fs_kHz == 12 ) {
            psEnc->sCmn.mu_LTP_Q9 = SKP_FIX_CONST( MU_LTP_QUANT_MB, 9 );
            psEnc->sCmn.pitch_lag_low_bits_iCDF = SKP_Silk_uniform6_iCDF;
        } else {
            psEnc->sCmn.mu_LTP_Q9 = SKP_FIX_CONST( MU_LTP_QUANT_NB, 9 );
            psEnc->sCmn.pitch_lag_low_bits_iCDF = SKP_Silk_uniform4_iCDF;
        }
    }

    /* Check that settings are valid */
    SKP_assert( ( psEnc->sCmn.subfr_length * psEnc->sCmn.nb_subfr ) == psEnc->sCmn.frame_length );
 
    return( ret );
}

SKP_INLINE SKP_int SKP_Silk_setup_rate(
    SKP_Silk_encoder_state_FIX      *psEnc,             /* I/O                      */
    SKP_int32                       TargetRate_bps      /* I                        */
)
{
    SKP_int k, ret = SKP_SILK_NO_ERROR;
    SKP_int32 frac_Q6;
    const SKP_int32 *rateTable;

    /* Set bitrate/coding quality */
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
            SKP_assert( 0 );
        }
        /* Reduce bitrate for 10 ms modes in these calculations */
        if( psEnc->sCmn.nb_subfr == 2 ) {
            TargetRate_bps -= REDUCE_BITRATE_10_MS_BPS;
        }
        for( k = 1; k < TARGET_RATE_TAB_SZ; k++ ) {
            /* Find bitrate interval in table and interpolate */
            if( TargetRate_bps <= rateTable[ k ] ) {
                frac_Q6 = SKP_DIV32( SKP_LSHIFT( TargetRate_bps - rateTable[ k - 1 ], 6 ), 
                                                 rateTable[ k ] - rateTable[ k - 1 ] );
                psEnc->SNR_dB_Q7 = SKP_LSHIFT( SNR_table_Q1[ k - 1 ], 6 ) + SKP_MUL( frac_Q6, SNR_table_Q1[ k ] - SNR_table_Q1[ k - 1 ] );
                break;
            }
        }
    }
    return( ret );
}
