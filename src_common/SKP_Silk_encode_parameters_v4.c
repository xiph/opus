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

#include "SKP_Silk_main.h"

/*******************************************/
/* Encode parameters to create the payload */
/*******************************************/
void SKP_Silk_encode_parameters_v4(
    SKP_Silk_encoder_state          *psEncC,        /* I/O  Encoder state                   */
    SKP_Silk_encoder_control        *psEncCtrlC,    /* I/O  Encoder control                 */
    SKP_Silk_range_coder_state      *psRC           /* I/O  Range encoder state             */
)
{
    SKP_int   i, k, typeOffset;
    SKP_int   encode_absolute_lagIndex, delta_lagIndex;
    const SKP_Silk_NLSF_CB_struct *psNLSF_CB;

#ifdef SAVE_ALL_INTERNAL_DATA
    SKP_int nBytes_lagIndex, nBytes_contourIndex, nBytes_LTP;
    SKP_int nBytes_after, nBytes_before;
#endif

    /*************************************/
    /* Encode sampling rate and          */
    /* number of subframes in each frame */
    /*************************************/
    /* only done for first frame in packet */
    if( psEncC->nFramesInPayloadBuf == 0 ) {
        /* get sampling rate index */
        for( i = 0; i < 3; i++ ) {
            if( SKP_Silk_SamplingRates_table[ i ] == psEncC->fs_kHz ) {
                break;
            }
        }
        SKP_Silk_range_encoder( psRC, i, SKP_Silk_SamplingRates_CDF );

        /* Convert number of subframes to index */
        SKP_assert( psEncC->nb_subfr == MAX_NB_SUBFR >> 1 || psEncC->nb_subfr == MAX_NB_SUBFR );
        i = (psEncC->nb_subfr >> 1) - 1;
        SKP_Silk_range_encoder( psRC, i, SKP_Silk_NbSubframes_CDF );
    }

    /*********************************************/
    /* Encode VAD flag                           */
    /*********************************************/
    SKP_Silk_range_encoder( psRC, psEncC->vadFlag, SKP_Silk_vadflag_CDF );

    /*******************************************/
    /* Encode signal type and quantizer offset */
    /*******************************************/
    typeOffset = 2 * psEncCtrlC->sigtype + psEncCtrlC->QuantOffsetType;
    if( psEncC->nFramesInPayloadBuf == 0 ) {
        /* first frame in packet: independent coding */
        SKP_Silk_range_encoder( psRC, typeOffset, SKP_Silk_type_offset_CDF );
    } else {
        /* condidtional coding */
        SKP_Silk_range_encoder( psRC, typeOffset, SKP_Silk_type_offset_joint_CDF[ psEncC->typeOffsetPrev ] );
    }
    psEncC->typeOffsetPrev = typeOffset;

    /****************/
    /* Encode gains */
    /****************/
#ifdef SAVE_ALL_INTERNAL_DATA
    SKP_Silk_range_encoder_get_length( psRC, &nBytes_before );
#endif
    /* first subframe */
    if( psEncC->nFramesInPayloadBuf == 0 ) {
        /* first frame in packet: independent coding */
        SKP_Silk_range_encoder( psRC, psEncCtrlC->GainsIndices[ 0 ], SKP_Silk_gain_CDF[ psEncCtrlC->sigtype ] );
    } else {
        /* condidtional coding */
        SKP_Silk_range_encoder( psRC, psEncCtrlC->GainsIndices[ 0 ], SKP_Silk_delta_gain_CDF );
    }

    /* remaining subframes */
    for( i = 1; i < psEncC->nb_subfr; i++ ) {
        SKP_Silk_range_encoder( psRC, psEncCtrlC->GainsIndices[ i ], SKP_Silk_delta_gain_CDF );
    }

#ifdef SAVE_ALL_INTERNAL_DATA
    SKP_Silk_range_encoder_get_length( psRC, &nBytes_after );
    nBytes_after -= nBytes_before; // bytes just added
    DEBUG_STORE_DATA( nBytes_gains.dat, &nBytes_after, sizeof( SKP_int ) );
#endif

    /****************/
    /* Encode NLSFs */
    /****************/
#ifdef SAVE_ALL_INTERNAL_DATA
    SKP_Silk_range_encoder_get_length( psRC, &nBytes_before );
#endif
    /* Range encoding of the NLSF path */
    psNLSF_CB = psEncC->psNLSF_CB[ psEncCtrlC->sigtype ];
    SKP_Silk_range_encoder_multi( psRC, psEncCtrlC->NLSFIndices, psNLSF_CB->StartPtr, psNLSF_CB->nStages );

    /* Encode NLSF interpolation factor */
    SKP_assert( psEncC->useInterpolatedNLSFs == 1 || psEncCtrlC->NLSFInterpCoef_Q2 == ( 1 << 2 ) );
    SKP_Silk_range_encoder( psRC, psEncCtrlC->NLSFInterpCoef_Q2, SKP_Silk_NLSF_interpolation_factor_CDF );

#ifdef SAVE_ALL_INTERNAL_DATA
    SKP_Silk_range_encoder_get_length( psRC, &nBytes_after );
    nBytes_after -= nBytes_before; // bytes just added
    DEBUG_STORE_DATA( nBytes_LSF.dat, &nBytes_after, sizeof( SKP_int ) );
#endif

    if( psEncCtrlC->sigtype == SIG_TYPE_VOICED ) {
        /*********************/
        /* Encode pitch lags */
        /*********************/
#ifdef SAVE_ALL_INTERNAL_DATA
        SKP_Silk_range_encoder_get_length( psRC, &nBytes_before );
#endif
        /* lag index */
        encode_absolute_lagIndex = 1;
        if( psEncC->nFramesInPayloadBuf > 0 && psEncC->prev_sigtype == SIG_TYPE_VOICED ) {
            /* Delta Encoding */
            delta_lagIndex = psEncCtrlC->lagIndex - psEncC->prev_lagIndex;
            if( delta_lagIndex > MAX_DELTA_LAG ) {
                delta_lagIndex = ( MAX_DELTA_LAG << 1 ) + 1;
            } else if ( delta_lagIndex < -MAX_DELTA_LAG ) {
                delta_lagIndex = ( MAX_DELTA_LAG << 1 ) + 1;
            } else {
                delta_lagIndex = delta_lagIndex + MAX_DELTA_LAG;
                encode_absolute_lagIndex = 0; /* Only use delta */
            }
            SKP_Silk_range_encoder( psRC, delta_lagIndex, SKP_Silk_pitch_delta_CDF );
        }
        if( encode_absolute_lagIndex ) {
            /* Absolute encoding */
            if( psEncC->fs_kHz == 8 ) {
                SKP_Silk_range_encoder( psRC, psEncCtrlC->lagIndex, SKP_Silk_pitch_lag_NB_CDF );
            } else if( psEncC->fs_kHz == 12 ) {
                SKP_Silk_range_encoder( psRC, psEncCtrlC->lagIndex, SKP_Silk_pitch_lag_MB_CDF );
            } else if( psEncC->fs_kHz == 16 ) {
                SKP_Silk_range_encoder( psRC, psEncCtrlC->lagIndex, SKP_Silk_pitch_lag_WB_CDF );
            } else {
                SKP_Silk_range_encoder( psRC, psEncCtrlC->lagIndex, SKP_Silk_pitch_lag_SWB_CDF );
            }
        }
        psEncC->prev_lagIndex = psEncCtrlC->lagIndex;

#ifdef SAVE_ALL_INTERNAL_DATA
        SKP_Silk_range_encoder_get_length( psRC, &nBytes_after );
        nBytes_lagIndex = nBytes_after - nBytes_before; // bytes just added
#endif

#ifdef SAVE_ALL_INTERNAL_DATA
        SKP_Silk_range_encoder_get_length( psRC, &nBytes_before );
#endif
        /* countour index */
        if( psEncC->fs_kHz == 8 ) {
            /* Less codevectors used in 8 khz mode */
            SKP_Silk_range_encoder( psRC, psEncCtrlC->contourIndex, SKP_Silk_pitch_contour_NB_CDF );
        } else {
            /* Joint for 12, 16, 24 khz */
            SKP_Silk_range_encoder( psRC, psEncCtrlC->contourIndex, SKP_Silk_pitch_contour_CDF );
        }
#ifdef SAVE_ALL_INTERNAL_DATA
        SKP_Silk_range_encoder_get_length( psRC, &nBytes_after ); 
        nBytes_contourIndex = nBytes_after - nBytes_before; // bytes just added
#endif

        /********************/
        /* Encode LTP gains */
        /********************/
#ifdef SAVE_ALL_INTERNAL_DATA
        SKP_Silk_range_encoder_get_length( psRC, &nBytes_before );
#endif
        /* PERIndex value */
        SKP_Silk_range_encoder( psRC, psEncCtrlC->PERIndex, SKP_Silk_LTP_per_index_CDF );

        /* Codebook Indices */
        for( k = 0; k < psEncC->nb_subfr; k++ ) {
            SKP_Silk_range_encoder( psRC, psEncCtrlC->LTPIndex[ k ], SKP_Silk_LTP_gain_CDF_ptrs[ psEncCtrlC->PERIndex ] );
        }

        /**********************/
        /* Encode LTP scaling */
        /**********************/
        SKP_Silk_range_encoder( psRC, psEncCtrlC->LTP_scaleIndex, SKP_Silk_LTPscale_CDF );
#ifdef SAVE_ALL_INTERNAL_DATA
        SKP_Silk_range_encoder_get_length( psRC, &nBytes_after );
        nBytes_LTP = nBytes_after - nBytes_before; // bytes just added
#endif
    }
#ifdef SAVE_ALL_INTERNAL_DATA
    else { 
        // Unvoiced speech
        nBytes_lagIndex     = 0;
        nBytes_contourIndex = 0;
        nBytes_LTP          = 0;
    }
    DEBUG_STORE_DATA( nBytes_lagIndex.dat,      &nBytes_lagIndex,       sizeof( SKP_int ) );
    DEBUG_STORE_DATA( nBytes_contourIndex.dat,  &nBytes_contourIndex,   sizeof( SKP_int ) );
    DEBUG_STORE_DATA( nBytes_LTP.dat,           &nBytes_LTP,            sizeof( SKP_int ) );
#endif

#ifdef SAVE_ALL_INTERNAL_DATA
    SKP_Silk_range_encoder_get_length( psRC, &nBytes_before );
#endif

    /***************/
    /* Encode seed */
    /***************/
    SKP_Silk_range_encoder( psRC, psEncCtrlC->Seed, SKP_Silk_Seed_CDF );
}
