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
void SKP_Silk_encode_indices(
    SKP_Silk_encoder_state      *psEncC,            /* I/O  Encoder state                               */
    SKP_Silk_encoder_control    *psEncCtrlC,        /* I/O  Encoder control                             */
    ec_enc                      *psRangeEnc         /* I/O  Compressor data structure                   */
)
{
    SKP_int   i, k, typeOffset;
    SKP_int   encode_absolute_lagIndex, delta_lagIndex;
    const SKP_Silk_NLSF_CB_struct *psNLSF_CB;

#ifdef SAVE_ALL_INTERNAL_DATA
    SKP_int nBytes_lagIndex, nBytes_contourIndex, nBytes_LTP;
    SKP_int nBytes_after, nBytes_before;
#endif

    /*******************************************/
    /* Encode signal type and quantizer offset */
    /*******************************************/
    typeOffset = 2 * psEncCtrlC->signalType + psEncCtrlC->quantOffsetType;
    if( psEncC->nFramesInPayloadBuf == 0 ) {
        /* first frame in packet: independent coding */
        ec_enc_icdf( psRangeEnc, typeOffset, SKP_Silk_type_offset_iCDF, 8 );
    } else {
        /* conditional coding */
        ec_enc_icdf( psRangeEnc, typeOffset, SKP_Silk_type_offset_joint_iCDF[ psEncC->typeOffsetPrev ], 8 );
    }
    psEncC->typeOffsetPrev = typeOffset;

    /****************/
    /* Encode gains */
    /****************/
#ifdef SAVE_ALL_INTERNAL_DATA
    nBytes_before = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
#endif
    /* first subframe */
    if( psEncC->nFramesInPayloadBuf == 0 ) {
        /* first frame in packet: independent coding, in two stages: MSB bits followed by 3 LSBs */
        ec_enc_icdf( psRangeEnc, SKP_RSHIFT( psEncCtrlC->GainsIndices[ 0 ], 3 ), SKP_Silk_gain_iCDF[ psEncCtrlC->signalType ], 8 );
        ec_enc_icdf( psRangeEnc, psEncCtrlC->GainsIndices[ 0 ] & 7, SKP_Silk_uniform8_iCDF, 8 );
    } else {
        /* conditional coding */
        ec_enc_icdf( psRangeEnc, psEncCtrlC->GainsIndices[ 0 ], SKP_Silk_delta_gain_iCDF, 8 );
    }

    /* remaining subframes */
    for( i = 1; i < psEncC->nb_subfr; i++ ) {
        ec_enc_icdf( psRangeEnc, psEncCtrlC->GainsIndices[ i ], SKP_Silk_delta_gain_iCDF, 8 );
    }

#ifdef SAVE_ALL_INTERNAL_DATA
    nBytes_after = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
    nBytes_after -= nBytes_before; // bytes just added
    DEBUG_STORE_DATA( nBytes_gains.dat, &nBytes_after, sizeof( SKP_int ) );
#endif

    /****************/
    /* Encode NLSFs */
    /****************/
#ifdef SAVE_ALL_INTERNAL_DATA
    nBytes_before = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
#endif
    /* Range encoding of the NLSF path */
    psNLSF_CB = psEncC->psNLSF_CB[ 1 - (psEncCtrlC->signalType>>1) ];
    for( i = 0; i < psNLSF_CB->nStages; i++ ) {
        ec_enc_icdf( psRangeEnc, psEncCtrlC->NLSFIndices[ i ], psNLSF_CB->StartPtr[ i ], 8 );
    }

    if( psEncC->nb_subfr == MAX_NB_SUBFR ) {
        /* Encode NLSF interpolation factor */
        SKP_assert( psEncC->useInterpolatedNLSFs == 1 || psEncCtrlC->NLSFInterpCoef_Q2 == ( 1 << 2 ) );
        ec_enc_icdf( psRangeEnc, psEncCtrlC->NLSFInterpCoef_Q2, SKP_Silk_NLSF_interpolation_factor_iCDF, 8 );
    }

#ifdef SAVE_ALL_INTERNAL_DATA
    DEBUG_STORE_DATA( lsf_interpol.dat, &psEncCtrlC->NLSFInterpCoef_Q2, sizeof(int) );
    nBytes_after = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
    nBytes_after -= nBytes_before; // bytes just added
    DEBUG_STORE_DATA( nBytes_LSF.dat, &nBytes_after, sizeof( SKP_int ) );
#endif

    if( psEncCtrlC->signalType == TYPE_VOICED ) {
        /*********************/
        /* Encode pitch lags */
        /*********************/
#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_before = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
#endif
        /* lag index */
        encode_absolute_lagIndex = 1;
        if( psEncC->nFramesInPayloadBuf > 0 && psEncC->prevSignalType == TYPE_VOICED ) {
            /* Delta Encoding */
            delta_lagIndex = psEncCtrlC->lagIndex - psEncC->prev_lagIndex;
            if( delta_lagIndex < -8 || delta_lagIndex > 11 ) {
                delta_lagIndex = 0;
            } else {
                delta_lagIndex = delta_lagIndex + 9;
                encode_absolute_lagIndex = 0; /* Only use delta */
            }
            SKP_assert( delta_lagIndex < 21 );
            ec_enc_icdf( psRangeEnc, delta_lagIndex, SKP_Silk_pitch_delta_iCDF, 8 );
        }
        if( encode_absolute_lagIndex ) {
            /* Absolute encoding */
            SKP_int32 pitch_high_bits, pitch_low_bits;
            pitch_high_bits = SKP_DIV32_16( psEncCtrlC->lagIndex, SKP_RSHIFT( psEncC->fs_kHz, 1 ) );
            pitch_low_bits = psEncCtrlC->lagIndex - SKP_SMULBB( pitch_high_bits, SKP_RSHIFT( psEncC->fs_kHz, 1 ) );
            SKP_assert( pitch_low_bits < psEncC->fs_kHz / 2 );
            SKP_assert( pitch_high_bits < 32 );
            ec_enc_icdf( psRangeEnc, pitch_high_bits, SKP_Silk_pitch_lag_iCDF, 8 );
            ec_enc_icdf( psRangeEnc, pitch_low_bits, psEncC->pitch_lag_low_bits_iCDF, 8 );
        }
        psEncC->prev_lagIndex = psEncCtrlC->lagIndex;

#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_after = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
        nBytes_lagIndex = nBytes_after - nBytes_before; // bytes just added
#endif

#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_before = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
#endif
        /* Countour index */
        SKP_assert( ( psEncCtrlC->contourIndex < 34 && psEncC->fs_kHz  > 8 && psEncC->nb_subfr == 4 ) ||
                    ( psEncCtrlC->contourIndex < 11 && psEncC->fs_kHz == 8 && psEncC->nb_subfr == 4 ) ||
                    ( psEncCtrlC->contourIndex < 12 && psEncC->fs_kHz  > 8 && psEncC->nb_subfr == 2 ) ||
                    ( psEncCtrlC->contourIndex <  3 && psEncC->fs_kHz == 8 && psEncC->nb_subfr == 2 ) );
        ec_enc_icdf( psRangeEnc, psEncCtrlC->contourIndex, psEncC->pitch_contour_iCDF, 8 );
#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_after = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
        nBytes_contourIndex = nBytes_after - nBytes_before; // bytes just added
#endif

        /********************/
        /* Encode LTP gains */
        /********************/
#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_before = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
#endif

        /* PERIndex value */
        ec_enc_icdf( psRangeEnc, psEncCtrlC->PERIndex, SKP_Silk_LTP_per_index_iCDF, 8 );

        /* Codebook Indices */
        for( k = 0; k < psEncC->nb_subfr; k++ ) {
            ec_enc_icdf( psRangeEnc, psEncCtrlC->LTPIndex[ k ], SKP_Silk_LTP_gain_iCDF_ptrs[ psEncCtrlC->PERIndex ], 8 );
        }

        /**********************/
        /* Encode LTP scaling */
        /**********************/
        ec_enc_icdf( psRangeEnc, psEncCtrlC->LTP_scaleIndex, SKP_Silk_LTPscale_iCDF, 8 );
#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_after = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
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
    nBytes_before = SKP_RSHIFT( ec_enc_tell( psRangeEnc, 0 ) + 7, 3 );
#endif

    /***************/
    /* Encode seed */
    /***************/
    ec_enc_icdf( psRangeEnc, psEncCtrlC->Seed, SKP_Silk_Seed_iCDF, 8 );
}
