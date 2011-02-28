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

/*******************************************/
/* Encode parameters to create the payload */
/*******************************************/
void SKP_Silk_encode_indices(
    SKP_Silk_encoder_state      *psEncC,            /* I/O  Encoder state                               */
    ec_enc                      *psRangeEnc,        /* I/O  Compressor data structure                   */
    SKP_int                     FrameIndex,         /* I    Frame number                                */
    SKP_int                     encode_LBRR         /* I    Flag indicating LBRR data is being encoded  */
)
{
    SKP_int   i, k, condCoding, typeOffset;
    SKP_int   encode_absolute_lagIndex, delta_lagIndex;
    SKP_int16 ec_ix[ MAX_LPC_ORDER ];
    SKP_uint8 pred_Q8[ MAX_LPC_ORDER ];
    const SideInfoIndices *psIndices;
#if SAVE_ALL_INTERNAL_DATA
    SKP_int nBytes_lagIndex, nBytes_contourIndex, nBytes_LTP;
    SKP_int nBytes_after, nBytes_before;
#endif

    if( FrameIndex > 0 && ( encode_LBRR == 0 || psEncC->LBRR_flags[ FrameIndex - 1 ] == 1 ) ) {
        condCoding = 1;
    } else {
        condCoding = 0;
    }

    if( encode_LBRR ) {
         psIndices = &psEncC->indices_LBRR[ FrameIndex ];
    } else {
         psIndices = &psEncC->indices;
    }

    /*******************************************/
    /* Encode signal type and quantizer offset */
    /*******************************************/
    typeOffset = 2 * psIndices->signalType + psIndices->quantOffsetType;
    SKP_assert( typeOffset >= 0 && typeOffset < 6 );
    SKP_assert( encode_LBRR == 0 || typeOffset >= 2 );
    if( encode_LBRR || typeOffset >= 2 ) {
        ec_enc_icdf( psRangeEnc, typeOffset - 2, SKP_Silk_type_offset_VAD_iCDF, 8 );
    } else {
        ec_enc_icdf( psRangeEnc, typeOffset, SKP_Silk_type_offset_no_VAD_iCDF, 8 );
    }

    /****************/
    /* Encode gains */
    /****************/
#ifdef SAVE_ALL_INTERNAL_DATA
    nBytes_before = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
#endif
    /* first subframe */
    if( condCoding ) {
        /* conditional coding */
        SKP_assert( psIndices->GainsIndices[ 0 ] >= 0 && psIndices->GainsIndices[ 0 ] < MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT + 1 );
        ec_enc_icdf( psRangeEnc, psIndices->GainsIndices[ 0 ], SKP_Silk_delta_gain_iCDF, 8 );
    } else {
        /* independent coding, in two stages: MSB bits followed by 3 LSBs */
        SKP_assert( psIndices->GainsIndices[ 0 ] >= 0 && psIndices->GainsIndices[ 0 ] < N_LEVELS_QGAIN );
        ec_enc_icdf( psRangeEnc, SKP_RSHIFT( psIndices->GainsIndices[ 0 ], 3 ), SKP_Silk_gain_iCDF[ psIndices->signalType ], 8 );
        ec_enc_icdf( psRangeEnc, psIndices->GainsIndices[ 0 ] & 7, SKP_Silk_uniform8_iCDF, 8 );
    }

    /* remaining subframes */
    for( i = 1; i < psEncC->nb_subfr; i++ ) {
        SKP_assert( psIndices->GainsIndices[ i ] >= 0 && psIndices->GainsIndices[ i ] < MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT + 1 );
        ec_enc_icdf( psRangeEnc, psIndices->GainsIndices[ i ], SKP_Silk_delta_gain_iCDF, 8 );
    }

#ifdef SAVE_ALL_INTERNAL_DATA
    nBytes_after = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
    nBytes_after -= nBytes_before; // bytes just added
    DEBUG_STORE_DATA( nBytes_gains.dat, &nBytes_after, sizeof( SKP_int ) );
#endif

    /****************/
    /* Encode NLSFs */
    /****************/
#ifdef SAVE_ALL_INTERNAL_DATA
    nBytes_before = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
#endif
    ec_enc_icdf( psRangeEnc, psIndices->NLSFIndices[ 0 ], &psEncC->psNLSF_CB->CB1_iCDF[ ( psIndices->signalType >> 1 ) * psEncC->psNLSF_CB->nVectors ], 8 );
    SKP_Silk_NLSF_unpack( ec_ix, pred_Q8, psEncC->psNLSF_CB, psIndices->NLSFIndices[ 0 ] );
    SKP_assert( psEncC->psNLSF_CB->order == psEncC->predictLPCOrder );
    for( i = 0; i < psEncC->psNLSF_CB->order; i++ ) {
        if( psIndices->NLSFIndices[ i+1 ] >= NLSF_QUANT_MAX_AMPLITUDE ) {
            ec_enc_icdf( psRangeEnc, 2 * NLSF_QUANT_MAX_AMPLITUDE, &psEncC->psNLSF_CB->ec_iCDF[ ec_ix[ i ] ], 8 );
            ec_enc_icdf( psRangeEnc, psIndices->NLSFIndices[ i+1 ] - NLSF_QUANT_MAX_AMPLITUDE, SKP_Silk_NLSF_EXT_iCDF, 8 );
        } else if( psIndices->NLSFIndices[ i+1 ] <= -NLSF_QUANT_MAX_AMPLITUDE ) {
            ec_enc_icdf( psRangeEnc, 0, &psEncC->psNLSF_CB->ec_iCDF[ ec_ix[ i ] ], 8 );
            ec_enc_icdf( psRangeEnc, -psIndices->NLSFIndices[ i+1 ] - NLSF_QUANT_MAX_AMPLITUDE, SKP_Silk_NLSF_EXT_iCDF, 8 );
        } else {
            ec_enc_icdf( psRangeEnc, psIndices->NLSFIndices[ i+1 ] + NLSF_QUANT_MAX_AMPLITUDE, &psEncC->psNLSF_CB->ec_iCDF[ ec_ix[ i ] ], 8 );
        }
    }

    /* Encode NLSF interpolation factor */
    if( psEncC->nb_subfr == MAX_NB_SUBFR ) {
        SKP_assert( psEncC->useInterpolatedNLSFs == 1 || psIndices->NLSFInterpCoef_Q2 == ( 1 << 2 ) );
        SKP_assert( psIndices->NLSFInterpCoef_Q2 >= 0 && psIndices->NLSFInterpCoef_Q2 < 5 );
        ec_enc_icdf( psRangeEnc, psIndices->NLSFInterpCoef_Q2, SKP_Silk_NLSF_interpolation_factor_iCDF, 8 );
    }

#ifdef SAVE_ALL_INTERNAL_DATA
    DEBUG_STORE_DATA( lsf_interpol.dat, &psIndices->NLSFInterpCoef_Q2, sizeof(int) );
    nBytes_after = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
    nBytes_after -= nBytes_before; // bytes just added
    DEBUG_STORE_DATA( nBytes_LSF.dat, &nBytes_after, sizeof( SKP_int ) );
#endif

    if( psIndices->signalType == TYPE_VOICED ) 
    {
        /*********************/
        /* Encode pitch lags */
        /*********************/
#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_before = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
#endif
        /* lag index */
        encode_absolute_lagIndex = 1;
        if( condCoding && psEncC->ec_prevSignalType == TYPE_VOICED ) {
            /* Delta Encoding */
            delta_lagIndex = psIndices->lagIndex - psEncC->ec_prevLagIndex;
            if( delta_lagIndex < -8 || delta_lagIndex > 11 ) {
                delta_lagIndex = 0;
            } else {
                delta_lagIndex = delta_lagIndex + 9;
                encode_absolute_lagIndex = 0; /* Only use delta */
            }
            SKP_assert( delta_lagIndex >= 0 && delta_lagIndex < 21 );
            ec_enc_icdf( psRangeEnc, delta_lagIndex, SKP_Silk_pitch_delta_iCDF, 8 );
        }
        if( encode_absolute_lagIndex ) {
            /* Absolute encoding */
            SKP_int32 pitch_high_bits, pitch_low_bits;
            pitch_high_bits = SKP_DIV32_16( psIndices->lagIndex, SKP_RSHIFT( psEncC->fs_kHz, 1 ) );
            pitch_low_bits = psIndices->lagIndex - SKP_SMULBB( pitch_high_bits, SKP_RSHIFT( psEncC->fs_kHz, 1 ) );
            SKP_assert( pitch_low_bits < psEncC->fs_kHz / 2 );
            SKP_assert( pitch_high_bits < 32 );
            ec_enc_icdf( psRangeEnc, pitch_high_bits, SKP_Silk_pitch_lag_iCDF, 8 );
            ec_enc_icdf( psRangeEnc, pitch_low_bits, psEncC->pitch_lag_low_bits_iCDF, 8 );
        }
        psEncC->ec_prevLagIndex = psIndices->lagIndex;

#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_after = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
        nBytes_lagIndex = nBytes_after - nBytes_before; // bytes just added
#endif

#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_before = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
#endif
        /* Countour index */
        SKP_assert(   psIndices->contourIndex  >= 0 );
        SKP_assert( ( psIndices->contourIndex < 34 && psEncC->fs_kHz  > 8 && psEncC->nb_subfr == 4 ) ||
                    ( psIndices->contourIndex < 11 && psEncC->fs_kHz == 8 && psEncC->nb_subfr == 4 ) ||
                    ( psIndices->contourIndex < 12 && psEncC->fs_kHz  > 8 && psEncC->nb_subfr == 2 ) ||
                    ( psIndices->contourIndex <  3 && psEncC->fs_kHz == 8 && psEncC->nb_subfr == 2 ) );
        ec_enc_icdf( psRangeEnc, psIndices->contourIndex, psEncC->pitch_contour_iCDF, 8 );
#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_after = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
        nBytes_contourIndex = nBytes_after - nBytes_before; // bytes just added
#endif

        /********************/
        /* Encode LTP gains */
        /********************/
#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_before = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
#endif

        /* PERIndex value */
        SKP_assert( psIndices->PERIndex >= 0 && psIndices->PERIndex < 3 );
        ec_enc_icdf( psRangeEnc, psIndices->PERIndex, SKP_Silk_LTP_per_index_iCDF, 8 );

        /* Codebook Indices */
        for( k = 0; k < psEncC->nb_subfr; k++ ) {
            SKP_assert( psIndices->LTPIndex[ k ] >= 0 && psIndices->LTPIndex[ k ] < ( 8 << psIndices->PERIndex ) );
            ec_enc_icdf( psRangeEnc, psIndices->LTPIndex[ k ], SKP_Silk_LTP_gain_iCDF_ptrs[ psIndices->PERIndex ], 8 );
        }

        /**********************/
        /* Encode LTP scaling */
        /**********************/
        if( !condCoding ) {
            SKP_assert( psIndices->LTP_scaleIndex >= 0 && psIndices->LTP_scaleIndex < 3 );
            ec_enc_icdf( psRangeEnc, psIndices->LTP_scaleIndex, SKP_Silk_LTPscale_iCDF, 8 );
        }
        SKP_assert( !condCoding || psIndices->LTP_scaleIndex == 0 );

#ifdef SAVE_ALL_INTERNAL_DATA
        nBytes_after = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
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

    psEncC->ec_prevSignalType = psIndices->signalType;

#ifdef SAVE_ALL_INTERNAL_DATA
    nBytes_before = SKP_RSHIFT( ec_tell( psRangeEnc ) + 7, 3 );
#endif

    /***************/
    /* Encode seed */
    /***************/
    SKP_assert( psIndices->Seed >= 0 && psIndices->Seed < 4 );
    ec_enc_icdf( psRangeEnc, psIndices->Seed, SKP_Silk_uniform4_iCDF, 8 );
}
