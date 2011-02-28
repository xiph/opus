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

/* Decode indices from payload */
void SKP_Silk_decode_indices(
    SKP_Silk_decoder_state      *psDec,             /* I/O  State                                       */
    ec_dec                      *psRangeDec,        /* I/O  Compressor data structure                   */
    SKP_int                     FrameIndex,         /* I    Frame number                                */
    SKP_int                     decode_LBRR         /* I    Flag indicating LBRR data is being decoded  */
)
{
    SKP_int   i, k, Ix, condCoding;
    SKP_int   decode_absolute_lagIndex, delta_lagIndex;
    SKP_int16 ec_ix[ MAX_LPC_ORDER ];
    SKP_uint8 pred_Q8[ MAX_LPC_ORDER ];

    if( FrameIndex > 0 && ( decode_LBRR == 0 || psDec->LBRR_flags[ FrameIndex - 1 ] == 1 ) ) {
        condCoding = 1;
    } else {
        condCoding = 0;
    }

    /*******************************************/
    /* Decode signal type and quantizer offset */
    /*******************************************/
    if( decode_LBRR || psDec->VAD_flags[ FrameIndex ] ) {
        Ix = ec_dec_icdf( psRangeDec, SKP_Silk_type_offset_VAD_iCDF, 8 ) + 2;
    } else {
        Ix = ec_dec_icdf( psRangeDec, SKP_Silk_type_offset_no_VAD_iCDF, 8 );
    }
    psDec->indices.signalType      = (SKP_int8)SKP_RSHIFT( Ix, 1 );
    psDec->indices.quantOffsetType = (SKP_int8)( Ix & 1 );

    /****************/
    /* Decode gains */
    /****************/
    /* first subframe */    
    if( condCoding ) {
        /* conditional coding */
        psDec->indices.GainsIndices[ 0 ] = (SKP_int8)ec_dec_icdf( psRangeDec, SKP_Silk_delta_gain_iCDF, 8 );
    } else {
        /* independent coding, in two stages: MSB bits followed by 3 LSBs */
        psDec->indices.GainsIndices[ 0 ]  = (SKP_int8)SKP_LSHIFT( ec_dec_icdf( psRangeDec, SKP_Silk_gain_iCDF[ psDec->indices.signalType ], 8 ), 3 );
        psDec->indices.GainsIndices[ 0 ] += (SKP_int8)ec_dec_icdf( psRangeDec, SKP_Silk_uniform8_iCDF, 8 );
    }

    /* remaining subframes */
    for( i = 1; i < psDec->nb_subfr; i++ ) {
        psDec->indices.GainsIndices[ i ] = (SKP_int8)ec_dec_icdf( psRangeDec, SKP_Silk_delta_gain_iCDF, 8 );
    }
        
    /**********************/
    /* Decode LSF Indices */
    /**********************/
    psDec->indices.NLSFIndices[ 0 ] = (SKP_int8)ec_dec_icdf( psRangeDec, &psDec->psNLSF_CB->CB1_iCDF[ ( psDec->indices.signalType >> 1 ) * psDec->psNLSF_CB->nVectors ], 8 );
    SKP_Silk_NLSF_unpack( ec_ix, pred_Q8, psDec->psNLSF_CB, psDec->indices.NLSFIndices[ 0 ] );
    SKP_assert( psDec->psNLSF_CB->order == psDec->LPC_order );
    for( i = 0; i < psDec->psNLSF_CB->order; i++ ) {
        Ix = ec_dec_icdf( psRangeDec, &psDec->psNLSF_CB->ec_iCDF[ ec_ix[ i ] ], 8 );
        if( Ix == 0 ) {
            Ix -= ec_dec_icdf( psRangeDec, SKP_Silk_NLSF_EXT_iCDF, 8 );
        } else if( Ix == 2 * NLSF_QUANT_MAX_AMPLITUDE ) {
            Ix += ec_dec_icdf( psRangeDec, SKP_Silk_NLSF_EXT_iCDF, 8 );
        }
        psDec->indices.NLSFIndices[ i+1 ] = (SKP_int8)( Ix - NLSF_QUANT_MAX_AMPLITUDE );
    }

    /* Decode LSF interpolation factor */
    if( psDec->nb_subfr == MAX_NB_SUBFR ) {
        psDec->indices.NLSFInterpCoef_Q2 = (SKP_int8)ec_dec_icdf( psRangeDec, SKP_Silk_NLSF_interpolation_factor_iCDF, 8 );
    } else {
        psDec->indices.NLSFInterpCoef_Q2 = 4;
    }
        
    if( psDec->indices.signalType == TYPE_VOICED ) 
    {
        /*********************/
        /* Decode pitch lags */
        /*********************/
        /* Get lag index */
        decode_absolute_lagIndex = 1;
        if( condCoding && psDec->ec_prevSignalType == TYPE_VOICED ) {
            /* Decode Delta index */
            delta_lagIndex = (SKP_int16)ec_dec_icdf( psRangeDec, SKP_Silk_pitch_delta_iCDF, 8 );
            if( delta_lagIndex > 0 ) {
                delta_lagIndex = delta_lagIndex - 9;
                psDec->indices.lagIndex = (SKP_int16)( psDec->ec_prevLagIndex + delta_lagIndex );
                decode_absolute_lagIndex = 0;
            }
        }
        if( decode_absolute_lagIndex ) {
            /* Absolute decoding */
            psDec->indices.lagIndex  = (SKP_int16)ec_dec_icdf( psRangeDec, SKP_Silk_pitch_lag_iCDF, 8 ) * SKP_RSHIFT( psDec->fs_kHz, 1 );
            psDec->indices.lagIndex += (SKP_int16)ec_dec_icdf( psRangeDec, psDec->pitch_lag_low_bits_iCDF, 8 );
        }
        psDec->ec_prevLagIndex = psDec->indices.lagIndex;

        /* Get countour index */
        psDec->indices.contourIndex = (SKP_int8)ec_dec_icdf( psRangeDec, psDec->pitch_contour_iCDF, 8 );
            
        /********************/
        /* Decode LTP gains */
        /********************/
        /* Decode PERIndex value */
        psDec->indices.PERIndex = (SKP_int8)ec_dec_icdf( psRangeDec, SKP_Silk_LTP_per_index_iCDF, 8 );

        for( k = 0; k < psDec->nb_subfr; k++ ) {
            psDec->indices.LTPIndex[ k ] = (SKP_int8)ec_dec_icdf( psRangeDec, SKP_Silk_LTP_gain_iCDF_ptrs[ psDec->indices.PERIndex ], 8 );
        }

        /**********************/
        /* Decode LTP scaling */
        /**********************/
        if( !condCoding ) {
            psDec->indices.LTP_scaleIndex = (SKP_int8)ec_dec_icdf( psRangeDec, SKP_Silk_LTPscale_iCDF, 8 );
        } else {
            psDec->indices.LTP_scaleIndex = 0;
        }
    }
    psDec->ec_prevSignalType = psDec->indices.signalType;

    /***************/
    /* Decode seed */
    /***************/
    psDec->indices.Seed = (SKP_int8)ec_dec_icdf( psRangeDec, SKP_Silk_uniform4_iCDF, 8 );
}
