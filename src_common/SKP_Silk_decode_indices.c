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

/* Decode indices from payload */
void SKP_Silk_decode_indices(
    SKP_Silk_decoder_state      *psDec,             /* I/O  State                                       */
    ec_dec                      *psRangeDec         /* I/O  Compressor data structure                   */
)
{
    SKP_int   i, k, Ix, FrameIndex;
    SKP_int   signalType, quantOffsetType, nBytesUsed;
    SKP_int   decode_absolute_lagIndex, delta_lagIndex, prev_lagIndex = 0;
    const SKP_Silk_NLSF_CB_struct *psNLSF_CB = NULL;

    for( FrameIndex = 0; FrameIndex < psDec->nFramesInPacket; FrameIndex++ ) {
        /*******************************************/
        /* Decode signal type and quantizer offset */
        /*******************************************/
        if( FrameIndex == 0 ) {
            /* first frame in packet: independent coding */
            Ix = ec_dec_icdf( psRangeDec, SKP_Silk_type_offset_iCDF, 8 );
        } else {
            /* conditional coding */
            Ix = ec_dec_icdf( psRangeDec, SKP_Silk_type_offset_joint_iCDF[ psDec->typeOffsetPrev ], 8 );
        }
        signalType            = SKP_RSHIFT( Ix, 1 );
        quantOffsetType       = Ix & 1;
        psDec->typeOffsetPrev = Ix;

        /****************/
        /* Decode gains */
        /****************/
        /* first subframe */    
        if( FrameIndex == 0 ) {
            /* first frame in packet: independent coding, in two stages: MSB bits followed by 3 LSBs */
            psDec->GainsIndices[ FrameIndex ][ 0 ] = SKP_LSHIFT( ec_dec_icdf( psRangeDec, SKP_Silk_gain_iCDF[ signalType ], 8 ), 3 );
            psDec->GainsIndices[ FrameIndex ][ 0 ] += ec_dec_icdf( psRangeDec, SKP_Silk_uniform8_iCDF, 8 );
        } else {
            /* conditional coding */
            psDec->GainsIndices[ FrameIndex ][ 0 ] = ec_dec_icdf( psRangeDec, SKP_Silk_delta_gain_iCDF, 8 );
        }

        /* remaining subframes */
        for( i = 1; i < psDec->nb_subfr; i++ ) {
            psDec->GainsIndices[ FrameIndex ][ i ] = ec_dec_icdf( psRangeDec, SKP_Silk_delta_gain_iCDF, 8 );
        }
        
        /**********************/
        /* Decode LSF Indices */
        /**********************/
        /* Set pointer to LSF VQ CB for the current signal type */
        psNLSF_CB = psDec->psNLSF_CB[ 1 - (signalType >> 1) ];

        /* Range decoding of the NLSF path */
        for( i = 0; i < psNLSF_CB->nStages; i++ ) {
            psDec->NLSFIndices[ FrameIndex ][ i ] = ec_dec_icdf( psRangeDec, psNLSF_CB->StartPtr[ i ], 8 );
        }
        
        /***********************************/
        /* Decode LSF interpolation factor */
        /***********************************/
        if( psDec->nb_subfr == MAX_NB_SUBFR ) {
            psDec->NLSFInterpCoef_Q2[ FrameIndex ] = ec_dec_icdf( psRangeDec, SKP_Silk_NLSF_interpolation_factor_iCDF, 8 );
        } else {
            psDec->NLSFInterpCoef_Q2[ FrameIndex ] = 4;
        }
        
        if( signalType == TYPE_VOICED ) {
            /*********************/
            /* Decode pitch lags */
            /*********************/
            /* Get lag index */
            decode_absolute_lagIndex = 1;
            if( FrameIndex > 0 && psDec->signalType[ FrameIndex - 1 ] == TYPE_VOICED ) {
                /* Decode Delta index */
                delta_lagIndex = ec_dec_icdf( psRangeDec, SKP_Silk_pitch_delta_iCDF, 8 );
                if( delta_lagIndex > 0 ) {
                    delta_lagIndex = delta_lagIndex - 9;
                    psDec->lagIndex[ FrameIndex ] = prev_lagIndex + delta_lagIndex;
                    decode_absolute_lagIndex = 0;
                }
            }
            if( decode_absolute_lagIndex ) {
                /* Absolute decoding */
                psDec->lagIndex[ FrameIndex ]  = ec_dec_icdf( psRangeDec, SKP_Silk_pitch_lag_iCDF, 8 ) * SKP_RSHIFT( psDec->fs_kHz, 1 );
                psDec->lagIndex[ FrameIndex ] += ec_dec_icdf( psRangeDec, psDec->pitch_lag_low_bits_iCDF, 8 );
            }
            prev_lagIndex = psDec->lagIndex[ FrameIndex ];

            /* Get countour index */
            psDec->contourIndex[ FrameIndex ] = ec_dec_icdf( psRangeDec, psDec->pitch_contour_iCDF, 8 );
            
            /********************/
            /* Decode LTP gains */
            /********************/
            /* Decode PERIndex value */
            psDec->PERIndex[ FrameIndex ] = ec_dec_icdf( psRangeDec, SKP_Silk_LTP_per_index_iCDF, 8 );

            for( k = 0; k < psDec->nb_subfr; k++ ) {
                psDec->LTPIndex[ FrameIndex ][ k ] = ec_dec_icdf( psRangeDec, SKP_Silk_LTP_gain_iCDF_ptrs[ psDec->PERIndex[ FrameIndex ] ], 8 );
            }

            /**********************/
            /* Decode LTP scaling */
            /**********************/
            psDec->LTP_scaleIndex[ FrameIndex ] = ec_dec_icdf( psRangeDec, SKP_Silk_LTPscale_iCDF, 8 );
        }

        /***************/
        /* Decode seed */
        /***************/
        psDec->Seed[ FrameIndex ] = ec_dec_icdf( psRangeDec, SKP_Silk_Seed_iCDF, 8 );

        psDec->signalType[ FrameIndex ]      = signalType;
        psDec->quantOffsetType[ FrameIndex ] = quantOffsetType;
    }

    /**************************************/
    /* Decode Frame termination indicator */
    /**************************************/
    psDec->FrameTermination = ec_dec_icdf( psRangeDec, SKP_Silk_LBRR_Present_iCDF, 8 );

    /****************************************/
    /* Get number of bytes used so far      */
    /****************************************/
    nBytesUsed = SKP_RSHIFT( ec_tell( psRangeDec ) + 7, 3 );
    psDec->nBytesLeft = psRangeDec->storage - nBytesUsed;
}
