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

#include "SKP_Silk_main_FLP.h"

/***********************/
/* NLSF vector encoder */
/***********************/
void SKP_Silk_NLSF_MSVQ_encode_FLP(
          SKP_int8                  *NLSFIndices,       /* O    Codebook path vector [ CB_STAGES ]      */
          SKP_float                 *pNLSF,             /* I/O  Quantized NLSF vector [ LPC_ORDER ]     */
    const SKP_Silk_NLSF_CB_struct   *psNLSF_CB,         /* I    Codebook object                         */
    const SKP_float                 *pNLSF_q_prev,      /* I    Prev. quantized NLSF vector [LPC_ORDER] */
    const SKP_float                 *pW,                /* I    NLSF weight vector [ LPC_ORDER ]        */
    const SKP_float                 NLSF_mu,            /* I    Rate weight for the RD optimization     */
    const SKP_float                 NLSF_mu_fluc_red,   /* I    Fluctuation reduction error weight      */
    const SKP_int                   NLSF_MSVQ_Survivors,/* I    Max survivors from each stage           */
    const SKP_int                   LPC_order,          /* I    LPC order                               */
    const SKP_int                   deactivate_fluc_red /* I    Deactivate fluctuation reduction        */
)
{
    SKP_int     i, s, k, cur_survivors, prev_survivors, min_survivors, input_index, cb_index, bestIndex;
    SKP_float   rateDistThreshold;
#if( NLSF_MSVQ_FLUCTUATION_REDUCTION == 1 )
    SKP_float   se, wsse, bestRateDist;
#endif

    SKP_float   pRateDist[      NLSF_MSVQ_TREE_SEARCH_MAX_VECTORS_EVALUATED ];
    SKP_float   pRate[          MAX_NLSF_MSVQ_SURVIVORS ];
    SKP_float   pRate_new[      MAX_NLSF_MSVQ_SURVIVORS ];
    SKP_int     pTempIndices[   MAX_NLSF_MSVQ_SURVIVORS ];
    SKP_int8    pPath[          MAX_NLSF_MSVQ_SURVIVORS * NLSF_MSVQ_MAX_CB_STAGES ];
    SKP_int8    pPath_new[      MAX_NLSF_MSVQ_SURVIVORS * NLSF_MSVQ_MAX_CB_STAGES ];
    SKP_float   pRes_Q8[        MAX_NLSF_MSVQ_SURVIVORS * MAX_LPC_ORDER ];
    SKP_float   pRes_Q8_new[    MAX_NLSF_MSVQ_SURVIVORS * MAX_LPC_ORDER ];

    const SKP_float *pConstFloat;
          SKP_float *pFloat;
    const SKP_int8  *pConstInt8;
          SKP_int8  *pInt8;
    const SKP_int8  *pCB_element;
    const SKP_Silk_NLSF_CBS *pCurrentCBStage;

    SKP_assert( NLSF_MSVQ_Survivors <= MAX_NLSF_MSVQ_SURVIVORS );

#ifdef SAVE_ALL_INTERNAL_DATA
    DEBUG_STORE_DATA( NLSF.dat,    pNLSF,    LPC_order * sizeof( SKP_float ) );
    DEBUG_STORE_DATA( WNLSF.dat,   pW,       LPC_order * sizeof( SKP_float ) );
    DEBUG_STORE_DATA( NLSF_mu.dat, &NLSF_mu,             sizeof( SKP_float ) );
#endif

    cur_survivors = NLSF_MSVQ_Survivors;

    /****************************************************/
    /* Tree search for the multi-stage vector quantizer */
    /****************************************************/

    /* Clear accumulated rates */
    SKP_memset( pRate, 0, NLSF_MSVQ_Survivors * sizeof( SKP_float ) );

    /* Subtract 1/2 from NLSF input vector to create initial residual, and scale to Q8 */
    for( i = 0; i < LPC_order; i++ ) {
        pRes_Q8[ i ] = ( pNLSF[ i ] - 0.5f ) * 256.0f;        
    }

    /* Set first stage values */
    prev_survivors = 1;

    /* Minimum number of survivors */
    min_survivors = NLSF_MSVQ_Survivors / 2;

    /* Loop over all stages */
    for( s = 0; s < psNLSF_CB->nStages; s++ ) {

        /* Set a pointer to the current stage codebook */
        pCurrentCBStage = &psNLSF_CB->CBStages[ s ];

        /* Calculate the number of survivors in the current stage */
        cur_survivors = SKP_min_32( NLSF_MSVQ_Survivors, prev_survivors * pCurrentCBStage->nVectors );

#if( NLSF_MSVQ_FLUCTUATION_REDUCTION == 0 )
        /* Find a single best survivor in the last stage, if we */
        /* do not need candidates for fluctuation reduction     */
        if( s == psNLSF_CB->nStages - 1 ) {
            cur_survivors = 1;
        }
#endif
        /* Nearest neighbor clustering for multiple input data vectors */
        SKP_Silk_NLSF_VQ_rate_distortion_FLP( pRateDist, pCurrentCBStage, pRes_Q8, pW, pRate, NLSF_mu, prev_survivors, LPC_order );

        /* Sort the rate-distortion errors */
        SKP_Silk_insertion_sort_increasing_FLP( pRateDist, pTempIndices, prev_survivors * pCurrentCBStage->nVectors, cur_survivors );

        /* Discard survivors with rate-distortion values too far above the best one */
        rateDistThreshold = ( 1.0f + NLSF_MSVQ_Survivors * NLSF_MSVQ_SURV_MAX_REL_RD ) * pRateDist[ 0 ];
        while( pRateDist[ cur_survivors - 1 ] > rateDistThreshold && cur_survivors > min_survivors ) {
            cur_survivors--;
        }

        /* Update accumulated codebook contributions for the 'cur_survivors' best codebook indices */
        for( k = 0; k < cur_survivors; k++ ) { 
            if( s > 0 ) {
                /* Find the indices of the input and the codebook vector */
                if( pCurrentCBStage->nVectors == 8 ) {
                    input_index = SKP_RSHIFT( pTempIndices[ k ], 3 );
                    cb_index    = pTempIndices[ k ] & 7;
                } else {
                    input_index = pTempIndices[ k ] / pCurrentCBStage->nVectors;  
                    cb_index    = pTempIndices[ k ] - input_index * pCurrentCBStage->nVectors;
                }
            } else {
                /* Find the indices of the input and the codebook vector */
                input_index = 0;
                cb_index    = pTempIndices[ k ];
            }

            /* Subtract new contribution from the previous residual vector for each of 'cur_survivors' */
            pConstFloat = &pRes_Q8[ input_index * LPC_order ];
            pCB_element = &pCurrentCBStage->CB_NLSF_Q8[ cb_index * LPC_order ];
            pFloat      = &pRes_Q8_new[ k * LPC_order ];
            for( i = 0; i < LPC_order; i++ ) {
                pFloat[ i ] = pConstFloat[ i ] - pCB_element[ i ];
            }

            /* Update accumulated rate for stage 1 to the current */
            pRate_new[ k ] = pRate[ input_index ] + 0.0625f * ( SKP_float )pCurrentCBStage->Rates_Q4[ cb_index ];

            /* Copy paths from previous matrix, starting with the best path */
            pConstInt8 = &pPath[ input_index * psNLSF_CB->nStages ];
            pInt8      = &pPath_new[       k * psNLSF_CB->nStages ];
            for( i = 0; i < s; i++ ) {
                pInt8[ i ] = pConstInt8[ i ];
            }
            /* Write the current stage indices for the 'cur_survivors' to the best path matrix */
            pInt8[ s ] = (SKP_int8)cb_index;
        }

        if( s < psNLSF_CB->nStages - 1 ) {
            /* Copy NLSF residual matrix for next stage */
            SKP_memcpy( pRes_Q8, pRes_Q8_new, cur_survivors * LPC_order * sizeof( SKP_float ) );

            /* Copy rate vector for next stage */
            SKP_memcpy( pRate, pRate_new, cur_survivors * sizeof( SKP_float ) );

            /* Copy best path matrix for next stage */
            SKP_memcpy( pPath, pPath_new, cur_survivors * psNLSF_CB->nStages * sizeof( SKP_int8) );
        }

        prev_survivors = cur_survivors;
    }

    /* (Preliminary) index of the best survivor, later to be decoded */
    bestIndex = 0;

#if( NLSF_MSVQ_FLUCTUATION_REDUCTION == 1 )
    /******************************/
    /* NLSF fluctuation reduction */
    /******************************/
    if( deactivate_fluc_red != 1 ) {
    
        /* Search among all survivors, now taking also weighted fluctuation errors into account */
        bestRateDist = SKP_float_MAX;
        for( s = 0; s < cur_survivors; s++ ) {
            /* Decode survivor to compare with previous quantized NLSF vector */
            SKP_Silk_NLSF_MSVQ_decode_FLP( pNLSF, psNLSF_CB, &pPath_new[ s * psNLSF_CB->nStages ], LPC_order );

            /* Compare decoded NLSF vector with the previously quantized vector */ 
            wsse = 0;
            for( i = 0; i < LPC_order; i += 2 ) {
                /* Compute weighted squared quantization error for index i */
                se = pNLSF[ i ] - pNLSF_q_prev[ i ];
                wsse += pW[ i ] * se * se;

                /* Compute weighted squared quantization error for index i + 1 */
                se = pNLSF[ i + 1 ] - pNLSF_q_prev[ i + 1 ];
                wsse += pW[ i + 1 ] * se * se;
            }

            /* Add the fluctuation reduction penalty to the rate distortion error */
            wsse = pRateDist[s] + wsse * NLSF_mu_fluc_red;

            /* Keep index of best survivor */
            if( wsse < bestRateDist ) {
                bestRateDist = wsse;
                bestIndex = s;
            }
        }
    }
#endif

    /* Copy best path to output argument */
    SKP_memcpy( NLSFIndices, &pPath_new[ bestIndex * psNLSF_CB->nStages ], psNLSF_CB->nStages * sizeof( SKP_int8 ) );

    /* Decode and stabilize the best survivor */
    SKP_Silk_NLSF_MSVQ_decode_FLP( pNLSF, psNLSF_CB, NLSFIndices, LPC_order );

}
