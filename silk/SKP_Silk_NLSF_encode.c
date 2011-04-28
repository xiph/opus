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

#define STORE_LSF_DATA_FOR_TRAINING          0

/***********************/
/* NLSF vector encoder */
/***********************/
SKP_int32 SKP_Silk_NLSF_encode(                             /* O    Returns RD value in Q25                 */
          SKP_int8                  *NLSFIndices,           /* I    Codebook path vector [ LPC_ORDER + 1 ]  */
          SKP_int16                 *pNLSF_Q15,             /* I/O  Quantized NLSF vector [ LPC_ORDER ]     */
    const SKP_Silk_NLSF_CB_struct   *psNLSF_CB,             /* I    Codebook object                         */
    const SKP_int16                 *pW_Q5,                 /* I    NLSF weight vector [ LPC_ORDER ]        */
    const SKP_int                   NLSF_mu_Q20,            /* I    Rate weight for the RD optimization     */
    const SKP_int                   nSurvivors,             /* I    Max survivors after first stage         */
    const SKP_int                   signalType              /* I    Signal type: 0/1/2                      */
)
{
    SKP_int         i, s, ind1, bestIndex, prob_Q8, bits_q7;
    SKP_int32       W_tmp_Q9;
    SKP_int32       err_Q26[      NLSF_VQ_MAX_VECTORS ];
    SKP_int32       RD_Q25[       NLSF_VQ_MAX_SURVIVORS ];
    SKP_int         tempIndices1[ NLSF_VQ_MAX_SURVIVORS ];
    SKP_int8        tempIndices2[ NLSF_VQ_MAX_SURVIVORS * MAX_LPC_ORDER ];
    SKP_int16       res_Q15[      MAX_LPC_ORDER ];
    SKP_int16       res_Q10[      MAX_LPC_ORDER ];
    SKP_int16       NLSF_tmp_Q15[ MAX_LPC_ORDER ];
    SKP_int16       W_tmp_Q5[     MAX_LPC_ORDER ];
    SKP_int16       W_adj_Q5[     MAX_LPC_ORDER ];
    SKP_uint8       pred_Q8[      MAX_LPC_ORDER ];
    SKP_int16       ec_ix[        MAX_LPC_ORDER ];
    const SKP_uint8 *pCB_element, *iCDF_ptr;

#if STORE_LSF_DATA_FOR_TRAINING
    SKP_int16       pNLSF_Q15_orig[MAX_LPC_ORDER ];
    DEBUG_STORE_DATA( NLSF.dat,    pNLSF_Q15,    psNLSF_CB->order * sizeof( SKP_int16 ) );
    DEBUG_STORE_DATA( WNLSF.dat,   pW_Q5,        psNLSF_CB->order * sizeof( SKP_int16 ) );
    DEBUG_STORE_DATA( NLSF_mu.dat, &NLSF_mu_Q20,                    sizeof( SKP_int   ) );
    DEBUG_STORE_DATA( sigType.dat, &signalType,                     sizeof( SKP_int   ) );
    SKP_memcpy(pNLSF_Q15_orig, pNLSF_Q15, sizeof( pNLSF_Q15_orig ));
#endif

    SKP_assert( nSurvivors <= NLSF_VQ_MAX_SURVIVORS );
    SKP_assert( signalType >= 0 && signalType <= 2 );
    SKP_assert( NLSF_mu_Q20 <= 32767 && NLSF_mu_Q20 >= 0 );

    /* NLSF stabilization */
    SKP_Silk_NLSF_stabilize( pNLSF_Q15, psNLSF_CB->deltaMin_Q15, psNLSF_CB->order );

    /* First stage: VQ */
    SKP_Silk_NLSF_VQ( err_Q26, pNLSF_Q15, psNLSF_CB->CB1_NLSF_Q8, psNLSF_CB->nVectors, psNLSF_CB->order );

    /* Sort the quantization errors */
    SKP_Silk_insertion_sort_increasing( err_Q26, tempIndices1, psNLSF_CB->nVectors, nSurvivors );

    /* Loop over survivors */
    for( s = 0; s < nSurvivors; s++ ) {
        ind1 = tempIndices1[ s ]; 

        /* Residual after first stage */
        pCB_element = &psNLSF_CB->CB1_NLSF_Q8[ ind1 * psNLSF_CB->order ];
        for( i = 0; i < psNLSF_CB->order; i++ ) {
            NLSF_tmp_Q15[ i ] = SKP_LSHIFT16( ( SKP_int16 )pCB_element[ i ], 7 );
            res_Q15[ i ] = pNLSF_Q15[ i ] - NLSF_tmp_Q15[ i ];
        }

        /* Weights from codebook vector */
        SKP_Silk_NLSF_VQ_weights_laroia( W_tmp_Q5, NLSF_tmp_Q15, psNLSF_CB->order );

        /* Apply square-rooted weights */
        for( i = 0; i < psNLSF_CB->order; i++ ) {
            W_tmp_Q9 = SKP_Silk_SQRT_APPROX( SKP_LSHIFT( ( SKP_int32 )W_tmp_Q5[ i ], 13 ) );
            res_Q10[ i ] = ( SKP_int16 )SKP_RSHIFT( SKP_SMULBB( res_Q15[ i ], W_tmp_Q9 ), 14 );
        }

        /* Modify input weights accordingly */
        for( i = 0; i < psNLSF_CB->order; i++ ) {
            W_adj_Q5[ i ] = SKP_DIV32_16( SKP_LSHIFT( ( SKP_int32 )pW_Q5[ i ], 5 ), W_tmp_Q5[ i ] );
        }

        /* Unpack entropy table indices and predictor for current CB1 index */
        SKP_Silk_NLSF_unpack( ec_ix, pred_Q8, psNLSF_CB, ind1 );

        /* Trellis quantizer */
        RD_Q25[ s ] = SKP_Silk_NLSF_del_dec_quant( &tempIndices2[ s * MAX_LPC_ORDER ], res_Q10, W_adj_Q5, pred_Q8, ec_ix, 
            psNLSF_CB->ec_Rates_Q5, psNLSF_CB->quantStepSize_Q16, psNLSF_CB->invQuantStepSize_Q6, NLSF_mu_Q20, psNLSF_CB->order );

        /* Add rate for first stage */
        iCDF_ptr = &psNLSF_CB->CB1_iCDF[ ( signalType >> 1 ) * psNLSF_CB->nVectors ];
        if( ind1 == 0 ) {
            prob_Q8 = 256 - iCDF_ptr[ ind1 ];
        } else {
            prob_Q8 = iCDF_ptr[ ind1 - 1 ] - iCDF_ptr[ ind1 ];
        }
        bits_q7 = ( 8 << 7 ) - SKP_Silk_lin2log( prob_Q8 );
        RD_Q25[ s ] = SKP_SMLABB( RD_Q25[ s ], bits_q7, SKP_RSHIFT( NLSF_mu_Q20, 2 ) );
    }

    /* Find the lowest rate-distortion error */
    SKP_Silk_insertion_sort_increasing( RD_Q25, &bestIndex, nSurvivors, 1 );

    NLSFIndices[ 0 ] = ( SKP_int8 )tempIndices1[ bestIndex ];
    SKP_memcpy( &NLSFIndices[ 1 ], &tempIndices2[ bestIndex * MAX_LPC_ORDER ], psNLSF_CB->order * sizeof( SKP_int8 ) );

    /* Decode */
    SKP_Silk_NLSF_decode( pNLSF_Q15, NLSFIndices, psNLSF_CB );

#if STORE_LSF_DATA_FOR_TRAINING
    {
		/* code for training the codebooks */
        SKP_int32 RD_dec_Q22, Dist_Q22_dec, Rate_Q7, diff_Q15;
        ind1 = NLSFIndices[ 0 ];
        SKP_Silk_NLSF_unpack( ec_ix, pred_Q8, psNLSF_CB, ind1 );

        pCB_element = &psNLSF_CB->CB1_NLSF_Q8[ ind1 * psNLSF_CB->order ];
        for( i = 0; i < psNLSF_CB->order; i++ ) {
            NLSF_tmp_Q15[ i ] = SKP_LSHIFT16( ( SKP_int16 )pCB_element[ i ], 7 );
        }
        SKP_Silk_NLSF_VQ_weights_laroia( W_tmp_Q5, NLSF_tmp_Q15, psNLSF_CB->order );
        for( i = 0; i < psNLSF_CB->order; i++ ) {
            W_tmp_Q9 = SKP_Silk_SQRT_APPROX( SKP_LSHIFT( ( SKP_int32 )W_tmp_Q5[ i ], 13 ) );
            res_Q15[ i ] = pNLSF_Q15_orig[ i ] - NLSF_tmp_Q15[ i ];
            res_Q10[ i ] = (SKP_int16)SKP_RSHIFT( SKP_SMULBB( res_Q15[ i ], W_tmp_Q9 ), 14 );
            DEBUG_STORE_DATA( NLSF_res_q10.dat, &res_Q10[ i ], sizeof( SKP_int16 ) );
            res_Q15[ i ] = pNLSF_Q15[ i ] - NLSF_tmp_Q15[ i ];
            res_Q10[ i ] = (SKP_int16)SKP_RSHIFT( SKP_SMULBB( res_Q15[ i ], W_tmp_Q9 ), 14 );
            DEBUG_STORE_DATA( NLSF_resq_q10.dat, &res_Q10[ i ], sizeof( SKP_int16 ) );
        }

        Dist_Q22_dec = 0;
        for( i = 0; i < psNLSF_CB->order; i++ ) {
            diff_Q15 = pNLSF_Q15_orig[ i ] - pNLSF_Q15[ i ];
            Dist_Q22_dec += ( ( (diff_Q15 >> 5) * (diff_Q15 >> 5) ) * pW_Q5[ i ] ) >> 3;
        }
        iCDF_ptr = &psNLSF_CB->CB1_iCDF[ ( signalType >> 1 ) * psNLSF_CB->nVectors ];
        if( ind1 == 0 ) {
            prob_Q8 = 256 - iCDF_ptr[ ind1 ];
        } else {
            prob_Q8 = iCDF_ptr[ ind1 - 1 ] - iCDF_ptr[ ind1 ];
        }
        Rate_Q7 = ( 8 << 7 ) - SKP_Silk_lin2log( prob_Q8 );
        for( i = 0; i < psNLSF_CB->order; i++ ) {
            Rate_Q7 += ((int)psNLSF_CB->ec_Rates_Q5[ ec_ix[ i ] + SKP_LIMIT( NLSFIndices[ i + 1 ] + NLSF_QUANT_MAX_AMPLITUDE, 0, 2 * NLSF_QUANT_MAX_AMPLITUDE ) ] ) << 2;
            if( SKP_abs( NLSFIndices[ i + 1 ] ) >= NLSF_QUANT_MAX_AMPLITUDE ) {
                Rate_Q7 += 128 << ( SKP_abs( NLSFIndices[ i + 1 ] ) - NLSF_QUANT_MAX_AMPLITUDE );
            }
        }
        RD_dec_Q22 = Dist_Q22_dec + Rate_Q7 * NLSF_mu_Q20 >> 5;
        DEBUG_STORE_DATA( dec_dist_q22.dat, &Dist_Q22_dec, sizeof( SKP_int32 ) );
        DEBUG_STORE_DATA( dec_rate_q7.dat, &Rate_Q7, sizeof( SKP_int32 ) );
        DEBUG_STORE_DATA( dec_rd_q22.dat, &RD_dec_Q22, sizeof( SKP_int32 ) );
    }
    DEBUG_STORE_DATA( NLSF_ind.dat, NLSFIndices, (psNLSF_CB->order+1) * sizeof( SKP_int8 ) );
#endif

    return RD_Q25[ 0 ];
}
