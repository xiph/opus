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

/* Predictive dequantizer for NLSF residuals */
void SKP_Silk_NLSF_residual_dequant(                        /* O    Returns RD value in Q30                     */
          SKP_int16         x_Q10[],                        /* O    Output [ order ]                            */
    const SKP_int8          indices[],                      /* I    Quantization indices [ order ]              */
    const SKP_uint8         pred_coef_Q8[],                 /* I    Backward predictor coefs [ order ]          */
    const SKP_int           quant_step_size_Q16,            /* I    Quantization step size                      */
    const SKP_int16         order                           /* I    Number of input values                      */
)
{
    SKP_int     i, out_Q10, pred_Q10;
    
    out_Q10 = 0;
    for( i = order-1; i >= 0; i-- ) {
        pred_Q10 = SKP_RSHIFT( SKP_SMULBB( out_Q10, (SKP_int16)pred_coef_Q8[ i ] ), 8 );
        out_Q10  = SKP_LSHIFT( indices[ i ], 10 );
        if( out_Q10 > 0 ) {
            out_Q10 = SKP_SUB16( out_Q10, SKP_FIX_CONST( NLSF_QUANT_LEVEL_ADJ, 10 ) );
        } else if( out_Q10 < 0 ) {
            out_Q10 = SKP_ADD16( out_Q10, SKP_FIX_CONST( NLSF_QUANT_LEVEL_ADJ, 10 ) );
        }
        out_Q10  = SKP_SMLAWB( pred_Q10, out_Q10, quant_step_size_Q16 );
        x_Q10[ i ] = out_Q10;
    }
}


/***********************/
/* NLSF vector decoder */
/***********************/
void SKP_Silk_NLSF_decode(
          SKP_int16                 *pNLSF_Q15,             /* O    Quantized NLSF vector [ LPC_ORDER ]     */
          SKP_int8                  *NLSFIndices,           /* I    Codebook path vector [ LPC_ORDER + 1 ]  */
    const SKP_Silk_NLSF_CB_struct   *psNLSF_CB              /* I    Codebook object                         */
)
{
    SKP_int         i;
    SKP_uint8       pred_Q8[  MAX_LPC_ORDER ];
    SKP_int16       ec_ix[    MAX_LPC_ORDER ];
    SKP_int16       res_Q10[  MAX_LPC_ORDER ];
    SKP_int16       W_tmp_Q5[ MAX_LPC_ORDER ];
    SKP_int32       W_tmp_Q9;
    const SKP_uint8 *pCB_element;

    /* Decode first stage */
    pCB_element = &psNLSF_CB->CB1_NLSF_Q8[ NLSFIndices[ 0 ] * psNLSF_CB->order ];
    for( i = 0; i < psNLSF_CB->order; i++ ) {
        pNLSF_Q15[ i ] = SKP_LSHIFT( ( SKP_int16 )pCB_element[ i ], 7 );
    }

    /* Unpack entropy table indices and predictor for current CB1 index */
    SKP_Silk_NLSF_unpack( ec_ix, pred_Q8, psNLSF_CB, NLSFIndices[ 0 ] );

    /* Trellis dequantizer */
    SKP_Silk_NLSF_residual_dequant( res_Q10, &NLSFIndices[ 1 ], pred_Q8, psNLSF_CB->quantStepSize_Q16, psNLSF_CB->order );

    /* Weights from codebook vector */
    SKP_Silk_NLSF_VQ_weights_laroia( W_tmp_Q5, pNLSF_Q15, psNLSF_CB->order );

    /* Apply inverse square-rooted weights and add to output */
    for( i = 0; i < psNLSF_CB->order; i++ ) {
        W_tmp_Q9 = SKP_Silk_SQRT_APPROX( SKP_LSHIFT( ( SKP_int32 )W_tmp_Q5[ i ], 13 ) );
        pNLSF_Q15[ i ] = SKP_ADD16( pNLSF_Q15[ i ], (SKP_int16)SKP_DIV32_16( SKP_LSHIFT( ( SKP_int32 )res_Q10[ i ], 14 ), W_tmp_Q9 ) );
    }

    /* NLSF stabilization */
    SKP_Silk_NLSF_stabilize( pNLSF_Q15, psNLSF_CB->deltaMin_Q15, psNLSF_CB->order );
}
