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

void SKP_Silk_quant_LTP_gains(
    SKP_int16           B_Q14[ MAX_NB_SUBFR * LTP_ORDER ],              /* I/O  (un)quantized LTP gains     */
    SKP_int8            cbk_index[ MAX_NB_SUBFR ],                      /* O    Codebook Index              */
    SKP_int8            *periodicity_index,                             /* O    Periodicity Index           */
    const SKP_int32     W_Q18[ MAX_NB_SUBFR*LTP_ORDER*LTP_ORDER ],      /* I    Error Weights in Q18        */
    SKP_int             mu_Q9,                                          /* I    Mu value (R/D tradeoff)     */
    SKP_int             lowComplexity,                                  /* I    Flag for low complexity     */
    const SKP_int       nb_subfr                                        /* I    number of subframes         */
)
{
    SKP_int             j, k, cbk_size;
	SKP_int8            temp_idx[ MAX_NB_SUBFR ];
    const SKP_uint8     *cl_ptr_Q5;
    const SKP_int8      *cbk_ptr_Q7;
    const SKP_int16     *b_Q14_ptr;
    const SKP_int32     *W_Q18_ptr;
    SKP_int32           rate_dist_Q14_subfr, rate_dist_Q14, min_rate_dist_Q14;

TIC(quant_LTP)

    /***************************************************/
    /* iterate over different codebooks with different */
    /* rates/distortions, and choose best */
    /***************************************************/
    min_rate_dist_Q14 = SKP_int32_MAX;
    for( k = 0; k < 3; k++ ) {
        cl_ptr_Q5  = SKP_Silk_LTP_gain_BITS_Q5_ptrs[ k ];
        cbk_ptr_Q7 = SKP_Silk_LTP_vq_ptrs_Q7[        k ];
        cbk_size   = SKP_Silk_LTP_vq_sizes[          k ];

        /* Setup pointer to first subframe */
        W_Q18_ptr = W_Q18;
        b_Q14_ptr = B_Q14;

        rate_dist_Q14 = 0;
        for( j = 0; j < nb_subfr; j++ ) {

            SKP_Silk_VQ_WMat_EC(
                &temp_idx[ j ],         /* O    index of best codebook vector                           */
                &rate_dist_Q14_subfr,   /* O    best weighted quantization error + mu * rate            */
                b_Q14_ptr,              /* I    input vector to be quantized                            */
                W_Q18_ptr,              /* I    weighting matrix                                        */
                cbk_ptr_Q7,             /* I    codebook                                                */
                cl_ptr_Q5,              /* I    code length for each codebook vector                    */
                mu_Q9,                  /* I    tradeoff between weighted error and rate                */
                cbk_size                /* I    number of vectors in codebook                           */
            );

            rate_dist_Q14 = SKP_ADD_POS_SAT32( rate_dist_Q14, rate_dist_Q14_subfr );

            b_Q14_ptr += LTP_ORDER;
            W_Q18_ptr += LTP_ORDER * LTP_ORDER;
        }

        /* Avoid never finding a codebook */
        rate_dist_Q14 = SKP_min( SKP_int32_MAX - 1, rate_dist_Q14 );

        if( rate_dist_Q14 < min_rate_dist_Q14 ) {
            min_rate_dist_Q14 = rate_dist_Q14;
            *periodicity_index = (SKP_int8)k;
			SKP_memcpy( cbk_index, temp_idx, nb_subfr * sizeof( SKP_int8 ) );
        }

        /* Break early in low-complexity mode if rate distortion is below threshold */
        if( lowComplexity && ( rate_dist_Q14 < SKP_Silk_LTP_gain_middle_avg_RD_Q14 ) ) {
            break;
        }
    }

    cbk_ptr_Q7 = SKP_Silk_LTP_vq_ptrs_Q7[ *periodicity_index ];
    for( j = 0; j < nb_subfr; j++ ) {
        for( k = 0; k < LTP_ORDER; k++ ) { 
            B_Q14[ j * LTP_ORDER + k ] = SKP_LSHIFT( cbk_ptr_Q7[ cbk_index[ j ] * LTP_ORDER + k ], 7 );
        }
    }
TOC(quant_LTP)
}

