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

#include "SKP_Silk_main_FLP.h"
    
/* Rate-Distortion calculations for multiple input data vectors */
void SKP_Silk_NLSF_VQ_rate_distortion_FLP(
          SKP_float             *pRD,               /* O   Rate-distortion values [psNLSF_CBS_FLP->nVectors*N] */
    const SKP_Silk_NLSF_CBS_FLP *psNLSF_CBS_FLP,    /* I   NLSF codebook stage struct                          */
    const SKP_float             *in,                /* I   Input vectors to be quantized                       */
    const SKP_float             *w,                 /* I   Weight vector                                       */
    const SKP_float             *rate_acc,          /* I   Accumulated rates from previous stage               */
    const SKP_float             mu,                 /* I   Weight between weighted error and rate              */
    const SKP_int               N,                  /* I   Number of input vectors to be quantized             */
    const SKP_int               LPC_order           /* I   LPC order                                           */
)
{
    SKP_float *pRD_vec;
    SKP_int   i, n;

    /* Compute weighted quantization errors for all input vectors over one codebook stage */
    SKP_Silk_NLSF_VQ_sum_error_FLP( pRD, in, w, psNLSF_CBS_FLP->CB, N, psNLSF_CBS_FLP->nVectors, LPC_order );

    /* Loop over input vectors */
    pRD_vec = pRD;
    for( n = 0; n < N; n++ ) {
        /* Add rate cost to error for each codebook vector */
        for( i = 0; i < psNLSF_CBS_FLP->nVectors; i++ ) {
            pRD_vec[ i ] += mu * ( rate_acc[n] + psNLSF_CBS_FLP->Rates[ i ] );
        }
        pRD_vec += psNLSF_CBS_FLP->nVectors;
    }
}
