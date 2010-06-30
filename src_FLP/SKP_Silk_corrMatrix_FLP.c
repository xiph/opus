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

/**********************************************************************
 * Correlation matrix computations for LS estimate. 
 **********************************************************************/

#include "SKP_Silk_main_FLP.h"

/* Calculates correlation vector X'*t */
void SKP_Silk_corrVector_FLP(
    const SKP_float                 *x,                 /* I    x vector [L+order-1] used to create X   */
    const SKP_float                 *t,                 /* I    Target vector [L]                       */
    const SKP_int                   L,                  /* I    Length of vecors                        */
    const SKP_int                   Order,              /* I    Max lag for correlation                 */
          SKP_float                 *Xt                 /* O    X'*t correlation vector [order]         */
)
{
    SKP_int lag;
    const SKP_float *ptr1;
    
    ptr1 = &x[ Order - 1 ];                     /* Points to first sample of column 0 of X: X[:,0] */
    for( lag = 0; lag < Order; lag++ ) {
        /* Calculate X[:,lag]'*t */
        Xt[ lag ] = (SKP_float)SKP_Silk_inner_product_FLP( ptr1, t, L );
        ptr1--;                                 /* Next column of X */
    }   
}

/* Calculates correlation matrix X'*X */
void SKP_Silk_corrMatrix_FLP(
    const SKP_float                 *x,                 /* I    x vector [ L+order-1 ] used to create X */
    const SKP_int                   L,                  /* I    Length of vectors                       */
    const SKP_int                   Order,              /* I    Max lag for correlation                 */
          SKP_float                 *XX                 /* O    X'*X correlation matrix [order x order] */
)
{
    SKP_int j, lag;
    double  energy;
    const SKP_float *ptr1, *ptr2;

    ptr1 = &x[ Order - 1 ];                     /* First sample of column 0 of X */
    energy = SKP_Silk_energy_FLP( ptr1, L );  /* X[:,0]'*X[:,0] */
    matrix_ptr( XX, 0, 0, Order ) = ( SKP_float )energy;
    for( j = 1; j < Order; j++ ) {
        /* Calculate X[:,j]'*X[:,j] */
        energy += ptr1[ -j ] * ptr1[ -j ] - ptr1[ L - j ] * ptr1[ L - j ];
        matrix_ptr( XX, j, j, Order ) = ( SKP_float )energy;
    }
 
    ptr2 = &x[ Order - 2 ];                     /* First sample of column 1 of X */
    for( lag = 1; lag < Order; lag++ ) {
        /* Calculate X[:,0]'*X[:,lag] */
        energy = SKP_Silk_inner_product_FLP( ptr1, ptr2, L );   
        matrix_ptr( XX, lag, 0, Order ) = ( SKP_float )energy;
        matrix_ptr( XX, 0, lag, Order ) = ( SKP_float )energy;
        /* Calculate X[:,j]'*X[:,j + lag] */
        for( j = 1; j < ( Order - lag ); j++ ) {
            energy += ptr1[ -j ] * ptr2[ -j ] - ptr1[ L - j ] * ptr2[ L - j ];
            matrix_ptr( XX, lag + j, j, Order ) = ( SKP_float )energy;
            matrix_ptr( XX, j, lag + j, Order ) = ( SKP_float )energy;
        }
        ptr2--;                                 /* Next column of X */
    }
}
