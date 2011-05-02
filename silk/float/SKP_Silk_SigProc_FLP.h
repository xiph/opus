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


#ifndef _SKP_SILK_SIGPROC_FLP_H_
#define _SKP_SILK_SIGPROC_FLP_H_

#include "SKP_Silk_SigProc_FIX.h"
#include "float_cast.h"
#include <math.h>

#ifdef  __cplusplus
extern "C"
{
#endif

/********************************************************************/
/*                    SIGNAL PROCESSING FUNCTIONS                   */
/********************************************************************/

/* Chirp (bw expand) LP AR filter */
void SKP_Silk_bwexpander_FLP( 
    SKP_float *ar,                     /* io   AR filter to be expanded (without leading 1)    */
    const SKP_int d,                   /* i	length of ar                                       */
    const SKP_float chirp              /* i	chirp factor (typically in range (0..1) )          */
);

/* compute inverse of LPC prediction gain, and							*/
/* test if LPC coefficients are stable (all poles within unit circle)	*/
/* this code is based on SKP_Silk_FLP_a2k()								*/
SKP_int SKP_Silk_LPC_inverse_pred_gain_FLP( /* O:   returns 1 if unstable, otherwise 0    */
    SKP_float            *invGain,      /* O:   inverse prediction gain, energy domain	  */
    const SKP_float      *A,            /* I:   prediction coefficients [order]           */
    SKP_int32            order          /* I:   prediction order                          */
);

SKP_float SKP_Silk_schur_FLP(           /* O    returns residual energy                     */
    SKP_float       refl_coef[],        /* O    reflection coefficients (length order)      */
    const SKP_float auto_corr[],        /* I    autocorrelation sequence (length order+1)   */
    SKP_int         order               /* I    order                                       */
);

void SKP_Silk_k2a_FLP(
    SKP_float           *A,             /* O:	prediction coefficients [order]           */
    const SKP_float     *rc,            /* I:	reflection coefficients [order]           */
    SKP_int32           order           /* I:	prediction order                          */
);

/* Solve the normal equations using the Levinson-Durbin recursion */
SKP_float SKP_Silk_levinsondurbin_FLP(	/* O	prediction error energy						*/
	SKP_float		A[],				/* O	prediction coefficients	[order]				*/
	const SKP_float corr[],				/* I	input auto-correlations [order + 1]			*/
	const SKP_int	order				/* I	prediction order 							*/
);

/* compute autocorrelation */
void SKP_Silk_autocorrelation_FLP( 
    SKP_float *results,                 /* o    result (length correlationCount)            */
    const SKP_float *inputData,         /* i    input data to correlate                     */
    SKP_int inputDataSize,              /* i    length of input                             */
    SKP_int correlationCount            /* i    number of correlation taps to compute       */
);

/* Pitch estimator */
#define SigProc_PE_MIN_COMPLEX        0
#define SigProc_PE_MID_COMPLEX        1
#define SigProc_PE_MAX_COMPLEX        2

SKP_int SKP_Silk_pitch_analysis_core_FLP( /* O voicing estimate: 0 voiced, 1 unvoiced                       */
    const SKP_float *signal,            /* I signal of length PE_FRAME_LENGTH_MS*Fs_kHz                     */
    SKP_int         *pitch_out,         /* O 4 pitch lag values                                             */
    SKP_int16       *lagIndex,          /* O lag Index                                                      */
    SKP_int8        *contourIndex,      /* O pitch contour Index                                            */
    SKP_float       *LTPCorr,           /* I/O normalized correlation; input: value from previous frame     */
    SKP_int         prevLag,            /* I last lag of previous frame; set to zero is unvoiced            */
    const SKP_float search_thres1,      /* I first stage threshold for lag candidates 0 - 1                 */
    const SKP_float search_thres2,      /* I final threshold for lag candidates 0 - 1                       */
    const SKP_int   Fs_kHz,             /* I sample frequency (kHz)                                         */
    const SKP_int   complexity,         /* I Complexity setting, 0-2, where 2 is highest                    */
    const SKP_int   nb_subfr            /* I    number of 5 ms subframes                                    */
);

#define PI               (3.1415926536f)

void SKP_Silk_insertion_sort_decreasing_FLP(
    SKP_float            *a,            /* I/O:  Unsorted / Sorted vector                */
    SKP_int              *index,        /* O:    Index vector for the sorted elements    */
    const SKP_int        L,             /* I:    Vector length                           */
    const SKP_int        K              /* I:    Number of correctly sorted positions    */
);

/* Compute reflection coefficients from input signal */
SKP_float SKP_Silk_burg_modified_FLP(       /* O    returns residual energy                                         */
    SKP_float           A[],                /* O    prediction coefficients (length order)                          */
    const SKP_float     x[],                /* I    input signal, length: nb_subfr*(D+L_sub)                        */
    const SKP_int       subfr_length,       /* I    input signal subframe length (including D preceeding samples)   */
    const SKP_int       nb_subfr,           /* I    number of subframes stacked in x                                */
    const SKP_float     WhiteNoiseFrac,     /* I    fraction added to zero-lag autocorrelation                      */
    const SKP_int       D                   /* I    order                                                           */
);

/* multiply a vector by a constant */
void SKP_Silk_scale_vector_FLP( 
    SKP_float           *data1,
    SKP_float           gain, 
    SKP_int             dataSize
);

/* copy and multiply a vector by a constant */
void SKP_Silk_scale_copy_vector_FLP( 
    SKP_float           *data_out, 
    const SKP_float     *data_in, 
    SKP_float           gain, 
    SKP_int             dataSize
);

/* inner product of two SKP_float arrays, with result as double */
double SKP_Silk_inner_product_FLP( 
    const SKP_float     *data1, 
    const SKP_float     *data2, 
    SKP_int             dataSize
);

/* sum of squares of a SKP_float array, with result as double */
double SKP_Silk_energy_FLP( 
    const SKP_float     *data, 
    SKP_int             dataSize
);

/********************************************************************/
/*                                MACROS                                */
/********************************************************************/

#define SKP_min_float(a, b)			(((a) < (b)) ? (a) :  (b)) 
#define SKP_max_float(a, b)			(((a) > (b)) ? (a) :  (b)) 
#define SKP_abs_float(a)			((SKP_float)fabs(a))

#define SKP_LIMIT_float( a, limit1, limit2)	((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a))) \
															     : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))))

/* sigmoid function */
SKP_INLINE SKP_float SKP_sigmoid(SKP_float x)
{
    return (SKP_float)(1.0 / (1.0 + exp(-x)));
}

/* floating-point to integer conversion (rounding) */
#if 1
/* use implementation in float_cast.h */
#define SKP_float2int(x)   float2int(x)
#else
SKP_INLINE SKP_int32 SKP_float2int(SKP_float x) 
{
    double y = x;
    return (SKP_int32)( ( y > 0 ) ? y + 0.5 : y - 0.5 );
}
#endif

/* floating-point to integer conversion (rounding) */
SKP_INLINE void SKP_float2short_array(
    SKP_int16       *out, 
    const SKP_float *in, 
    SKP_int32       length
) 
{
    SKP_int32 k;
    for (k = length-1; k >= 0; k--) {
        out[k] = (SKP_int16)SKP_SAT16( float2int( in[k] ) );
    }
}

/* integer to floating-point conversion */
SKP_INLINE void SKP_short2float_array(
    SKP_float       *out, 
    const SKP_int16 *in, 
    SKP_int32       length
) 
{
    SKP_int32 k;
    for (k = length-1; k >= 0; k--) {
        out[k] = (SKP_float)in[k];
    }
}

#define SKP_round(x)		(SKP_float)((x)>=0 ? (SKP_int64)((x)+0.5) : (SKP_int64)((x)-0.5))

#ifdef  __cplusplus
}
#endif

#endif
