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

#include "SKP_Silk_structs.h"
#include "SKP_Silk_define.h"
#include "SKP_Silk_tables.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Piece-wise linear mapping from bitrate in kbps to coding quality in dB SNR */
const SKP_uint16 TargetRate_table_NB[ TARGET_RATE_TAB_SZ ] = {
    0,      8000,   9000,   11000,  13000,  16000,  21000,  MAX_TARGET_RATE_BPS
};
const SKP_uint16 TargetRate_table_MB[ TARGET_RATE_TAB_SZ ] = {
    0,      10000,  12000,  14000,  17000,  21000,  27000,  MAX_TARGET_RATE_BPS
};
const SKP_uint16 TargetRate_table_WB[ TARGET_RATE_TAB_SZ ] = {
    0,      11000,  14000,  17000,  21000,  26000,  35000,  MAX_TARGET_RATE_BPS
};
const SKP_uint16 SNR_table_Q1[ TARGET_RATE_TAB_SZ ] = {
    19,     31,     35,     39,     44,     49,     55,     60
};

/* Decoder high-pass filter coefficients, -6 dB @ 50 Hz, 0.05 dB ripple */
const SKP_int32 SKP_Silk_Dec_A_HP_16[ DEC_HP_ORDER ]     = {-527234079,  258986528};            /* second order AR coefs, Q28 */
const SKP_int32 SKP_Silk_Dec_B_HP_16[ DEC_HP_ORDER + 1 ] = { 263603618, -527207236, 263603618}; /* second order MA coefs, Q28 */

const SKP_int32 SKP_Silk_Dec_A_HP_12[ DEC_HP_ORDER ]     = {-524058488,  255953207};            /* second order AR coefs, Q28 */
const SKP_int32 SKP_Silk_Dec_B_HP_12[ DEC_HP_ORDER + 1 ] = { 261993005, -523986010, 261993005}; /* second order MA coefs, Q28 */

const SKP_int32 SKP_Silk_Dec_A_HP_8[ DEC_HP_ORDER   ]    = {-517610668,  249913410};            /* second order AR coefs, Q28 */
const SKP_int32 SKP_Silk_Dec_B_HP_8[ DEC_HP_ORDER + 1 ]  = { 258905997, -517811995, 258905997}; /* second order MA coefs, Q28 */

/* table for LSB coding */
const SKP_uint8 SKP_Silk_lsb_iCDF[ 2 ] = { 100, 0 };

/* tables for LTPScale */
const SKP_uint8 SKP_Silk_LTPscale_iCDF[ 3 ] = { 128, 64, 0 };

/* tables for VAD flag */
const SKP_uint8  SKP_Silk_vadflag_iCDF[ 2 ] = { 100,  0 };

/* tables for signal type and offset coding */
const SKP_uint8 SKP_Silk_type_offset_iCDF[6] = {
	   253,    222,    202,    138,      7,      0
};
const SKP_uint8 SKP_Silk_type_offset_joint_iCDF[6][6] = 
{
{
	    67,     34,      8,      5,      4,      0
},
{
	   156,     42,     16,     11,     10,      0
},
{
	   154,    140,     55,     29,     15,      0
},
{
	   169,    156,    109,     23,     18,      0
},
{
	   156,    150,    136,    129,     68,      0
},
{
	   195,    187,    169,    147,    141,      0
}
};


/* tables for NLSF interpolation factor */
const SKP_uint8 SKP_Silk_NLSF_interpolation_factor_iCDF[ 5 ] = { 243, 221, 192, 181, 0 };

/* Table for frame termination indication */
const SKP_uint8 SKP_Silk_LBRR_Present_iCDF[ 2 ] = { 100, 0 };

/* Table for random seed */
const SKP_uint8 SKP_Silk_Seed_iCDF[ 4 ] = { 192, 128, 64, 0 };

/* Quantization offsets */
const SKP_int16  SKP_Silk_Quantization_Offsets_Q10[ 2 ][ 2 ] = {
    { OFFSET_UVL_Q10, OFFSET_UVH_Q10 }, { OFFSET_VL_Q10, OFFSET_VH_Q10 }
};

/* Table for LTPScale */
const SKP_int16 SKP_Silk_LTPScales_table_Q14[ 3 ] = { 15565, 11469, 8192 };

/* Uniform entropy tables */
const SKP_uint8 SKP_Silk_uniform4_iCDF[   4 ] = { 192, 128, 64, 0 };
const SKP_uint8 SKP_Silk_uniform6_iCDF[   6 ] = { 213, 171, 128, 85, 43, 0 };
const SKP_uint8 SKP_Silk_uniform8_iCDF[   8 ] = { 224, 192, 160, 128, 96, 64, 32, 0 };

#if SWITCH_TRANSITION_FILTERING
/*  Elliptic/Cauer filters designed with 0.1 dB passband ripple, 
        80 dB minimum stopband attenuation, and
        [0.95 : 0.15 : 0.35] normalized cut off frequencies. */

/* Interpolation points for filter coefficients used in the bandwidth transition smoother */
const SKP_int32 SKP_Silk_Transition_LP_B_Q28[ TRANSITION_INT_NUM ][ TRANSITION_NB ] = 
{
{    250767114,  501534038,  250767114  },
{    209867381,  419732057,  209867381  },
{    170987846,  341967853,  170987846  },
{    131531482,  263046905,  131531482  },
{     89306658,  178584282,   89306658  }
};

/* Interpolation points for filter coefficients used in the bandwidth transition smoother */
const SKP_int32 SKP_Silk_Transition_LP_A_Q28[ TRANSITION_INT_NUM ][ TRANSITION_NA ] = 
{
{    506393414,  239854379  },
{    411067935,  169683996  },
{    306733530,  116694253  },
{    185807084,   77959395  },
{     35497197,   57401098  }
};
#endif

#ifdef __cplusplus
}
#endif

