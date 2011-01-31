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

#include "SKP_Silk_tables.h"

const SKP_uint8 SKP_Silk_LTP_per_index_iCDF[3] = {
	   188,    111,      0
};

const SKP_uint8 SKP_Silk_LTP_gain_iCDF_0[8] = {
	    83,     65,     48,     34,     24,     15,      7,      0
};

const SKP_uint8 SKP_Silk_LTP_gain_iCDF_1[16] = {
	   211,    186,    163,    144,    129,    115,    102,     89,
	    77,     65,     53,     42,     31,     19,      9,      0
};

const SKP_uint8 SKP_Silk_LTP_gain_iCDF_2[32] = {
	   241,    227,    214,    202,    191,    180,    170,    160,
	   149,    139,    129,    120,    110,    101,     92,     84,
	    76,     69,     61,     54,     47,     40,     35,     30,
	    26,     21,     17,     13,     10,      6,      3,      0
};

const SKP_int16 SKP_Silk_LTP_gain_middle_avg_RD_Q14 = 14008;

const SKP_int8 SKP_Silk_LTP_gain_BITS_Q4_0[8] = {
	     9,     61,     63,     66,     76,     77,     81,     83
};

const SKP_int8 SKP_Silk_LTP_gain_BITS_Q4_1[16] = {
	    40,     54,     56,     60,     65,     68,     68,     68,
	    71,     71,     72,     72,     72,     72,     76,     76
};

const SKP_int8 SKP_Silk_LTP_gain_BITS_Q4_2[32] = {
	    65,     67,     69,     70,     73,     74,     74,     74,
	    74,     75,     75,     76,     76,     76,     78,     80,
	    80,     81,     82,     82,     82,     86,     90,     91,
	    92,     95,     95,     96,     99,    100,    102,    102
};

const SKP_uint8 * const SKP_Silk_LTP_gain_iCDF_ptrs[NB_LTP_CBKS] = {
    SKP_Silk_LTP_gain_iCDF_0,
    SKP_Silk_LTP_gain_iCDF_1,
    SKP_Silk_LTP_gain_iCDF_2
};

const SKP_int8 * const SKP_Silk_LTP_gain_BITS_Q4_ptrs[NB_LTP_CBKS] = {
    SKP_Silk_LTP_gain_BITS_Q4_0,
    SKP_Silk_LTP_gain_BITS_Q4_1,
    SKP_Silk_LTP_gain_BITS_Q4_2
};

const SKP_int8 SKP_Silk_LTP_gain_vq_0[8][5] = 
{
{
	     6,      9,     28,      9,      8
},
{
	     0,      0,      2,      0,      0
},
{
	    -4,     32,     55,      5,     -1
},
{
	    -6,      6,     51,     34,      1
},
{
	    27,      8,     43,     26,    -20
},
{
	     2,      5,     11,      5,      3
},
{
	   -23,      8,     35,      4,     22
},
{
	    22,     18,     26,     -6,    -14
}
};

const SKP_int8 SKP_Silk_LTP_gain_vq_1[16][5] = 
{
{
	    10,     22,     39,     23,     17
},
{
	     5,     29,     56,     38,     -8
},
{
	    20,     40,     54,      4,     -5
},
{
	     0,      1,      7,      0,      1
},
{
	   -20,     12,     67,     46,      9
},
{
	    -9,     51,     75,     12,    -12
},
{
	     6,    -11,     84,     41,    -14
},
{
	    -8,     15,     96,      5,      3
},
{
	    41,      5,     32,      4,     10
},
{
	   -13,     50,     73,    -18,     18
},
{
	     7,    -14,     56,     47,     10
},
{
	    -3,     -2,     73,      8,     10
},
{
	   -11,     30,     42,     -7,     14
},
{
	   -23,     42,     57,     15,     21
},
{
	    -1,      0,     21,     23,     38
},
{
	    -1,     -6,     37,     20,     -3
}
};

const SKP_int8 SKP_Silk_LTP_gain_vq_2[32][5] = 
{
{
	    -6,     59,     68,      3,      0
},
{
	    -6,     36,     72,     29,     -6
},
{
	    -3,      1,     93,     24,      6
},
{
	    -4,     16,     52,     39,     19
},
{
	     0,    -11,     80,     52,      1
},
{
	     3,      4,     19,      3,      6
},
{
	    -3,     35,     98,      5,    -13
},
{
	    -9,     16,     75,     60,    -15
},
{
	     9,    -21,     84,     71,    -17
},
{
	    11,    -20,    106,     44,    -13
},
{
	     3,      9,     98,     33,    -23
},
{
	    22,     -4,     65,     51,    -15
},
{
	    -6,     11,    109,      7,      2
},
{
	    21,     38,     41,     22,      4
},
{
	   -19,     72,     83,    -22,     13
},
{
	     1,     24,     77,     -5,     17
},
{
	     4,    -12,    121,     19,     -6
},
{
	    -6,     25,    115,    -17,      7
},
{
	    27,     11,     84,     12,    -27
},
{
	   -13,     49,    105,    -24,      8
},
{
	   -21,     43,     91,     -4,     15
},
{
	     7,     49,     88,    -25,     -4
},
{
	     0,      5,    128,     -5,     -4
},
{
	    -9,     67,     95,     -4,    -23
},
{
	     6,    -10,     45,     93,     -9
},
{
	    -8,     95,     43,    -11,      6
},
{
	    17,    -19,    101,     77,    -42
},
{
	    35,     74,     19,     -7,      1
},
{
	     3,     12,    128,     27,    -39
},
{
	     1,     -2,     14,     80,     31
},
{
	    86,     10,     12,     -3,      3
},
{
	     5,     -3,     14,      7,     85
}
};

const SKP_int8 * const SKP_Silk_LTP_vq_ptrs_Q7[NB_LTP_CBKS] = {
    (SKP_int8 *)&SKP_Silk_LTP_gain_vq_0[0][0],
    (SKP_int8 *)&SKP_Silk_LTP_gain_vq_1[0][0],
    (SKP_int8 *)&SKP_Silk_LTP_gain_vq_2[0][0]
};
 
const SKP_int8 SKP_Silk_LTP_vq_sizes[NB_LTP_CBKS] = {
    8, 16, 32 
};
