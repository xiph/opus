/* (C) 2007 Jean-Marc Valin, CSIRO
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "modes.h"

#define NBANDS 18
#define PBANDS 8
#define PITCH_END 74

#define NBANDS128 15
#define PBANDS128 8
#define PITCH_END128 45

static const float means[15] = {
   14.8621, 12.6918, 10.2978, 9.5862, 10.3784, 
   10.4555, 9.1594, 9.0280, 8.3291, 8.3410,
   8.5737, 8.5614, 9.0107, 7.6809, 7.0665};

static const float means18[18] = {
   9.9067, 8.4524, 6.8577, 6.3804, 6.1786, 5.9815,
   6.2068, 6.1076, 5.7711, 5.7734, 5.7935, 5.3981,
   5.1992, 5.7214, 5.9656, 5.7548, 5.0802, 4.2626};

static const int decay[15] = {
   14800, 13800, 12600, 12000, 11000, 10000, 9800, 8400, 8000, 7500, 7000, 7000, 7000, 6000, 6000
};

static const int decay18[18] = {
   14800, 13800, 12600, 12000, 11000, 11000, 10000, 10000, 9800, 8400, 8400, 8000, 7500, 7000, 7000, 7000, 6000, 6000
};

const int qbank0[NBANDS   +2] = {0,  4,  8, 12, 16, 20, 24, 28, 32, 38, 44, 52, 62, 74, 90,112,142,182, 232,256};
const int pbank0[PBANDS   +2] = {0,  4,  8, 12, 16,     24,         38,         62, PITCH_END, 256};
//const int pbank0[PBANDS   +2] = {0, 4, 8, 12, 19, PITCH_END, 128};
const int qpulses0[NBANDS   ] = {7,  6,  5,  5,  4,  3,  3,  3,  3,  3,  3,  2,  2,  1,  0,  0,  0,  0};
//const int qpulses0[NBANDS   ] = {7, 5, 5, 5, 4,  4,  3,  3,  3,  3,  4,  3,  3, -2,  0,  0,  0,  0};


const int qbank1[NBANDS128+2] = {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 36, 44, 52, 68, 84, 116, 128};

const int qpulses1[NBANDS128] = {7, 5, 5, 5, 4,  5,  4,  5,  5,  4,  2, 0, 0, 0,  0};
const int qpulses2[NBANDS128] = {28,24,20,16,24,20, 18, 12, 10,  10,-7, -4, 0, 0,  0};
const int qpulses2s[NBANDS128] ={38,30,24,20,24,20, 18, 16, 14, 20,-20,-14, -8, -8,  -5};

const int pbank1[PBANDS128+2] = {0, 2, 4, 6, 8, 12, 20, 28, PITCH_END128, 128};
//const int pbank1[PBANDS128+2] = {0, 4, 8, 12, 20, PITCH_END128, 128};


#define NBANDS256 15
#define PBANDS256 8
#define PITCH_END256 88
const int qbank3[NBANDS256+2] = {0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 72, 88, 104, 136, 168, 232, 256};
//const int pbank3[PBANDS256+2] = {0, 8, 16, 24, 40, PITCH_END256, 256};
const int pbank3[PBANDS256+2] = {0, 4, 8, 12, 16, 24, 40, 56, PITCH_END256, 256};

const CELTMode mode0 = {
   128,         /**< overlap */
   256,         /**< mdctSize */
   1,           /**< nbMdctBlocks */
   1,           /**< channels */
   
   NBANDS,      /**< nbEBands */
   PBANDS,      /**< nbPBands */
   PITCH_END,   /**< pitchEnd */
   
   qbank0,      /**< eBands */
   pbank0,      /**< pBands*/
   qpulses0,    /**< nbPulses */
   
   0.8,         /**< ePredCoef */
   means18,     /**< eMeans */
   decay18,     /**< eDecay */
};


/* Approx 38 kbps @ 44.1 kHz */
const CELTMode mode1 = {
   128,         /**< overlap */
   128,         /**< mdctSize */
   2,           /**< nbMdctBlocks */
   1,           /**< channels */
   
   NBANDS128,   /**< nbEBands */
   PBANDS128,   /**< nbPBands */
   PITCH_END128,/**< pitchEnd */
   
   qbank1,      /**< eBands */
   pbank1,      /**< pBands*/
   qpulses1,    /**< nbPulses */
   
   0.7,         /**< ePredCoef */
   means,       /**< eMeans */
   decay,       /**< eDecay */
};

/* Approx 58 kbps @ 44.1 kHz */
const CELTMode mode2 = {
   128,         /**< overlap */
   128,         /**< mdctSize */
   2,           /**< nbMdctBlocks */
   1,           /**< channels */
   
   NBANDS128,   /**< nbEBands */
   PBANDS128,   /**< nbPBands */
   PITCH_END128,/**< pitchEnd */
   
   qbank1,      /**< eBands */
   pbank1,      /**< pBands*/
   qpulses2,    /**< nbPulses */
   
   0.7,         /**< ePredCoef */
   means,       /**< eMeans */
   decay,       /**< eDecay */
};

const CELTMode mode3 = {
   128,         /**< overlap */
   256,         /**< mdctSize */
   1,           /**< nbMdctBlocks */
   1,           /**< channels */
   
   NBANDS256,   /**< nbEBands */
   PBANDS256,   /**< nbPBands */
   PITCH_END256,/**< pitchEnd */
   
   qbank3,      /**< eBands */
   pbank3,      /**< pBands*/
   qpulses1,    /**< nbPulses */
   
   0.7,         /**< ePredCoef */
   means,       /**< eMeans */
   decay,       /**< eDecay */
};

/* Stereo mode around 120 kbps */
const CELTMode mode4 = {
   128,         /**< overlap */
   256,         /**< mdctSize */
   1,           /**< nbMdctBlocks */
   2,           /**< channels */
   
   NBANDS256,   /**< nbEBands */
   PBANDS256,   /**< nbPBands */
   PITCH_END256,/**< pitchEnd */
   
   qbank3,      /**< eBands */
   pbank3,      /**< pBands*/
   qpulses2s,   /**< nbPulses */
   
   0.7,         /**< ePredCoef */
   means,       /**< eMeans */
   decay,       /**< eDecay */
};

const CELTMode const *celt_mode0 = &mode0;
const CELTMode const *celt_mode1 = &mode1;
const CELTMode const *celt_mode2 = &mode2;
const CELTMode const *celt_mode3 = &mode3;
const CELTMode const *celt_mode4 = &mode4;
