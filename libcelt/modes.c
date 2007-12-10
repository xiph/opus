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

#define NBANDS128 15
#define PBANDS128 5
#define PITCH_END128 36

const int qbank1[NBANDS128+2] =   {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 36, 44, 52, 68, 84, 116, 128};

const int qpulses1[NBANDS128] =   {7, 5, 5, 5, 4,  5,  4,  5,  5,  4, -2, 0, 0, 0,  0};
const int qpulses2[NBANDS128] =   {28,24,20,16,24,20, 18, 12, 10,  10,-7, -4, 0, 0,  0};

const int pbank1[PBANDS128+2] =   {0, 4, 8, 12, 20, PITCH_END128, 128};

const int qbank3[NBANDS128+2] =   {0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 72, 88, 104, 126, 168, 232, 256};
#define PITCH_END256 72
const int pbank3[PBANDS128+2] =   {0, 8, 16, 24, 40, PITCH_END256, 256};

/* Approx 38 kbps @ 44.1 kHz */
const CELTMode mode1 = {
   256,         /**< frameSize */
   128,         /**< mdctSize */
   2,           /**< nbMdctBlocks */
   1,           /**< channels */
   
   NBANDS128,   /**< nbEBands */
   PBANDS128,   /**< nbPBands */
   PITCH_END128,/**< pitchEnd */
   
   qbank1,      /**< eBands */
   pbank1,      /**< pBands*/
   qpulses1     /**< nbPulses */
};

/* Approx 58 kbps @ 44.1 kHz */
const CELTMode mode2 = {
   256,         /**< frameSize */
   128,         /**< mdctSize */
   2,           /**< nbMdctBlocks */
   1,           /**< channels */
   
   NBANDS128,   /**< nbEBands */
   PBANDS128,   /**< nbPBands */
   PITCH_END128,/**< pitchEnd */
   
   qbank1,      /**< eBands */
   pbank1,      /**< pBands*/
   qpulses2     /**< nbPulses */
};

const CELTMode mode3 = {
   512,         /**< frameSize */
   256,         /**< mdctSize */
   2,           /**< nbMdctBlocks */
   1,           /**< channels */
   
   NBANDS128,   /**< nbEBands */
   PBANDS128,   /**< nbPBands */
   PITCH_END256,/**< pitchEnd */
   
   qbank3,      /**< eBands */
   pbank3,      /**< pBands*/
   qpulses2     /**< nbPulses */
};

const CELTMode mode4 = {
   256,         /**< frameSize */
   128,         /**< mdctSize */
   2,           /**< nbMdctBlocks */
   2,           /**< channels */
   
   NBANDS128,   /**< nbEBands */
   PBANDS128,   /**< nbPBands */
   PITCH_END128,/**< pitchEnd */
   
   qbank1,      /**< eBands */
   pbank1,      /**< pBands*/
   qpulses2     /**< nbPulses */
};

const CELTMode const *celt_mode1 = &mode1;
const CELTMode const *celt_mode2 = &mode2;
const CELTMode const *celt_mode3 = &mode3;
const CELTMode const *celt_mode4 = &mode4;
