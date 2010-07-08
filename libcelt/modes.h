/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2008 Gregory Maxwell 
   Written by Jean-Marc Valin and Gregory Maxwell */
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

#ifndef MODES_H
#define MODES_H

#include "celt_types.h"
#include "celt.h"
#include "arch.h"
#include "mdct.h"
#include "pitch.h"
#include "entenc.h"
#include "entdec.h"

#define MAX_CONFIG_SIZES 5

#define CELT_BITSTREAM_VERSION 0x8000000c

#ifdef STATIC_MODES
#include "static_modes.h"
#endif

#define MAX_PERIOD 1024

#ifndef MCHANNELS
# ifdef DISABLE_STEREO
#  define MCHANNELS(mode) (1)
# else
#  define MCHANNELS(mode) ((mode)->nbChannels)
# endif
#endif

#ifndef CHANNELS
# ifdef DISABLE_STEREO
#  define CHANNELS(_C) (1)
# else
#  define CHANNELS(_C) (_C)
# endif
#endif

#ifndef OVERLAP
#define OVERLAP(mode) ((mode)->overlap)
#endif

#ifndef FRAMESIZE
#define FRAMESIZE(mode) ((mode)->mdctSize)
#endif

/** Mode definition (opaque)
 @brief Mode definition 
 */
struct CELTMode {
   celt_uint32 marker_start;
   celt_int32 Fs;
   int          overlap;

   int          nbEBands;
   int          pitchEnd;
   
   const celt_int16   *eBands;   /**< Definition for each "pseudo-critical band" */
   
   celt_word16 ePredCoef;/**< Prediction coefficient for the energy encoding */
   
   int          nbAllocVectors; /**< Number of lines in the matrix below */
   const unsigned char   *allocVectors;   /**< Number of bits in each band for several rates */
   
   const celt_int16 * const **bits;
   const celt_int16 * const *(_bits[MAX_CONFIG_SIZES]); /**< Cache for pulses->bits mapping in each band */

   /* Stuff that could go in the {en,de}coder, but we save space this way */
   mdct_lookup mdct;

   const celt_word16 *window;

   int         maxLM;
   int         nbShortMdcts;
   int         shortMdctSize;

   int *prob;
   const celt_int16 *logN;
   celt_uint32 marker_end;
};

int check_mode(const CELTMode *mode);

/* Prototypes for _ec versions of the encoder/decoder calls (not public) */
int celt_encode_with_ec(CELTEncoder * restrict st, const celt_int16 * pcm, celt_int16 * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc);
int celt_encode_with_ec_float(CELTEncoder * restrict st, const float * pcm, float * optional_resynthesis, int frame_size, unsigned char *compressed, int nbCompressedBytes, ec_enc *enc);
int celt_decode_with_ec(CELTDecoder * restrict st, const unsigned char *data, int len, celt_int16 * restrict pcm, int frame_size, ec_dec *dec);
int celt_decode_with_ec_float(CELTDecoder * restrict st, const unsigned char *data, int len, float * restrict pcm, int frame_size, ec_dec *dec);
#endif
