/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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
#ifndef PSY_H
#define PSY_H

#include "arch.h"
#include "celt.h"

struct PsyDecay {
   /*celt_word16 *decayL;*/
   const celt_word16 * restrict decayR;
};

/** Pre-compute the decay of the psycho-acoustic spreading function */
void psydecay_init(struct PsyDecay *decay, int len, celt_int32 Fs);

/** Free the memory allocated for the spreading function */
void psydecay_clear(struct PsyDecay *decay);

/** Compute the masking curve for an input (DFT) spectrum X */
void compute_masking(const struct PsyDecay *decay, celt_word16 *X, celt_mask *mask, int len);

/** Compute the masking curve for an input (MDCT) spectrum X */
void compute_mdct_masking(const struct PsyDecay *decay, celt_word32 *X, celt_word16 *tonality, celt_word16 *long_window, celt_mask *mask, int len);

void compute_tonality(const CELTMode *m, celt_word16 * restrict X, celt_word16 * mem, int len, celt_word16 *tonality, int mdct_size);

#endif /* PSY_H */
