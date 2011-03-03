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

#ifndef QUANT_BANDS
#define QUANT_BANDS

#include "arch.h"
#include "modes.h"
#include "entenc.h"
#include "entdec.h"
#include "mathops.h"

void amp2Log2(const CELTMode *m, int effEnd, int end,
      celt_ener *bandE, celt_word16 *bandLogE, int _C);

void log2Amp(const CELTMode *m, int start, int end,
      celt_ener *eBands, celt_word16 *oldEBands, int _C);

unsigned char *quant_prob_alloc(const CELTMode *m);
void quant_prob_free(const celt_int16 *freq);

void quant_coarse_energy(const CELTMode *m, int start, int end, int effEnd,
      const celt_word16 *eBands, celt_word16 *oldEBands, celt_uint32 budget,
      celt_word16 *error, ec_enc *enc, int _C, int LM,
      int nbAvailableBytes, int force_intra, int *delayedIntra, int two_pass);

void quant_fine_energy(const CELTMode *m, int start, int end, celt_word16 *oldEBands, celt_word16 *error, int *fine_quant, ec_enc *enc, int _C);

void quant_energy_finalise(const CELTMode *m, int start, int end, celt_word16 *oldEBands, celt_word16 *error, int *fine_quant, int *fine_priority, int bits_left, ec_enc *enc, int _C);

void unquant_coarse_energy(const CELTMode *m, int start, int end, celt_word16 *oldEBands, int intra, ec_dec *dec, int _C, int LM);

void unquant_fine_energy(const CELTMode *m, int start, int end, celt_word16 *oldEBands, int *fine_quant, ec_dec *dec, int _C);

void unquant_energy_finalise(const CELTMode *m, int start, int end, celt_word16 *oldEBands, int *fine_quant, int *fine_priority, int bits_left, ec_dec *dec, int _C);

#endif /* QUANT_BANDS */
