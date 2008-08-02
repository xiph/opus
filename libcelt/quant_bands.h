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

#ifndef QUANT_BANDS
#define QUANT_BANDS

#include "arch.h"
#include "modes.h"
#include "entenc.h"
#include "entdec.h"

int *quant_prob_alloc(const CELTMode *m);
void quant_prob_free(int *freq);

void compute_fine_allocation(const CELTMode *m, int *bits, int budget);

void quant_coarse_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int budget, int *prob, celt_word16_t *error, ec_enc *enc);

void quant_fine_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, celt_word16_t *error, int *fine_quant, ec_enc *enc);

void unquant_coarse_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int budget, int *prob, ec_dec *dec);

void unquant_fine_energy(const CELTMode *m, celt_ener_t *eBands, celt_word16_t *oldEBands, int *fine_quant, ec_dec *dec);

#endif /* QUANT_BANDS */
