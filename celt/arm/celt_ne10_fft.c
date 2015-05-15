/* Copyright (c) 2015 Xiph.Org Foundation
   Written by Viswanath Puttagunta */
/**
   @file celt_ne10_fft.c
   @brief ARM Neon optimizations for fft using NE10 library
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

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef SKIP_CONFIG_H
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#endif

#include <NE10_init.h>
#include <NE10_dsp.h>
#include "os_support.h"
#include "kiss_fft.h"
#include "stack_alloc.h"

#if !defined(FIXED_POINT)
# if defined(CUSTOM_MODES)

/* nfft lengths in NE10 that support scaled fft */
#define NE10_FFTSCALED_SUPPORT_MAX 4
static const int ne10_fft_scaled_support[NE10_FFTSCALED_SUPPORT_MAX] = {
   480, 240, 120, 60
};

int opus_fft_alloc_arm_float_neon(kiss_fft_state *st)
{
   int i;
   size_t memneeded = sizeof(struct arch_fft_state);

   st->arch_fft = (arch_fft_state *)opus_alloc(memneeded);
   if (!st->arch_fft)
      return -1;

   for (i = 0; i < NE10_FFTSCALED_SUPPORT_MAX; i++) {
      if(st->nfft == ne10_fft_scaled_support[i])
         break;
   }
   if (i == NE10_FFTSCALED_SUPPORT_MAX) {
      /* This nfft length (scaled fft) is not supported in NE10 */
      st->arch_fft->is_supported = 0;
      st->arch_fft->priv = NULL;
   }
   else {
      st->arch_fft->is_supported = 1;
      st->arch_fft->priv = (void *)ne10_fft_alloc_c2c_float32_neon(st->nfft);
      if (st->arch_fft->priv == NULL) {
         return -1;
      }
   }
   return 0;
}

void opus_fft_free_arm_float_neon(kiss_fft_state *st)
{
   ne10_fft_cfg_float32_t cfg;

   if (!st->arch_fft)
      return;

   cfg = (ne10_fft_cfg_float32_t)st->arch_fft->priv;
   if (cfg)
      ne10_fft_destroy_c2c_float32(cfg);
   opus_free(st->arch_fft);
}
# endif

void opus_fft_float_neon(const kiss_fft_state *st,
                         const kiss_fft_cpx *fin,
                         kiss_fft_cpx *fout)
{
   ne10_fft_state_float32_t state;
   ne10_fft_cfg_float32_t cfg = &state;
   VARDECL(ne10_fft_cpx_float32_t, buffer);
   SAVE_STACK;
   ALLOC(buffer, st->nfft, ne10_fft_cpx_float32_t);

   if (!st->arch_fft->is_supported) {
      /* This nfft length (scaled fft) not supported in NE10 */
      opus_fft_c(st, fin, fout);
   }
   else {
      memcpy((void *)cfg, st->arch_fft->priv, sizeof(ne10_fft_state_float32_t));
      state.buffer = (ne10_fft_cpx_float32_t *)&buffer[0];
      state.is_forward_scaled = 1;

      ne10_fft_c2c_1d_float32_neon((ne10_fft_cpx_float32_t *)fout,
                                   (ne10_fft_cpx_float32_t *)fin,
                                   cfg, 0);
   }
   RESTORE_STACK;
}

void opus_ifft_float_neon(const kiss_fft_state *st,
                          const kiss_fft_cpx *fin,
                          kiss_fft_cpx *fout)
{
   ne10_fft_state_float32_t state;
   ne10_fft_cfg_float32_t cfg = &state;
   VARDECL(ne10_fft_cpx_float32_t, buffer);
   SAVE_STACK;
   ALLOC(buffer, st->nfft, ne10_fft_cpx_float32_t);

   if (!st->arch_fft->is_supported) {
      /* This nfft length (scaled fft) not supported in NE10 */
      opus_ifft_c(st, fin, fout);
   }
   else {
      memcpy((void *)cfg, st->arch_fft->priv, sizeof(ne10_fft_state_float32_t));
      state.buffer = (ne10_fft_cpx_float32_t *)&buffer[0];
      state.is_backward_scaled = 0;

      ne10_fft_c2c_1d_float32_neon((ne10_fft_cpx_float32_t *)fout,
                                   (ne10_fft_cpx_float32_t *)fin,
                                   cfg, 1);
   }
   RESTORE_STACK;
}
#endif /* !defined(FIXED_POINT) */
