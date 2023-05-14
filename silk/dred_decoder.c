/* Copyright (c) 2022 Amazon
   Written by Jan Buethe */
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

#include <string.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "os_support.h"
#include "dred_decoder.h"
#include "dred_coding.h"
#include "celt/entdec.h"


int opus_dred_get_size(void)
{
  return sizeof(OpusDRED);
}

OpusDRED *opus_dred_create(int *error)
{
  OpusDRED *dec;
  dec = (OpusDRED *)opus_alloc(opus_dred_get_size());
  if (dec == NULL)
  {
    if (error)
      *error = OPUS_ALLOC_FAIL;
    return NULL;
  }
  return dec;

}

void opus_dred_destroy(OpusDRED *dec)
{
  free(dec);
}

int dred_ec_decode(OpusDRED *dec, const opus_uint8 *bytes, int num_bytes, int min_feature_frames)
{
  const opus_uint16 *p0              = DRED_rdovae_get_p0_pointer();
  const opus_uint16 *quant_scales    = DRED_rdovae_get_quant_scales_pointer();
  const opus_uint16 *r               = DRED_rdovae_get_r_pointer();
  ec_dec ec;
  int q_level;
  int i;
  int offset;


  /* since features are decoded in quadruples, it makes no sense to go with an uneven number of redundancy frames */
  celt_assert(DRED_NUM_REDUNDANCY_FRAMES % 2 == 0);

  /* decode initial state and initialize RDOVAE decoder */
  ec_dec_init(&ec, (unsigned char*)bytes, num_bytes);
  dred_decode_state(&ec, dec->state);

  /* decode newest to oldest and store oldest to newest */
  for (i = 0; i < IMIN(DRED_NUM_REDUNDANCY_FRAMES, (min_feature_frames+1)/2); i += 2)
  {
      /* FIXME: Figure out how to avoid missing a last frame that would take up < 8 bits. */
      if (8*num_bytes - ec_tell(&ec) <= 7)
         break;
      q_level = (int) round(DRED_ENC_Q0 + 1.f * (DRED_ENC_Q1 - DRED_ENC_Q0) * i / (DRED_NUM_REDUNDANCY_FRAMES - 2));
      offset = q_level * DRED_LATENT_DIM;
      dred_decode_latents(
          &ec,
          &dec->latents[(i/2)*DRED_LATENT_DIM],
          quant_scales + offset,
          r + offset,
          p0 + offset
          );

      offset = 2 * i * DRED_NUM_FEATURES;
  }
  dec->process_stage = 1;
  dec->nb_latents = i/2;
  return i/2;
}


