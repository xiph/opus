/* Copyright (c) 2022 Amazon
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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdio.h>

#include "celt/entenc.h"
#include "celt/laplace.h"
#include "os_support.h"
#include "dred_config.h"
#include "dred_coding.h"

#define LATENT_DIM 80
#define STATE_DIM 80

int compute_quantizer(int q0, int dQ, int i) {
  int quant;
  static const int dQ_table[8] = {0, 2, 3, 4, 6, 8, 12, 16};
  quant = 0;
  quant = q0 + (dQ_table[dQ]*i + 8)/16;
  return quant > 15 ? 15 : quant;
  return (int) floor(0.5f + DRED_ENC_Q0 + 1.f * (DRED_ENC_Q1 - DRED_ENC_Q0) * i / (DRED_NUM_REDUNDANCY_FRAMES - 2));
}

void dred_encode_latents(ec_enc *enc, const float *x, const opus_uint16 *scale, const opus_uint16 *dzone, const opus_uint16 *r, const opus_uint16 *p0) {
    int i;
    float eps = .1f;
    for (i=0;i<LATENT_DIM;i++) {
        float delta;
        float xq;
        int q;
        delta = dzone[i]*(1.f/1024.f);
        xq = x[i]*scale[i]*(1.f/256.f);
        xq = xq - delta*tanh(xq/(delta+eps));
        q = (int)floor(.5f+xq);
        /* Make the impossible actually impossible. */
        if (r[i] == 0 || p0[i] >= 32767) q = 0;
        ec_laplace_encode_p0(enc, q, p0[i], r[i]);
    }
}

void dred_decode_latents(ec_dec *dec, float *x, const opus_uint16 *scale, const opus_uint16 *r, const opus_uint16 *p0) {
    int i;
    for (i=0;i<LATENT_DIM;i++) {
        int q;
        q = ec_laplace_decode_p0(dec, p0[i], r[i]);
        x[i] = q*256.f/(scale[i] == 0 ? 1 : scale[i]);
    }
}

#if 0
#include <stdlib.h>

#define DATA_SIZE 10000

int main()
{
    ec_enc enc;
    ec_dec dec;
    int iter;
    int bytes;
    opus_int16 scale[LATENT_DIM];
    opus_int16 dzone[LATENT_DIM];
    opus_int16 r[LATENT_DIM];
    opus_int16 p0[LATENT_DIM];
    unsigned char *ptr;
    int k;

    for (k=0;k<LATENT_DIM;k++) {
        scale[k] = 256;
        dzone[k] = 0;
        r[k] = 12054;
        p0[k] = 12893;
    }
    ptr = (unsigned char *)malloc(DATA_SIZE);
    ec_enc_init(&enc,ptr,DATA_SIZE);
    for (iter=0;iter<1;iter++) {
        float x[PVQ_DIM];
        float sum=1e-30;
        for (k=0;k<PVQ_DIM;k++) {
            x[k] = log(1e-15+(float)rand()/RAND_MAX)-log(1e-15+(float)rand()/RAND_MAX);
            sum += fabs(x[k]);
        }
        for (k=0;k<PVQ_DIM;k++) x[k] *= (1.f/sum);
        /*for (k=0;k<PVQ_DIM;k++) printf("%f ", x[k]);
        printf("\n");*/
        dred_encode_state(&enc, x);
    }
    for (iter=0;iter<1;iter++) {
        float x[LATENT_DIM];
        for (k=0;k<LATENT_DIM;k++) {
            x[k] = log(1e-15+(float)rand()/RAND_MAX)-log(1e-15+(float)rand()/RAND_MAX);
        }
        for (k=0;k<LATENT_DIM;k++) printf("%f ", x[k]);
        printf("\n");
        dred_encode_latents(&enc, x, scale, dzone, r, p0);
    }
    bytes = (ec_tell(&enc)+7)/8;
    ec_enc_shrink(&enc, bytes);
    ec_enc_done(&enc);

    ec_dec_init(&dec,ec_get_buffer(&enc),bytes);
    for (iter=0;iter<1;iter++) {
        float x[PVQ_DIM];
        dred_decode_state(&dec, x);
    }
    for (iter=0;iter<1;iter++) {
        float x[LATENT_DIM];
        dred_decode_latents(&dec, x, scale, r, p0);
        for (k=0;k<LATENT_DIM;k++) printf("%f ", x[k]);
        printf("\n");
    }
}
#endif
