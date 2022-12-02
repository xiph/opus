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

#include "celt/vq.h"
#include "celt/cwrs.h"

#define LATENT_DIM 80
#define PVQ_DIM 24
#define PVQ_K 82

static void encode_pvq(const int *iy, int N, int K, ec_enc *enc) {
    int fits;
    celt_assert(N==24 || N==12 || N==6);
    fits = (N==24 && K<=9) || (N==12 && K<=16) || (N==6);
    /*printf("encode(%d,%d), fits=%d\n", N, K, fits);*/
    if (fits) encode_pulses(iy, N, K, enc);
    else {
        int N2 = N/2;
        int K0=0;
        int i;
        for (i=0;i<N2;i++) K0 += abs(iy[i]);
        /* FIXME: Don't use uniform probability for K0. */
        ec_enc_uint(enc, K0, K+1);
        /*printf("K0 = %d\n", K0);*/
        encode_pvq(iy, N2, K0, enc);
        encode_pvq(&iy[N2], N2, K-K0, enc);
    }
}

void dred_encode_state(ec_enc *enc, float *x) {
    int k;
    int iy[PVQ_DIM];
    op_pvq_search_c(x, iy, PVQ_K, PVQ_DIM, 0);
    for (k=0;k<PVQ_DIM;k++) printf("%d ", iy[k]);
    printf("\n");
    int tell1 = ec_tell(enc);
    encode_pvq(iy, PVQ_DIM, PVQ_K, enc);
    printf("tell: %d\n", ec_tell(enc)-tell1);
}

void dred_encode_latents(ec_enc *enc, const float *x, const opus_int16 *scale, const opus_int16 *dzone, const opus_int16 *r, const opus_int16 *p0) {
    int i;
    float eps = .1f;
    int tell1 = ec_tell(enc);
    for (i=0;i<LATENT_DIM;i++) {
        float delta;
        float xq;
        int q;
        delta = dzone[i]*(1.f/1024.f);
        xq = x[i]*scale[i]*(1.f/256.f);
        xq = xq - delta*tanh(xq/(delta+eps));
        q = (int)floor(.5f+xq);
        ec_laplace_encode_p0(enc, q, p0[i], r[i]);
    }
    printf("tell: %d\n", ec_tell(enc)-tell1);
}



static void decode_pvq(int *iy, int N, int K, ec_dec *dec) {
    int fits;
    celt_assert(N==24 || N==12 || N==6);
    fits = (N==24 && K<=9) || (N==12 && K<=16) || (N==6);
    /*printf("encode(%d,%d), fits=%d\n", N, K, fits);*/
    if (fits) decode_pulses(iy, N, K, dec);
    else {
        int N2 = N/2;
        int K0;
        /* FIXME: Don't use uniform probability for K0. */
        K0 = ec_dec_uint(dec, K+1);
        /*printf("K0 = %d\n", K0);*/
        decode_pvq(iy, N2, K0, dec);
        decode_pvq(&iy[N2], N2, K-K0, dec);
    }
}

void dred_pvq_dec(ec_enc *dec, float *x) {
    int k;
    int iy[PVQ_DIM];
    int tell1 = ec_tell(dec);
    decode_pvq(iy, PVQ_DIM, PVQ_K, dec);
    /*printf("tell: %d\n", ec_tell(dec)-tell1);*/
    for (k=0;k<PVQ_DIM;k++) printf("%d ", iy[k]);
    printf("\n");
    
}

void dred_rdovae_dec(ec_dec *dec, float *x, const opus_int16 *scale, const opus_int16 *dzone, const opus_int16 *r, const opus_int16 *p0) {
    int i;
    int tell1 = ec_tell(dec);
    for (i=0;i<LATENT_DIM;i++) {
        float xq;
        int q;
        q = ec_laplace_decode_p0(dec, p0[i], r[i]);
        x[i] = q*256.f/scale[i];
    }
    printf("tell: %d\n", ec_tell(dec)-tell1);
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
        dred_pvq_dec(&dec, x);        
    }
    for (iter=0;iter<1;iter++) {
        float x[LATENT_DIM];
        dred_rdovae_dec(&dec, x, scale, dzone, r, p0);
        for (k=0;k<LATENT_DIM;k++) printf("%f ", x[k]);
        printf("\n");
    }
}
#endif
