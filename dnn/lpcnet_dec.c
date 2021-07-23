/* Copyright (c) 2017-2019 Mozilla */
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "freq.h"
#include "pitch.h"
#include "arch.h"
#include "celt_lpc.h"
#include <assert.h>
#include "lpcnet_private.h"
#include "lpcnet.h"


typedef struct {
    int byte_pos;
    int bit_pos;
    int max_bytes;
    const unsigned char *chars;
} unpacker;

void bits_unpacker_init(unpacker *bits, const unsigned char *buf, int size) {
  bits->byte_pos = 0;
  bits->bit_pos = 0;
  bits->max_bytes = size;
  bits->chars = buf;
}

unsigned int bits_unpack(unpacker *bits, int nb_bits) {
  unsigned int d=0;
  while(nb_bits)
  {
    if (bits->byte_pos == bits->max_bytes) {
      fprintf(stderr, "something went horribly wrong\n");
      return 0;
    }
    d<<=1;
    d |= (bits->chars[bits->byte_pos]>>(BITS_PER_CHAR-1 - bits->bit_pos))&1;
    bits->bit_pos++;
    if (bits->bit_pos==BITS_PER_CHAR)
    {
      bits->bit_pos=0;
      bits->byte_pos++;
    }
    nb_bits--;
  }
  return d;
}

void decode_packet(float features[4][NB_TOTAL_FEATURES], float *vq_mem, const unsigned char buf[8])
{
  int c0_id;
  int main_pitch;
  int modulation;
  int corr_id;
  int vq_end[3];
  int vq_mid;
  int interp_id;
  
  int i;
  int sub;
  int voiced = 1;
  float frame_corr;
  ;
  unpacker bits;
  
  bits_unpacker_init(&bits, buf, 8);
  c0_id = bits_unpack(&bits, 7);
  main_pitch = bits_unpack(&bits, 6);
  modulation = bits_unpack(&bits, 3);
  corr_id = bits_unpack(&bits, 2);
  vq_end[0] = bits_unpack(&bits, 10);
  vq_end[1] = bits_unpack(&bits, 10);
  vq_end[2] = bits_unpack(&bits, 10);
  vq_mid = bits_unpack(&bits, 13);
  interp_id = bits_unpack(&bits, 3);
  //fprintf(stdout, "%d %d %d %d %d %d %d %d %d\n", c0_id, main_pitch, modulation, corr_id, vq_end[0], vq_end[1], vq_end[2], vq_mid, interp_id);

  
  for (i=0;i<4;i++) RNN_CLEAR(&features[i][0], NB_TOTAL_FEATURES);

  modulation -= 4;
  if (modulation==-4) {
    voiced = 0;
    modulation = 0;
  }
  if (voiced) {
    frame_corr = 0.3875f + .175f*corr_id;
  } else {
    frame_corr = 0.0375f + .075f*corr_id;
  }
  for (sub=0;sub<4;sub++) {
    float p = pow(2.f, main_pitch/21.)*PITCH_MIN_PERIOD;
    p *= 1 + modulation/16./7.*(2*sub-3);
    p = MIN16(255, MAX16(33, p));
    features[sub][NB_BANDS] = .02*(p-100);
    features[sub][NB_BANDS + 1] = frame_corr-.5;
  }
  
  features[3][0] = (c0_id-64)/4.;
  for (i=0;i<NB_BANDS_1;i++) {
    features[3][i+1] = ceps_codebook1[vq_end[0]*NB_BANDS_1 + i] + ceps_codebook2[vq_end[1]*NB_BANDS_1 + i] + ceps_codebook3[vq_end[2]*NB_BANDS_1 + i];
  }

  float sign = 1;
  if (vq_mid >= 4096) {
    vq_mid -= 4096;
    sign = -1;
  }
  for (i=0;i<NB_BANDS;i++) {
    features[1][i] = sign*ceps_codebook_diff4[vq_mid*NB_BANDS + i];
  }
  if ((vq_mid&MULTI_MASK) < 2) {
    for (i=0;i<NB_BANDS;i++) features[1][i] += .5*(vq_mem[i] + features[3][i]);
  } else if ((vq_mid&MULTI_MASK) == 2) {
    for (i=0;i<NB_BANDS;i++) features[1][i] += vq_mem[i];
  } else {
    for (i=0;i<NB_BANDS;i++) features[1][i] += features[3][i];
  }
  
  perform_double_interp(features, vq_mem, interp_id);

  RNN_COPY(vq_mem, &features[3][0], NB_BANDS);
}
