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


//#define NB_FEATURES (2*NB_BANDS+3+LPC_ORDER)


#define SURVIVORS 5


void vq_quantize_mbest(const float *codebook, int nb_entries, const float *x, int ndim, int mbest, float *dist, int *index)
{
  int i, j;
  for (i=0;i<mbest;i++) dist[i] = 1e15;
  
  for (i=0;i<nb_entries;i++)
  {
    float d=0;
    for (j=0;j<ndim;j++)
      d += (x[j]-codebook[i*ndim+j])*(x[j]-codebook[i*ndim+j]);
    if (d<dist[mbest-1])
    {
      int pos;
      for (j=0;j<mbest-1;j++) {
        if (d < dist[j]) break;
      }
      pos = j;
      for (j=mbest-1;j>=pos+1;j--) {
        dist[j] = dist[j-1];
        index[j] = index[j-1];
      }
      dist[pos] = d;
      index[pos] = i;
    }
  }
}


int vq_quantize(const float *codebook, int nb_entries, const float *x, int ndim, float *dist)
{
  int i, j;
  float min_dist = 1e15;
  int nearest = 0;
  
  for (i=0;i<nb_entries;i++)
  {
    float dist=0;
    for (j=0;j<ndim;j++)
      dist += (x[j]-codebook[i*ndim+j])*(x[j]-codebook[i*ndim+j]);
    if (dist<min_dist)
    {
      min_dist = dist;
      nearest = i;
    }
  }
  if (dist)
    *dist = min_dist;
  return nearest;
}

int quantize_2stage(float *x)
{
    int i;
    int id, id2, id3;
    float ref[NB_BANDS_1];
    RNN_COPY(ref, x, NB_BANDS_1);
    id = vq_quantize(ceps_codebook1, 1024, x, NB_BANDS_1, NULL);
    for (i=0;i<NB_BANDS_1;i++) {
        x[i] -= ceps_codebook1[id*NB_BANDS_1 + i];
    }
    id2 = vq_quantize(ceps_codebook2, 1024, x, NB_BANDS_1, NULL);
    for (i=0;i<NB_BANDS_1;i++) {
        x[i] -= ceps_codebook2[id2*NB_BANDS_1 + i];
    }
    id3 = vq_quantize(ceps_codebook3, 1024, x, NB_BANDS_1, NULL);
    for (i=0;i<NB_BANDS_1;i++) {
        x[i] = ceps_codebook1[id*NB_BANDS_1 + i] + ceps_codebook2[id2*NB_BANDS_1 + i] + ceps_codebook3[id3*NB_BANDS_1 + i];
    }
    if (0) {
        float err = 0;
        for (i=0;i<NB_BANDS_1;i++) {
            err += (x[i]-ref[i])*(x[i]-ref[i]);
        }
        printf("%f\n", sqrt(err/NB_BANDS));
    }
    
    return id;
}


int quantize_3stage_mbest(float *x, int entry[3])
{
    int i, k;
    int id, id2, id3;
    float ref[NB_BANDS_1];
    int curr_index[SURVIVORS];
    int index1[SURVIVORS][3];
    int index2[SURVIVORS][3];
    int index3[SURVIVORS][3];
    float curr_dist[SURVIVORS];
    float glob_dist[SURVIVORS];
    RNN_COPY(ref, x, NB_BANDS_1);
    vq_quantize_mbest(ceps_codebook1, 1024, x, NB_BANDS_1, SURVIVORS, curr_dist, curr_index);
    for (k=0;k<SURVIVORS;k++) {
      index1[k][0] = curr_index[k];
    }
    for (k=0;k<SURVIVORS;k++) {
      int m;
      float diff[NB_BANDS_1];
      for (i=0;i<NB_BANDS_1;i++) {
        diff[i] = x[i] - ceps_codebook1[index1[k][0]*NB_BANDS_1 + i];
      }
      vq_quantize_mbest(ceps_codebook2, 1024, diff, NB_BANDS_1, SURVIVORS, curr_dist, curr_index);
      if (k==0) {
        for (m=0;m<SURVIVORS;m++) {
          index2[m][0] = index1[k][0];
          index2[m][1] = curr_index[m];
          glob_dist[m] = curr_dist[m];
        }
        //printf("%f ", glob_dist[0]);
      } else if (curr_dist[0] < glob_dist[SURVIVORS-1]) {
        m=0;
        int pos;
        for (pos=0;pos<SURVIVORS;pos++) {
          if (curr_dist[m] < glob_dist[pos]) {
            int j;
            for (j=SURVIVORS-1;j>=pos+1;j--) {
              glob_dist[j] = glob_dist[j-1];
              index2[j][0] = index2[j-1][0];
              index2[j][1] = index2[j-1][1];
            }
            glob_dist[pos] = curr_dist[m];
            index2[pos][0] = index1[k][0];
            index2[pos][1] = curr_index[m];
            m++;
          }
        }
      }
    }
    for (k=0;k<SURVIVORS;k++) {
      int m;
      float diff[NB_BANDS_1];
      for (i=0;i<NB_BANDS_1;i++) {
        diff[i] = x[i] - ceps_codebook1[index2[k][0]*NB_BANDS_1 + i] - ceps_codebook2[index2[k][1]*NB_BANDS_1 + i];
      }
      vq_quantize_mbest(ceps_codebook3, 1024, diff, NB_BANDS_1, SURVIVORS, curr_dist, curr_index);
      if (k==0) {
        for (m=0;m<SURVIVORS;m++) {
          index3[m][0] = index2[k][0];
          index3[m][1] = index2[k][1];
          index3[m][2] = curr_index[m];
          glob_dist[m] = curr_dist[m];
        }
        //printf("%f ", glob_dist[0]);
      } else if (curr_dist[0] < glob_dist[SURVIVORS-1]) {
        m=0;
        int pos;
        for (pos=0;pos<SURVIVORS;pos++) {
          if (curr_dist[m] < glob_dist[pos]) {
            int j;
            for (j=SURVIVORS-1;j>=pos+1;j--) {
              glob_dist[j] = glob_dist[j-1];
              index3[j][0] = index3[j-1][0];
              index3[j][1] = index3[j-1][1];
              index3[j][2] = index3[j-1][2];
            }
            glob_dist[pos] = curr_dist[m];
            index3[pos][0] = index2[k][0];
            index3[pos][1] = index2[k][1];
            index3[pos][2] = curr_index[m];
            m++;
          }
        }
      }
    }
    entry[0] = id = index3[0][0];
    entry[1] = id2 = index3[0][1];
    entry[2] = id3 = index3[0][2];
    //printf("%f ", glob_dist[0]);
    for (i=0;i<NB_BANDS_1;i++) {
        x[i] -= ceps_codebook1[id*NB_BANDS_1 + i];
    }
    for (i=0;i<NB_BANDS_1;i++) {
        x[i] -= ceps_codebook2[id2*NB_BANDS_1 + i];
    }
    //id3 = vq_quantize(ceps_codebook3, 1024, x, NB_BANDS_1, NULL);
    for (i=0;i<NB_BANDS_1;i++) {
        x[i] = ceps_codebook1[id*NB_BANDS_1 + i] + ceps_codebook2[id2*NB_BANDS_1 + i] + ceps_codebook3[id3*NB_BANDS_1 + i];
    }
    if (0) {
        float err = 0;
        for (i=0;i<NB_BANDS_1;i++) {
            err += (x[i]-ref[i])*(x[i]-ref[i]);
        }
        printf("%f\n", sqrt(err/NB_BANDS));
    }
    
    return id;
}

static int find_nearest_multi(const float *codebook, int nb_entries, const float *x, int ndim, float *dist, int sign)
{
  int i, j;
  float min_dist = 1e15;
  int nearest = 0;

  for (i=0;i<nb_entries;i++)
  {
    int offset;
    float dist=0;
    offset = (i&MULTI_MASK)*ndim;
    for (j=0;j<ndim;j++)
      dist += (x[offset+j]-codebook[i*ndim+j])*(x[offset+j]-codebook[i*ndim+j]);
    if (dist<min_dist)
    {
      min_dist = dist;
      nearest = i;
    }
  }
  if (sign) {
    for (i=0;i<nb_entries;i++)
    {
      int offset;
      float dist=0;
      offset = (i&MULTI_MASK)*ndim;
      for (j=0;j<ndim;j++)
        dist += (x[offset+j]+codebook[i*ndim+j])*(x[offset+j]+codebook[i*ndim+j]);
      if (dist<min_dist)
      {
        min_dist = dist;
        nearest = i+nb_entries;
      }
    }
  }
  if (dist)
    *dist = min_dist;
  return nearest;
}


int quantize_diff(float *x, float *left, float *right, float *codebook, int bits, int sign, int *entry)
{
    int i;
    int nb_entries;
    int id;
    float ref[NB_BANDS];
    float pred[4*NB_BANDS];
    float target[4*NB_BANDS];
    float s = 1;
    nb_entries = 1<<bits;
    RNN_COPY(ref, x, NB_BANDS);
    for (i=0;i<NB_BANDS;i++) pred[i] = pred[NB_BANDS+i] = .5*(left[i] + right[i]);
    for (i=0;i<NB_BANDS;i++) pred[2*NB_BANDS+i] = left[i];
    for (i=0;i<NB_BANDS;i++) pred[3*NB_BANDS+i] = right[i];
    for (i=0;i<4*NB_BANDS;i++) target[i] = x[i%NB_BANDS] - pred[i];

    id = find_nearest_multi(codebook, nb_entries, target, NB_BANDS, NULL, sign);
    *entry = id;
    if (id >= 1<<bits) {
      s = -1;
      id -= (1<<bits);
    }
    for (i=0;i<NB_BANDS;i++) {
      x[i] = pred[(id&MULTI_MASK)*NB_BANDS + i] + s*codebook[id*NB_BANDS + i];
    }
    //printf("%d %f ", id&MULTI_MASK, s);
    if (0) {
        float err = 0;
        for (i=0;i<NB_BANDS;i++) {
            err += (x[i]-ref[i])*(x[i]-ref[i]);
        }
        printf("%f\n", sqrt(err/NB_BANDS));
    }
    
    return id;
}

int interp_search(const float *x, const float *left, const float *right, float *dist_out)
{
    int i, k;
    float min_dist = 1e15;
    int best_pred = 0;
    float pred[4*NB_BANDS];
    for (i=0;i<NB_BANDS;i++) pred[i] = pred[NB_BANDS+i] = .5*(left[i] + right[i]);
    for (i=0;i<NB_BANDS;i++) pred[2*NB_BANDS+i] = left[i];
    for (i=0;i<NB_BANDS;i++) pred[3*NB_BANDS+i] = right[i];

    for (k=1;k<4;k++) {
      float dist = 0;
      for (i=0;i<NB_BANDS;i++) dist += (x[i] - pred[k*NB_BANDS+i])*(x[i] - pred[k*NB_BANDS+i]);
      dist_out[k-1] = dist;
      if (dist < min_dist) {
        min_dist = dist;
        best_pred = k;
      }
    }
    return best_pred - 1;
}


void interp_diff(float *x, float *left, float *right, float *codebook, int bits, int sign)
{
    int i, k;
    float min_dist = 1e15;
    int best_pred = 0;
    float ref[NB_BANDS];
    float pred[4*NB_BANDS];
    (void)sign;
    (void)codebook;
    (void)bits;
    RNN_COPY(ref, x, NB_BANDS);
    for (i=0;i<NB_BANDS;i++) pred[i] = pred[NB_BANDS+i] = .5*(left[i] + right[i]);
    for (i=0;i<NB_BANDS;i++) pred[2*NB_BANDS+i] = left[i];
    for (i=0;i<NB_BANDS;i++) pred[3*NB_BANDS+i] = right[i];

    for (k=1;k<4;k++) {
      float dist = 0;
      for (i=0;i<NB_BANDS;i++) dist += (x[i] - pred[k*NB_BANDS+i])*(x[i] - pred[k*NB_BANDS+i]);
      if (dist < min_dist) {
        min_dist = dist;
        best_pred = k;
      }
    }
    //printf("%d ", best_pred);
    for (i=0;i<NB_BANDS;i++) {
      x[i] = pred[best_pred*NB_BANDS + i];
    }
    if (0) {
        float err = 0;
        for (i=0;i<NB_BANDS;i++) {
            err += (x[i]-ref[i])*(x[i]-ref[i]);
        }
        printf("%f\n", sqrt(err/NB_BANDS));
    }
}

int double_interp_search(float features[4][NB_TOTAL_FEATURES], const float *mem) {
    int i, j;
    int best_id=0;
    float min_dist = 1e15;
    float dist[2][3];
    interp_search(features[0], mem, features[1], dist[0]);
    interp_search(features[2], features[1], features[3], dist[1]);
    for (i=0;i<3;i++) {
        for (j=0;j<3;j++) {
            float d;
            int id;
            id = 3*i + j;
            d = dist[0][i] + dist[1][j];
            if (d < min_dist && id != FORBIDDEN_INTERP) {
                min_dist = d;
                best_id = id;
            }
        }
    }
    //printf("%d %d %f    %d %f\n", id0, id1, dist[0][id0] + dist[1][id1], best_id, min_dist);
    return best_id - (best_id >= FORBIDDEN_INTERP);
}


void perform_interp_relaxation(float features[4][NB_TOTAL_FEATURES], const float *mem) {
    int id0, id1;
    int best_id;
    int i;
    float count, count_1;
    best_id = double_interp_search(features, mem);
    best_id += (best_id >= FORBIDDEN_INTERP);
    id0 = best_id / 3;
    id1 = best_id % 3;
    count = 1;
    if (id0 != 1) {
        float t = (id0==0) ? .5 : 1.;
        for (i=0;i<NB_BANDS;i++) features[1][i] += t*features[0][i];
        count += t;
    }
    if (id1 != 2) {
        float t = (id1==0) ? .5 : 1.;
        for (i=0;i<NB_BANDS;i++) features[1][i] += t*features[2][i];
        count += t;
    }
    count_1 = 1.f/count;
    for (i=0;i<NB_BANDS;i++) features[1][i] *= count_1;
}

typedef struct {
    int byte_pos;
    int bit_pos;
    int max_bytes;
    unsigned char *chars;
} packer;


void bits_packer_init(packer *bits, unsigned char *buf, int size) {
  bits->byte_pos = 0;
  bits->bit_pos = 0;
  bits->max_bytes = size;
  bits->chars = buf;
  RNN_CLEAR(buf, size);
}

void bits_pack(packer *bits, unsigned int data, int nb_bits) {
  while(nb_bits)
  {
    int bit;
    if (bits->byte_pos == bits->max_bytes) {
      fprintf(stderr, "something went horribly wrong\n");
      return;
    }
    bit = (data>>(nb_bits-1))&1;
    bits->chars[bits->byte_pos] |= bit<<(BITS_PER_CHAR-1-bits->bit_pos);
    bits->bit_pos++;

    if (bits->bit_pos==BITS_PER_CHAR)
    {
      bits->bit_pos=0;
      bits->byte_pos++;
      if (bits->byte_pos < bits->max_bytes) bits->chars[bits->byte_pos] = 0;
    }
    nb_bits--;
  }
}


LPCNET_EXPORT int lpcnet_encoder_get_size() {
  return sizeof(LPCNetEncState);
}

LPCNET_EXPORT int lpcnet_encoder_init(LPCNetEncState *st) {
  memset(st, 0, sizeof(*st));
  st->exc_mem = lin2ulaw(0.f);
  return 0;
}

LPCNET_EXPORT LPCNetEncState *lpcnet_encoder_create() {
  LPCNetEncState *st;
  st = malloc(lpcnet_encoder_get_size());
  lpcnet_encoder_init(st);
  return st;
}

LPCNET_EXPORT void lpcnet_encoder_destroy(LPCNetEncState *st) {
  free(st);
}

static void frame_analysis(LPCNetEncState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, OVERLAP_SIZE);
  RNN_COPY(&x[OVERLAP_SIZE], in, FRAME_SIZE);
  RNN_COPY(st->analysis_mem, &in[FRAME_SIZE-OVERLAP_SIZE], OVERLAP_SIZE);
  apply_window(x);
  forward_transform(X, x);
  compute_band_energy(Ex, X);
}

void compute_frame_features(LPCNetEncState *st, const float *in) {
  float aligned_in[FRAME_SIZE];
  int i;
  float E = 0;
  float Ly[NB_BANDS];
  float follow, logMax;
  float g;
  kiss_fft_cpx X[FREQ_SIZE];
  float Ex[NB_BANDS];
  float xcorr[PITCH_MAX_PERIOD];
  float ener0;
  int sub;
  float ener;
  RNN_COPY(aligned_in, &st->analysis_mem[OVERLAP_SIZE-TRAINING_OFFSET], TRAINING_OFFSET);
  frame_analysis(st, X, Ex, in);
  logMax = -2;
  follow = -2;
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    Ly[i] = MAX16(logMax-8, MAX16(follow-2.5, Ly[i]));
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-2.5, Ly[i]);
    E += Ex[i];
  }
  dct(st->features[st->pcount], Ly);
  st->features[st->pcount][0] -= 4;
  g = lpc_from_cepstrum(st->lpc, st->features[st->pcount]);
  st->features[st->pcount][2*NB_BANDS+2] = log10(g);
  for (i=0;i<LPC_ORDER;i++) st->features[st->pcount][2*NB_BANDS+3+i] = st->lpc[i];
  RNN_MOVE(st->exc_buf, &st->exc_buf[FRAME_SIZE], PITCH_MAX_PERIOD);
  RNN_COPY(&aligned_in[TRAINING_OFFSET], in, FRAME_SIZE-TRAINING_OFFSET);
  for (i=0;i<FRAME_SIZE;i++) {
    int j;
    float sum = aligned_in[i];
    for (j=0;j<LPC_ORDER;j++)
      sum += st->lpc[j]*st->pitch_mem[j];
    RNN_MOVE(st->pitch_mem+1, st->pitch_mem, LPC_ORDER-1);
    st->pitch_mem[0] = aligned_in[i];
    st->exc_buf[PITCH_MAX_PERIOD+i] = sum + .7*st->pitch_filt;
    st->pitch_filt = sum;
    //printf("%f\n", st->exc_buf[PITCH_MAX_PERIOD+i]);
  }
  /* Cross-correlation on half-frames. */
  for (sub=0;sub<2;sub++) {
    int off = sub*FRAME_SIZE/2;
    celt_pitch_xcorr(&st->exc_buf[PITCH_MAX_PERIOD+off], st->exc_buf+off, xcorr, FRAME_SIZE/2, PITCH_MAX_PERIOD);
    ener0 = celt_inner_prod(&st->exc_buf[PITCH_MAX_PERIOD+off], &st->exc_buf[PITCH_MAX_PERIOD+off], FRAME_SIZE/2);
    st->frame_weight[2+2*st->pcount+sub] = ener0;
    //printf("%f\n", st->frame_weight[2+2*st->pcount+sub]);
    for (i=0;i<PITCH_MAX_PERIOD;i++) {
      ener = (1 + ener0 + celt_inner_prod(&st->exc_buf[i+off], &st->exc_buf[i+off], FRAME_SIZE/2));
      st->xc[2+2*st->pcount+sub][i] = 2*xcorr[i] / ener;
    }
#if 0
    for (i=0;i<PITCH_MAX_PERIOD;i++)
      printf("%f ", st->xc[2*st->pcount+sub][i]);
    printf("\n");
#endif
  }
}

void process_superframe(LPCNetEncState *st, unsigned char *buf, FILE *ffeat, int encode, int quantize) {
  int i;
  int sub;
  int best_i;
  int best[10];
  int pitch_prev[8][PITCH_MAX_PERIOD];
  float best_a=0;
  float best_b=0;
  float w;
  float sx=0, sxx=0, sxy=0, sy=0, sw=0;
  float frame_corr;
  int voiced;
  float frame_weight_sum = 1e-15;
  float center_pitch;
  int main_pitch;
  int modulation;
  int c0_id=0;
  int vq_end[3]={0};
  int vq_mid=0;
  int corr_id = 0;
  int interp_id=0;
  for(sub=0;sub<8;sub++) frame_weight_sum += st->frame_weight[2+sub];
  for(sub=0;sub<8;sub++) st->frame_weight[2+sub] *= (8.f/frame_weight_sum);
  for(sub=0;sub<8;sub++) {
    float max_path_all = -1e15;
    best_i = 0;
    for (i=0;i<PITCH_MAX_PERIOD-2*PITCH_MIN_PERIOD;i++) {
      float xc_half = MAX16(MAX16(st->xc[2+sub][(PITCH_MAX_PERIOD+i)/2], st->xc[2+sub][(PITCH_MAX_PERIOD+i+2)/2]), st->xc[2+sub][(PITCH_MAX_PERIOD+i-1)/2]);
      if (st->xc[2+sub][i] < xc_half*1.1) st->xc[2+sub][i] *= .8;
    }
    for (i=0;i<PITCH_MAX_PERIOD-PITCH_MIN_PERIOD;i++) {
      int j;
      float max_prev;
      max_prev = st->pitch_max_path_all - 6.f;
      pitch_prev[sub][i] = st->best_i;
      for (j=IMIN(0, 4-i);j<=4 && i+j<PITCH_MAX_PERIOD-PITCH_MIN_PERIOD;j++) {
        if (st->pitch_max_path[0][i+j] - .02f*abs(j)*abs(j) > max_prev) {
          max_prev = st->pitch_max_path[0][i+j] - .02f*abs(j)*abs(j);
          pitch_prev[sub][i] = i+j;
        }
      }
      st->pitch_max_path[1][i] = max_prev + st->frame_weight[2+sub]*st->xc[2+sub][i];
      if (st->pitch_max_path[1][i] > max_path_all) {
        max_path_all = st->pitch_max_path[1][i];
        best_i = i;
      }
    }
    /* Renormalize. */
    for (i=0;i<PITCH_MAX_PERIOD-PITCH_MIN_PERIOD;i++) st->pitch_max_path[1][i] -= max_path_all;
    //for (i=0;i<PITCH_MAX_PERIOD-PITCH_MIN_PERIOD;i++) printf("%f ", st->pitch_max_path[1][i]);
    //printf("\n");
    RNN_COPY(&st->pitch_max_path[0][0], &st->pitch_max_path[1][0], PITCH_MAX_PERIOD);
    st->pitch_max_path_all = max_path_all;
    st->best_i = best_i;
  }
  best_i = st->best_i;
  frame_corr = 0;
  /* Backward pass. */
  for (sub=7;sub>=0;sub--) {
    best[2+sub] = PITCH_MAX_PERIOD-best_i;
    frame_corr += st->frame_weight[2+sub]*st->xc[2+sub][best_i];
    best_i = pitch_prev[sub][best_i];
  }
  frame_corr /= 8;
  if (quantize && frame_corr < 0) frame_corr = 0;
  for (sub=0;sub<8;sub++) {
    //printf("%d %f\n", best[2+sub], frame_corr);
  }
  //printf("\n");
  for (sub=2;sub<10;sub++) {
    w = st->frame_weight[sub];
    sw += w;
    sx += w*sub;
    sxx += w*sub*sub;
    sxy += w*sub*best[sub];
    sy += w*best[sub];
  }
  voiced = frame_corr >= .3;
  /* Linear regression to figure out the pitch contour. */
  best_a = (sw*sxy - sx*sy)/(sw*sxx - sx*sx);
  if (voiced) {
    float max_a;
    float mean_pitch = sy/sw;
    /* Allow a relative variation of up to 1/4 over 8 sub-frames. */
    max_a = mean_pitch/32;
    best_a = MIN16(max_a, MAX16(-max_a, best_a));
    corr_id = (int)floor((frame_corr-.3f)/.175f);
    if (quantize) frame_corr = 0.3875f + .175f*corr_id;
  } else {
    best_a = 0;
    corr_id = (int)floor(frame_corr/.075f);
    if (quantize) frame_corr = 0.0375f + .075f*corr_id;
  }
  //best_b = (sxx*sy - sx*sxy)/(sw*sxx - sx*sx);
  best_b = (sy - best_a*sx)/sw;
  /* Quantizing the pitch as "main" pitch + slope. */
  center_pitch = best_b+5.5*best_a;
  main_pitch = (int)floor(.5 + 21.*log2(center_pitch/PITCH_MIN_PERIOD));
  main_pitch = IMAX(0, IMIN(63, main_pitch));
  modulation = (int)floor(.5 + 16*7*best_a/center_pitch);
  modulation = IMAX(-3, IMIN(3, modulation));
  //printf("%d %d\n", main_pitch, modulation);
  //printf("%f %f\n", best_a/center_pitch, best_corr);
  //for (sub=2;sub<10;sub++) printf("%f %d %f\n", best_b + sub*best_a, best[sub], best_corr);
  for (sub=0;sub<4;sub++) {
    if (quantize) {
      float p = pow(2.f, main_pitch/21.)*PITCH_MIN_PERIOD;
      p *= 1 + modulation/16./7.*(2*sub-3);
      p = MIN16(255, MAX16(32, p));
      st->features[sub][2*NB_BANDS] = .02*(p-100);
      st->features[sub][2*NB_BANDS + 1] = frame_corr-.5;
    } else {
      st->features[sub][2*NB_BANDS] = .01*(IMAX(64, IMIN(510, best[2+2*sub]+best[2+2*sub+1]))-200);
      st->features[sub][2*NB_BANDS + 1] = frame_corr-.5;
    }
    //printf("%f %d %f\n", st->features[sub][2*NB_BANDS], best[2+2*sub], frame_corr);
  }
  //printf("%d %f %f %f\n", best_period, best_a, best_b, best_corr);
  RNN_COPY(&st->xc[0][0], &st->xc[8][0], PITCH_MAX_PERIOD);
  RNN_COPY(&st->xc[1][0], &st->xc[9][0], PITCH_MAX_PERIOD);
  if (quantize) {
    //printf("%f\n", st->features[3][0]);
    c0_id = (int)floor(.5 + st->features[3][0]*4);
    c0_id = IMAX(-64, IMIN(63, c0_id));
    st->features[3][0] = c0_id/4.;
    quantize_3stage_mbest(&st->features[3][1], vq_end);
    /*perform_interp_relaxation(st->features, st->vq_mem);*/
    quantize_diff(&st->features[1][0], st->vq_mem, &st->features[3][0], ceps_codebook_diff4, 12, 1, &vq_mid);
    interp_id = double_interp_search(st->features, st->vq_mem);
    perform_double_interp(st->features, st->vq_mem, interp_id);
  }
  for (sub=0;sub<4;sub++) {
    float g = lpc_from_cepstrum(st->lpc, st->features[sub]);
    st->features[sub][2*NB_BANDS+2] = log10(g);
    for (i=0;i<LPC_ORDER;i++) st->features[sub][2*NB_BANDS+3+i] = st->lpc[i];
  }
  //printf("\n");
  RNN_COPY(st->vq_mem, &st->features[3][0], NB_BANDS);
  if (encode) {
    packer bits;
    //fprintf(stdout, "%d %d %d %d %d %d %d %d %d\n", c0_id+64, main_pitch, voiced ? modulation+4 : 0, corr_id, vq_end[0], vq_end[1], vq_end[2], vq_mid, interp_id);
    bits_packer_init(&bits, buf, 8);
    bits_pack(&bits, c0_id+64, 7);
    bits_pack(&bits, main_pitch, 6);
    bits_pack(&bits, voiced ? modulation+4 : 0, 3);
    bits_pack(&bits, corr_id, 2);
    bits_pack(&bits, vq_end[0], 10);
    bits_pack(&bits, vq_end[1], 10);
    bits_pack(&bits, vq_end[2], 10);
    bits_pack(&bits, vq_mid, 13);
    bits_pack(&bits, interp_id, 3);
    if (ffeat) fwrite(buf, 1, 8, ffeat);
  } else if (ffeat) {
    for (i=0;i<4;i++) {
      fwrite(st->features[i], sizeof(float), NB_TOTAL_FEATURES, ffeat);
    }
  }
}

void preemphasis(float *y, float *mem, const float *x, float coef, int N) {
  int i;
  for (i=0;i<N;i++) {
    float yi;
    yi = x[i] + *mem;
    *mem = -coef*x[i];
    y[i] = yi;
  }
}

LPCNET_EXPORT int lpcnet_encode(LPCNetEncState *st, const short *pcm, unsigned char *buf) {
  int i, k;
  for (k=0;k<4;k++) {
    float x[FRAME_SIZE];
    for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[k*FRAME_SIZE + i];
    preemphasis(x, &st->mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    st->pcount = k;
    compute_frame_features(st, x);
  }
  process_superframe(st, buf, NULL, 1, 1);
  return 0;
}

LPCNET_EXPORT int lpcnet_compute_features(LPCNetEncState *st, const short *pcm, float features[4][NB_TOTAL_FEATURES]) {
  int i, k;
  for (k=0;k<4;k++) {
    float x[FRAME_SIZE];
    for (i=0;i<FRAME_SIZE;i++) x[i] = pcm[k*FRAME_SIZE + i];
    preemphasis(x, &st->mem_preemph, x, PREEMPHASIS, FRAME_SIZE);
    st->pcount = k;
    compute_frame_features(st, x);
  }
  process_superframe(st, NULL, NULL, 0, 0);
  for (k=0;k<4;k++) {
    RNN_COPY(&features[k][0], &st->features[k][0], NB_TOTAL_FEATURES);
  }
  return 0;
}
