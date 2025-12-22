/* Copyright (c) 2017-2018 Mozilla */
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
#include <unistd.h>
#include <errno.h>
#include "common.h"
#include <math.h>
#include "freq.h"
#include "pitch.h"
#include "arch.h"
#include <assert.h>
#include "lpcnet.h"
#include "lpcnet_private.h"
#include "os_support.h"
#include "cpu_support.h"

#include "mini_kfft.c"

#ifndef M_PI
#define M_PI 3.141592653589793f
#endif

#define SEQUENCE_LENGTH 2000
#define SEQUENCE_SAMPLES (SEQUENCE_LENGTH*FRAME_SIZE)

#define ENABLE_RIR

#ifdef ENABLE_RIR

#define RIR_FFT_SIZE 32000
#define RIR_MAX_DURATION (RIR_FFT_SIZE/2)
#define FILENAME_MAX_SIZE 1000

struct rir_list {
  int nb_rirs;
  int block_size;
  mini_kiss_fft_state *fft;
  mini_kiss_fft_state *ifft;
  mini_kiss_fft_cpx **rir;
  mini_kiss_fft_cpx **early;
};

mini_kiss_fft_cpx *load_rir(const char *rir_file, mini_kiss_fft_state *fft, int early) {
  mini_kiss_fft_cpx *x, *X;
  float rir[RIR_MAX_DURATION];
  int len;
  int i;
  FILE *f;
  f = fopen(rir_file, "rb");
  if (f==NULL) {
    fprintf(stderr, "cannot open RIR file %s: %s\n", rir_file, strerror(errno));
    exit(1);
  }
  x = (mini_kiss_fft_cpx*)calloc(fft->nfft, sizeof(*x));
  X = (mini_kiss_fft_cpx*)calloc(fft->nfft, sizeof(*X));
  len = fread(rir, sizeof(*rir), RIR_MAX_DURATION, f);
  if (early) {
    for (i=0;i<240;i++) {
      rir[480+i] *= (1 - i/240.f);
    }
    OPUS_CLEAR(&rir[240+480], RIR_MAX_DURATION-240-480);
  }
  for (i=0;i<len;i++) x[i].r = rir[i];
  mini_kiss_fft(fft, x, X);
  free(x);
  fclose(f);
  return X;
}

void load_rir_list(const char *list_file, struct rir_list *rirs) {
  int allocated;
  char rir_filename[FILENAME_MAX_SIZE];
  FILE *f;
  f = fopen(list_file, "rb");
  if (f==NULL) {
    fprintf(stderr, "cannot open %s: %s\n", list_file, strerror(errno));
    exit(1);
  }
  rirs->nb_rirs = 0;
  allocated = 2;
  rirs->fft = mini_kiss_fft_alloc(RIR_FFT_SIZE, 0, NULL, NULL);
  rirs->ifft = mini_kiss_fft_alloc(RIR_FFT_SIZE, 1, NULL, NULL);
  rirs->rir = (mini_kiss_fft_cpx**)malloc(allocated*sizeof(rirs->rir[0]));
  rirs->early = (mini_kiss_fft_cpx**)malloc(allocated*sizeof(rirs->early[0]));
  while (fgets(rir_filename, FILENAME_MAX_SIZE, f) != NULL) {
    /* Chop trailing newline. */
    rir_filename[strcspn(rir_filename, "\n")] = 0;
    if (rirs->nb_rirs+1 > allocated) {
      allocated *= 2;
      rirs->rir = (mini_kiss_fft_cpx**)realloc(rirs->rir, allocated*sizeof(rirs->rir[0]));
      rirs->early = (mini_kiss_fft_cpx**)realloc(rirs->early, allocated*sizeof(rirs->early[0]));
    }
    rirs->rir[rirs->nb_rirs] = load_rir(rir_filename, rirs->fft, 0);
    rirs->early[rirs->nb_rirs] = load_rir(rir_filename, rirs->fft, 1);
    rirs->nb_rirs++;
  }
  fclose(f);
}

void rir_filter_sequence(const struct rir_list *rirs, float *audio, int rir_id, int early) {
  int i;
  mini_kiss_fft_cpx x[RIR_FFT_SIZE] = {{0,0}};
  mini_kiss_fft_cpx y[RIR_FFT_SIZE] = {{0,0}};
  mini_kiss_fft_cpx X[RIR_FFT_SIZE] = {{0,0}};
  const mini_kiss_fft_cpx *Y;
  if (early) Y = rirs->early[rir_id];
  else Y = rirs->rir[rir_id];
  i=0;
  while (i<SEQUENCE_SAMPLES) {
    int j;
    OPUS_COPY(&x[0], &x[RIR_FFT_SIZE/2], RIR_FFT_SIZE/2);
    for (j=0;j<IMIN(SEQUENCE_SAMPLES-i, RIR_FFT_SIZE/2);j++) x[RIR_FFT_SIZE/2+j].r = audio[i+j];
    for (;j<RIR_FFT_SIZE/2;j++) x[RIR_FFT_SIZE/2+j].r = 0;
    mini_kiss_fft(rirs->fft, x, X);
    for (j=0;j<RIR_FFT_SIZE;j++) {
      mini_kiss_fft_cpx tmp;
      C_MUL(tmp, X[j], Y[j]);
      X[j].r = tmp.r*(1.f/(2.f*RIR_FFT_SIZE));
      X[j].i = tmp.i*(1.f/(2.f*RIR_FFT_SIZE));
    }
    mini_kiss_fft(rirs->ifft, X, y);
    for (j=0;j<IMIN(SEQUENCE_SAMPLES-i, RIR_FFT_SIZE/2);j++) audio[i+j] = y[RIR_FFT_SIZE/2+j].r;
    i += RIR_FFT_SIZE/2;
  }
}
#endif


static unsigned rand_lcg(unsigned *seed) {
  *seed = 1664525**seed + 1013904223;
  return *seed;
}

static float randf(float f) {
  return f*rand()/(double)RAND_MAX;
}

static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

static float uni_rand(void) {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_filt(float *a) {
  if (rand()%3!=0) {
    a[0] = a[1] = 0;
  }
  else if (uni_rand()>0) {
    float r, theta;
    r = rand()/(double)RAND_MAX;
    r = .7*r*r;
    theta = rand()/(double)RAND_MAX;
    theta = M_PI*theta*theta;
    a[0] = -2*r*cos(theta);
    a[1] = r*r;
  } else {
    float r0,r1;
    r0 = 1.4*uni_rand();
    r1 = 1.4*uni_rand();
    a[0] = -r0-r1;
    a[1] = r0*r1;
  }
}

static void rand_resp(float *a, float *b) {
  rand_filt(a);
  rand_filt(b);
}

static opus_int16 float2short(float x)
{
  int i;
  i = (int)floor(.5+x);
  return IMAX(-32767, IMIN(32767, i));
}

static float weighted_rms(float *x) {
  int i;
  float tmp[SEQUENCE_SAMPLES];
  float weighting_b[2] = {-2.f, 1.f};
  float weighting_a[2] = {-1.89f, .895f};
  float mem[2] = {0};
  float mse = 1e-15f;
  biquad(tmp, mem, x, weighting_b, weighting_a, SEQUENCE_SAMPLES);
  for (i=0;i<SEQUENCE_SAMPLES;i++) mse += tmp[i]*tmp[i];
  return 0.9506*sqrt(mse/SEQUENCE_SAMPLES);
}


short speech16[SEQUENCE_LENGTH*FRAME_SIZE];
short noise16[SEQUENCE_LENGTH*FRAME_SIZE] = {0};
float x[SEQUENCE_LENGTH*FRAME_SIZE];
float n[SEQUENCE_LENGTH*FRAME_SIZE];
float xn[SEQUENCE_LENGTH*FRAME_SIZE];


int main(int argc, char **argv) {
  int i, j;
  const char *argv0;
  const char *noise_filename=NULL;
  const char *rir_filename=NULL;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_noise[2] = {0};
  float b_noise[2] = {0};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_preemph=0;
  FILE *f1, *f2=NULL;
  FILE *ffeat;
  FILE *fpcm=NULL;
  opus_int16 pcm[FRAME_SIZE]={0};
  float speech_gain=1;
  LPCNetEncState *st;
  int training = -1;
  int burg = 0;
  int pitch = 0;
  float noise_gain = 0;
  int arch;
  long speech_length, noise_length=0;
  int maxCount;
  unsigned seed;
#ifdef ENABLE_RIR
  struct rir_list rirs;
#endif
  srand(getpid());
  arch = opus_select_arch();
  st = lpcnet_encoder_create();
  argv0=argv[0];
  if (argc == 5 && strcmp(argv[1], "-btrain")==0) {
      burg = 1;
      training = 1;
  }
  else if (argc == 4 && strcmp(argv[1], "-btest")==0) {
      burg = 1;
      training = 0;
  }
  else if (argc == 5 && strcmp(argv[1], "-ptrain")==0) {
      pitch = 1;
      training = 1;
      noise_filename = argv[2];
      argv++;
  }
  else if (argc == 4 && strcmp(argv[1], "-ptest")==0) {
      pitch = 1;
      training = 0;
  }
  else if (argc == 7 && strcmp(argv[1], "-train")==0) {
     training = 1;
     noise_filename = argv[2];
     rir_filename = argv[3];
     argv+=2;
  } else if (argc == 4 && strcmp(argv[1], "-test")==0) training = 0;
  if (training == -1) {
    fprintf(stderr, "usage: %s -train <noise> <rir_list> <speech> <features out> <pcm out>\n", argv0);
    fprintf(stderr, "       %s -ptrain <noise> <speech> <features out>\n", argv0);
    fprintf(stderr, "       %s -test <speech> <features out>\n", argv0);
    return 1;
  }
  f1 = fopen(argv[2], "r");
  if (f1 == NULL) {
    fprintf(stderr,"Error opening input .s16 16kHz speech input file: %s\n", argv[2]);
    exit(1);
  }
  if (noise_filename != NULL) {
     f2 = fopen(noise_filename, "r");
     if (f2 == NULL) {
        fprintf(stderr,"Error opening input .s16 16kHz speech input file: %s\n", noise_filename);
        exit(1);
     }
     fseek(f2, 0, SEEK_END);
     noise_length = ftell(f2);
     fseek(f2, 0, SEEK_SET);
  }
  ffeat = fopen(argv[3], "wb");
  if (ffeat == NULL) {
    fprintf(stderr,"Error opening output feature file: %s\n", argv[3]);
    exit(1);
  }
  if (training && !pitch) {
    fpcm = fopen(argv[4], "wb");
    if (fpcm == NULL) {
      fprintf(stderr,"Error opening output PCM file: %s\n", argv[4]);
      exit(1);
    }
  }
#ifdef ENABLE_RIR
  if (rir_filename != NULL) {
     load_rir_list(rir_filename, &rirs);
  }
#endif

  seed = getpid();
  srand(seed);

  fseek(f1, 0, SEEK_END);
  speech_length = ftell(f1);
  fseek(f1, 0, SEEK_SET);
#ifndef ENABLE_RIR
  fprintf(stderr, "WARNING: dump_data was built without RIR support\n");
#endif

  maxCount = 20000;
  for (count=0;count<maxCount;count++) {
    int rir_id;
    int sequence_length;
    long speech_pos, noise_pos;
    int start_pos=0;
    float E[SEQUENCE_LENGTH] = {0};
    float mem[2]={0};
    int frame;
    float speech_rms, noise_rms;
    int ret;
    if ((count%1000)==0) fprintf(stderr, "%d\r", count);
    speech_pos = (rand_lcg(&seed)*2.3283e-10)*speech_length;
    if (speech_pos > speech_length-(long)sizeof(speech16)) speech_pos = speech_length-sizeof(speech16);
    speech_pos -= speech_pos&1;
    fseek(f1, speech_pos, SEEK_SET);
    ret = fread(speech16, sizeof(speech16), 1, f1);
    if (ret != 1) {
       fprintf(stderr, "reading speech failed\n");
       return 1;
    }
    if (f2!=NULL) {
       noise_pos = (rand_lcg(&seed)*2.3283e-10)*noise_length;
       if (noise_pos > noise_length-(long)sizeof(noise16)) noise_pos = noise_length-sizeof(noise16);
       noise_pos -= noise_pos&1;
       fseek(f2, noise_pos, SEEK_SET);
       ret = fread(noise16, sizeof(noise16), 1, f2);
       if (ret != 1) {
          fprintf(stderr, "reading noise failed\n");
          return 1;
       }
    }
    if (rand()%4) start_pos = 0;
    else start_pos = -(int)(1000*log(rand()/(float)RAND_MAX));
    start_pos = IMIN(start_pos, SEQUENCE_LENGTH*FRAME_SIZE);

    speech_gain = pow(10., (-30+(rand()%40))/20.);
    if (rand()&1) speech_gain = -speech_gain;
    if (rand()%20==0) speech_gain *= .01;
    if (!pitch && rand()%100==0) speech_gain = 0;

    noise_gain = pow(10., (-40+randf(25.f)+randf(15.f))/20.);
    if (rand()%2!=0) noise_gain = 0;
    if (rand()%12==0) {
      noise_gain *= 0.03;
    }
    noise_gain *= speech_gain;
    rand_resp(a_noise, b_noise);
    rand_resp(a_sig, b_sig);

    for (frame=0;frame<SEQUENCE_LENGTH;frame++) {
      E[frame] = 0;
      for(j=0;j<FRAME_SIZE;j++) {
        float s = speech16[frame*FRAME_SIZE+j];
        E[frame] += s*s;
        x[frame*FRAME_SIZE+j] = speech16[frame*FRAME_SIZE+j];
        n[frame*FRAME_SIZE+j] = noise16[frame*FRAME_SIZE+j];
      }
    }

    OPUS_CLEAR(mem, 2);
    biquad(x, mem, x, b_hp, a_hp, SEQUENCE_LENGTH*FRAME_SIZE);
    OPUS_CLEAR(mem, 2);
    biquad(x, mem, x, b_sig, a_sig, SEQUENCE_LENGTH*FRAME_SIZE);
    OPUS_CLEAR(mem, 2);
    biquad(n, mem, n, b_hp, a_hp, SEQUENCE_LENGTH*FRAME_SIZE);
    OPUS_CLEAR(mem, 2);
    biquad(n, mem, n, b_noise, a_noise, SEQUENCE_LENGTH*FRAME_SIZE);

    speech_rms = weighted_rms(x);
    noise_rms = weighted_rms(n);

    speech_gain *= 3000.f/(1+speech_rms);
    noise_gain *= 3000.f/(1+noise_rms);
    for (j=0;j<SEQUENCE_SAMPLES;j++) {
      x[j] *= speech_gain;
      n[j] *= noise_gain;
      xn[j] = x[j] + n[j];
    }
#ifdef ENABLE_RIR
    if (rir_filename!=NULL && rand()%3==0) {
      rir_id = rand()%rirs.nb_rirs;
      rir_filter_sequence(&rirs, x, rir_id, 1);
      rir_filter_sequence(&rirs, xn, rir_id, 0);
    }
#endif
    if (rand()%4==0) {
      /* Apply input clipping to 0 dBFS (don't clip target). */
      for (j=0;j<SEQUENCE_SAMPLES;j++) {
        xn[j] = MIN16(32767.f, MAX16(-32767.f, xn[j]));
      }
    }
    if (rand()%2==0) {
      /* Apply 16-bit quantization. */
      for (j=0;j<SEQUENCE_SAMPLES;j++) {
        xn[j] = floor(.5f + xn[j]);
      }
    }
#if 0
    for (frame=0;frame<SEQUENCE_LENGTH;frame++) {
       short tmp[FRAME_SIZE];
       for (j=0;j<FRAME_SIZE;j++) tmp[j] = MIN16(32767, MAX16(-32767, xn[frame*FRAME_SIZE+j]));
       fwrite(tmp, FRAME_SIZE, 2, fout);
    }
#endif

    sequence_length = IMIN(SEQUENCE_LENGTH, SEQUENCE_LENGTH/2 + rand()%(SEQUENCE_LENGTH/2+1));
    for (frame=0;frame<sequence_length;frame++) {
       float *xf = &xn[frame*FRAME_SIZE];
       if (burg) {
         float ceps[2*NB_BANDS];
         burg_cepstral_analysis(ceps, xf);
         fwrite(ceps, sizeof(float), 2*NB_BANDS, ffeat);
       }
       preemphasis(xf, &mem_preemph, xf, PREEMPHASIS, FRAME_SIZE);
       /* PCM is delayed by 1/2 frame to make the features centered on the frames. */
       for (i=0;i<FRAME_SIZE-TRAINING_OFFSET;i++) pcm[i+TRAINING_OFFSET] = float2short(xf[i]);
       compute_frame_features(st, xf, arch);

       if (pitch) {
         signed char pitch_features[PITCH_MAX_PERIOD-PITCH_MIN_PERIOD+PITCH_IF_FEATURES];
         for (i=0;i<PITCH_MAX_PERIOD-PITCH_MIN_PERIOD;i++) {
           pitch_features[i] = (int)floor(.5f + 127.f*st->xcorr_features[i]);
         }
         for (i=0;i<PITCH_IF_FEATURES;i++) {
           pitch_features[i+PITCH_MAX_PERIOD-PITCH_MIN_PERIOD] = (int)floor(.5f + 127.f*st->if_features[i]);
         }
         fwrite(pitch_features, PITCH_MAX_PERIOD-PITCH_MIN_PERIOD+PITCH_IF_FEATURES, 1, ffeat);
       } else {
         fwrite(st->features, sizeof(float), NB_TOTAL_FEATURES, ffeat);
       }
       /*if(pitch) fwrite(pcm, FRAME_SIZE, 2, stdout);*/
       if (fpcm) fwrite(pcm, FRAME_SIZE, 2, fpcm);
       for (i=0;i<TRAINING_OFFSET;i++) pcm[i] = float2short(xf[i+FRAME_SIZE-TRAINING_OFFSET]);
    }
  }

  fclose(f1);
  fclose(f2);
  fclose(ffeat);
  if (fpcm) fclose(fpcm);
  lpcnet_encoder_destroy(st);
  return 0;
}
