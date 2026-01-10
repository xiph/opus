/* Copyright (c) 2011-2012 Xiph.Org Foundation, Mozilla Corporation
   Written by Jean-Marc Valin and Timothy B. Terriberry */
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
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mini_kfft.c"

#define MAX(a,b) ((a)>(b) ? (a) : (b))
#define OPUS_PI (3.14159265F)

#define OPUS_COSF(_x)        ((float)cos(_x))
#define OPUS_SINF(_x)        ((float)sin(_x))

static void *check_alloc(void *_ptr){
  if(_ptr==NULL){
    fprintf(stderr,"Out of memory.\n");
    exit(EXIT_FAILURE);
  }
  return _ptr;
}

static void *opus_malloc(size_t _size){
  return check_alloc(malloc(_size));
}

static void *opus_realloc(void *_ptr,size_t _size){
  return check_alloc(realloc(_ptr,_size));
}

#define FORMAT_S16_LE 0
#define FORMAT_S24_LE 1
#define FORMAT_F32_LE 2

#define NBANDS (17)
#define NFREQS (320)
#define TEST_WIN_SIZE (640)
#define TEST_WIN_STEP (160)

static const int format_size[3] = {2, 3, 4};
typedef union {
    int i;
    float f;
} float_bits;

static void biquad(float *y, float mem[2], const float *x,
                   const float *b, const float *a, int N) {
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

static void buf_to_float(const unsigned char *buf, float *x, int len) {
   float_bits s;
   int i;
   for (i=0;i<len;i++) {
      s.i=(unsigned)buf[4*i+3]<<24
            |buf[4*i+2]<<16
            |buf[4*i+1]<<8
            |buf[4*i];
      x[i] = s.f;
   }
}

static size_t read_pcm(float **_samples,FILE *_fin,int _nchannels,
                       int format){
  unsigned char  buf[1024];
  float         *samples;
  size_t         nsamples;
  size_t         csamples;
  size_t         xi;
  size_t         nread;
  static const float a_hp[2] = {-1.97354f, 0.97417f};
  static const float b_hp[2] = {-2.f, 1.f};
  float mem[2] = {0};
  samples=NULL;
  nsamples=csamples=0;
  int size = format_size[format];

  for(;;){
    nread=fread(buf,size*_nchannels,1024/(size*_nchannels),_fin);
    if(nread<=0)break;
    if(nsamples+nread>csamples){
      do csamples=csamples<<1|1;
      while(nsamples+nread>csamples);
      samples=(float *)opus_realloc(samples,
       _nchannels*csamples*sizeof(*samples));
    }
    if (format==FORMAT_S16_LE) {
      for(xi=0;xi<nread;xi++){
        int ci;
        for(ci=0;ci<_nchannels;ci++){
          int s;
          s=buf[2*(xi*_nchannels+ci)+1]<<8|buf[2*(xi*_nchannels+ci)];
          s=((s&0xFFFF)^0x8000)-0x8000;
          samples[(nsamples+xi)*_nchannels+ci]=s;
        }
      }
    } else if (format==FORMAT_S24_LE) {
       for(xi=0;xi<nread;xi++){
         int ci;
         for(ci=0;ci<_nchannels;ci++){
           int s;
           s=buf[3*(xi*_nchannels+ci)+2]<<16
            |buf[3*(xi*_nchannels+ci)+1]<<8
            |buf[3*(xi*_nchannels+ci)];
           s=((s&0xFFFFFF)^0x800000)-0x800000;
           samples[(nsamples+xi)*_nchannels+ci]=(1.f/256.f)*s;
         }
       }
     } else if (format==FORMAT_F32_LE) {
        for(xi=0;xi<nread;xi++){
          int ci;
          for(ci=0;ci<_nchannels;ci++){
            float_bits s;
            s.i=(unsigned)buf[4*(xi*_nchannels+ci)+3]<<24
               |buf[4*(xi*_nchannels+ci)+2]<<16
               |buf[4*(xi*_nchannels+ci)+1]<<8
               |buf[4*(xi*_nchannels+ci)];
            samples[(nsamples+xi)*_nchannels+ci] = s.f*32768;
          }
        }
      } else {
        exit(1);
      }
    nsamples+=nread;
  }
  *_samples=(float *)opus_realloc(samples,
   _nchannels*nsamples*sizeof(*samples));
  biquad(*_samples, mem, *_samples, b_hp, a_hp, nsamples);
  return nsamples;
}

static void spectrum(float *_ps,const int *_bands, int _nbands,
                     const float *_in,int _nchannels, size_t _nframes,
                     int _window_sz, int _step, int _downsample){
  float window[TEST_WIN_SIZE];
  float x[TEST_WIN_SIZE];
  size_t xi;
  int    xj;
  int    ps_sz;
  mini_kiss_fft_cpx X[2][NFREQS+1];
  mini_kiss_fftr_cfg kfft;
  ps_sz=_window_sz/2;
  /* Blackman-Harris window. */
  for(xj=0;xj<_window_sz;xj++){
    double n = (xj+.5)/_window_sz;
    window[xj]=0.35875 - 0.48829*cos(2*OPUS_PI*n)
             + 0.14128*cos(4*OPUS_PI*n) - 0.01168*cos(6*OPUS_PI*n);
  }
  kfft = mini_kiss_fftr_alloc(_window_sz, 0, NULL, NULL);
  for(xi=0;xi<_nframes;xi++){
    int ci;
    int xk;
    int bi;
    for(ci=0;ci<_nchannels;ci++){
      for(xk=0;xk<_window_sz;xk++){
        x[xk]=window[xk]*_in[(xi*_step+xk)*_nchannels+ci];
      }
      mini_kiss_fftr(kfft, x, X[ci]);
    }
    for(bi=xj=0;bi<_nbands;bi++){
      float p[2]={0};
      for(;xj<_bands[bi+1];xj++){
        for(ci=0;ci<_nchannels;ci++){
          float re;
          float im;
          re = X[ci][xj].r*_downsample;
          im = X[ci][xj].i*_downsample;
          _ps[(xi*ps_sz+xj)*_nchannels+ci]=re*re+im*im+.1;
          p[ci]+=_ps[(xi*ps_sz+xj)*_nchannels+ci];
        }
      }
    }
  }
  free(kfft);
}


/*Bands on which we compute the pseudo-NMR (Bark-derived
  CELT bands).*/
static const int BANDS[NBANDS+1]={
  0,8,16,24,32,40,48,56,64,80,96,112,128,160,192,224,272,320
};


void usage(const char *_argv0) {
  fprintf(stderr,"Usage: %s -audio [-s16|-s24|-f32]"
  " [-thresholds err4 err16 pitch]"
  " <file1> <file2>\n", _argv0);
  fprintf(stderr,"       %s -features"
  " [-thresholds tot max pitch]"
  " <file1.f32> <file2.f32>\n", _argv0);

}

/* Taken from ancient CELT code */
void psydecay_init(float *decayL, float *decayR, int len, int Fs)
{
   int i;
   for (i=0;i<len;i++)
   {
      float f;
      float deriv;
      /* Real frequency (in Hz) */
      f = Fs*i*(1/(2.f*len));
      /* This is the derivative of the Vorbis freq->Bark function. */
      deriv = (8.288e-8 * f)/(3.4225e-16 *f*f*f*f + 1)
            + .009694/(5.476e-7 *f*f + 1) + 1e-4;
      /* Back to FFT bin units */
      deriv *= Fs*(1/(2.f*len));
      /* decay corresponding to -10dB/Bark */
      decayR[i] = pow(.1f, deriv);
      /* decay corresponding to -25dB/Bark */
      decayL[i] = pow(0.0031623f, deriv);
      /*printf ("%f %f\n", decayL[i], decayR[i]);*/
   }
}

#define PITCH_MIN 32
#define PITCH_MAX 256
#define PITCH_FRAME 320

static float inner_prod(const float *x, const float *y, int len) {
  float sum=0;
  int i;
  for (i=0;i<len;i++) sum += x[i]*y[i];
  return sum;
}

static void compute_xcorr(const float *x, float *xcorr) {
  int i;
  float xx;
  float filtered[PITCH_FRAME+PITCH_MAX];
  for (i=0;i<PITCH_FRAME+PITCH_MAX;i++) {
    filtered[i] = x[i-PITCH_MAX] - .8*x[i-PITCH_MAX-1];
  }
  xx = inner_prod(&filtered[PITCH_MAX],
                  &filtered[PITCH_MAX], PITCH_FRAME);
  for (i=0;i<=PITCH_MAX;i++) {
    float xy, yy;
    xy = inner_prod(&filtered[PITCH_MAX],
                    &filtered[PITCH_MAX-i], PITCH_FRAME);
    yy = inner_prod(&filtered[PITCH_MAX-i],
                    &filtered[PITCH_MAX-i], PITCH_FRAME);
    xcorr[i] = xy/sqrt(xx*yy+PITCH_FRAME);
  }
}

#define LOUDNESS 0.2f
int compare_audio(int _argc,const char **_argv, const char *argv0){
  FILE    *fin1;
  FILE    *fin2;
  float   *x;
  float   *y;
  float   *X;
  float   *Y;
  double    err4;
  double    err16;
  double    T2;
  size_t   xlength;
  size_t   ylength;
  size_t   nframes;
  size_t   xi;
  int      ci;
  int      xj;
  int      bi;
  int      nchannels;
  int      downsample;
  int      nbands;
  int      ybands;
  int      nfreqs;
  int      yfreqs;
  size_t   test_win_size;
  size_t   test_win_step;
  int      max_compare;
  int      format;
  int      skip=0;
  double err4_threshold=-1, err16_threshold=-1, pitch_threshold=-1;
  int compare_thresholds=0;
  float decayL[NFREQS], decayR[NFREQS];
  float norm[NFREQS];
  float pitch_error=0;
  int pitch_count=0;
  psydecay_init(decayL, decayR, NFREQS, 16000);
  if(_argc<3){
    usage(argv0);
    return EXIT_FAILURE;
  }
  nchannels=1;
  ybands=nbands=NBANDS;
  nfreqs=NFREQS;
  test_win_size=TEST_WIN_SIZE;
  test_win_step=TEST_WIN_STEP;
  downsample=1;
  format=FORMAT_S16_LE;
  while (_argc > 3) {
    if(strcmp(_argv[1],"-s16")==0){
      format=FORMAT_S16_LE;
      _argv++;
      _argc--;
    } else if(strcmp(_argv[1],"-s24")==0){
      format=FORMAT_S24_LE;
      _argv++;
      _argc--;
    } else if(strcmp(_argv[1],"-f32")==0){
      format=FORMAT_F32_LE;
      _argv++;
      _argc--;
    } else if(strcmp(_argv[1],"-skip")==0){
      skip=atoi(_argv[2]);
      _argv+=2;
      _argc-=2;
    } else if(strcmp(_argv[1],"-thresholds")==0){
      if (_argc < 7) {
        usage(argv0);
        return EXIT_FAILURE;
      }
      err4_threshold=atof(_argv[2]);
      err16_threshold=atof(_argv[3]);
      pitch_threshold=atof(_argv[4]);
      compare_thresholds=1;
      _argv+=4;
      _argc-=4;
    } else {
      usage(argv0);
      return EXIT_FAILURE;
    }
  }
  if(_argc!=3){
    usage(argv0);
    return EXIT_FAILURE;
  }
  downsample=1;
  yfreqs=nfreqs/downsample;
  fin1=fopen(_argv[1],"rb");
  if(fin1==NULL){
    fprintf(stderr,"Error opening '%s'.\n",_argv[1]);
    return EXIT_FAILURE;
  }
  fin2=fopen(_argv[2],"rb");
  if(fin2==NULL){
    fprintf(stderr,"Error opening '%s'.\n",_argv[2]);
    fclose(fin1);
    return EXIT_FAILURE;
  }
  /*Read in the data and allocate scratch space.*/
  xlength=read_pcm(&x,fin1,1,format);
  fclose(fin1);
  ylength=read_pcm(&y,fin2,nchannels,format);
  fclose(fin2);
  skip *= nchannels;
  y += skip/downsample;
  ylength -= skip/downsample;
  if (ylength*downsample > xlength) ylength = xlength/downsample;
  if(xlength!=ylength*downsample){
    fprintf(stderr,"Sample counts do not match (%lu!=%lu).\n",
     (unsigned long)xlength,(unsigned long)ylength*downsample);
    return EXIT_FAILURE;
  }
  if(xlength<test_win_size){
    fprintf(stderr,"Insufficient sample data (%lu<%lu).\n",
     (unsigned long)xlength,(unsigned long)test_win_size);
    return EXIT_FAILURE;
  }
  nframes=(xlength-test_win_size+test_win_step)/test_win_step;
  X=(float *)opus_malloc(nframes*nfreqs*nchannels*sizeof(*X));
  Y=(float *)opus_malloc(nframes*yfreqs*nchannels*sizeof(*Y));

  for(xi=2;xi<nframes-2;xi++){
    float xcorr[PITCH_MAX+1], ycorr[PITCH_MAX+1];
    int i;
    float maxcorr=-1;
    int pitch=0;
    compute_xcorr(&x[xi*TEST_WIN_STEP], xcorr);
    compute_xcorr(&y[xi*TEST_WIN_STEP], ycorr);
    for (i=PITCH_MIN;i<=PITCH_MAX;i++) {
      if (xcorr[i] > maxcorr) {
        maxcorr = xcorr[i];
        pitch = i;
      }
    }
    if (xcorr[pitch] > .7) {
      pitch_error += fabs(xcorr[pitch]-ycorr[pitch]);
      pitch_count++;
    }
  }
  pitch_error /= pitch_count;

  /*Compute the per-band spectral energy of the original signal
     and the error.*/
  spectrum(X,BANDS,nbands,x,nchannels,nframes,
        test_win_size,test_win_step,1);
  free(x);
  spectrum(Y,BANDS,ybands,y,nchannels,nframes,
        test_win_size/downsample,test_win_step/downsample,downsample);
  free(y-skip/downsample);

  norm[0]=1;
  for(xj=1;xj<NFREQS;xj++){
    norm[xj] = 1 + decayR[xj]*norm[xj-1];
  }
  for(xj=NFREQS-2;xj>=0;xj--){
    norm[xj] = norm[xj] + decayL[xj]*norm[xj+1];
  }
  for(xj=0;xj<NFREQS;xj++) norm[xj] = 1.f/norm[xj];
  for(xi=0;xi<nframes;xi++){
    for(xj=1;xj<NFREQS;xj++){
      X[xi*nfreqs+xj] = X[xi*nfreqs+xj]+decayR[xj]*X[xi*nfreqs+xj-1];
      Y[xi*nfreqs+xj] = Y[xi*nfreqs+xj]+decayR[xj]*Y[xi*nfreqs+xj-1];
    }
    for(xj=NFREQS-2;xj>=0;xj--){
      X[xi*nfreqs+xj] = X[xi*nfreqs+xj]+decayL[xj]*X[xi*nfreqs+xj+1];
      Y[xi*nfreqs+xj] = Y[xi*nfreqs+xj]+decayL[xj]*Y[xi*nfreqs+xj+1];
    }
    for(xj=0;xj<NFREQS;xj++){
      X[xi*nfreqs+xj] *= norm[xj];
      Y[xi*nfreqs+xj] *= norm[xj];
    }
  }

  for(xi=0;xi<nframes;xi++){
    float maxE=0;
    for(xj=0;xj<NFREQS;xj++){
      maxE = MAX(maxE, X[xi*nfreqs+xj]);
    }
    /* Allow for up to 80 dB instantaneous dynamic range. */
    for(xj=0;xj<NFREQS;xj++){
      X[xi*nfreqs+xj] = MAX(1e-8*maxE, X[xi*nfreqs+xj]);
      Y[xi*nfreqs+xj] = MAX(1e-8*maxE, Y[xi*nfreqs+xj]);
    }
    /* Forward temporal masking: -3 dB/2.5ms slope.*/
    if(xi>0){
      for(xj=0;xj<NFREQS;xj++){
        X[xi*nfreqs+xj] += .5f*X[(xi-1)*nfreqs+xj];
        Y[xi*nfreqs+xj] += .5f*Y[(xi-1)*nfreqs+xj];
      }
    }
  }

  /*Backward temporal masking: -10 dB/2.5ms slope.*/
  for(xi=nframes-2;xi-->0;){
    for(xj=0;xj<NFREQS;xj++){
      X[xi*nfreqs+xj] += .1f*X[(xi+1)*nfreqs+xj];
      Y[xi*nfreqs+xj] += .1f*Y[(xi+1)*nfreqs+xj];
    }
  }

  max_compare=BANDS[nbands];
  err4=0;
  err16=0;
  T2=0;
  for(xi=0;xi<nframes;xi++){
    double Ef2, Ef4, Tf2;
    Ef2=0;
    Ef4=0;
    Tf2=0;
    for(bi=0;bi<ybands;bi++){
      double Eb2, Eb4, Tb2;
      double w;
      Tb2=Eb2=Eb4=0;
      w = 1.f/(BANDS[bi+1]-BANDS[bi]);
      for(xj=BANDS[bi];xj<BANDS[bi+1]&&xj<max_compare;xj++){
        float f, thresh;
        f = xj*OPUS_PI/960;
        /* Shape the lower threshold similar to 1/(1 - 0.85*z^-1)
           deemphasis filter at 48 kHz. */
        thresh = .1/(.15*.15 + f*f);
        for(ci=0;ci<nchannels;ci++){
          float re;
          float im;
          re = pow(Y[(xi*yfreqs+xj)*nchannels+ci]+thresh,LOUDNESS)
             - pow(X[(xi*nfreqs+xj)*nchannels+ci]+thresh,LOUDNESS);
          im = re*re;
          Tb2 += w*pow(X[(xi*nfreqs+xj)*nchannels+ci]+thresh,
                     2*LOUDNESS);
          /* Per-band error weighting. */
          im *= w;
          Eb2+=im;
          /* Same for 4th power, but make it less sensitive to
             very low energies. */
          re = pow(Y[(xi*yfreqs+xj)*nchannels+ci]+10*thresh,LOUDNESS)
             - pow(X[(xi*nfreqs+xj)*nchannels+ci]+10*thresh,LOUDNESS);
          im = re*re;
          /* Per-band error weighting. */
          im *= w;
          Eb4+=im;
        }
      }
      Eb2 /= (BANDS[bi+1]-BANDS[bi])*nchannels;
      Eb4 /= (BANDS[bi+1]-BANDS[bi])*nchannels;
      Tb2 /= (BANDS[bi+1]-BANDS[bi])*nchannels;
      Ef2 += Eb2;
      Ef4 += Eb4*Eb4;
      Tf2 += Tb2;
    }
    Ef2/=nbands;
    Ef4/=nbands;
    Ef4*=Ef4;
    Tf2/=nbands;
    err4+=Ef2*Ef2;
    err16+=Ef4*Ef4;
    T2 += Tf2;
  }
  free(X);
  free(Y);
  err4=100*pow(err4/nframes,1.0/4)/sqrt(T2);
  err16=100*pow(err16/nframes,1.0/16)/sqrt(T2);
  fprintf(stderr, "err4 = %f, err16 = %f, pitch = %f\n",
          err4, err16, pitch_error);
  if (compare_thresholds) {
    if (err4 <= err4_threshold && err16 <= err16_threshold && pitch_error <= pitch_threshold) {
      fprintf(stderr, "Comparison PASSED\n");
    } else {
      fprintf(stderr, "*** Comparison FAILED ***"
      " (thresholds were %f %f %f)\n", err4_threshold, err16_threshold, pitch_threshold);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

#define NB_FEATURES 20

int compare_features(int _argc,const char **_argv, const char *argv0){
   FILE    *fin1=NULL;
   FILE    *fin2=NULL;
   int i;
   float mse[NB_FEATURES]={0};
   int count=0;
   int pitch_count=0;
   float pitch_error=0;
   float tot_error=0, max_error=0;
   float tot_threshold=-1, max_threshold=-1, pitch_threshold=-1;
   int compare_thresholds=0;
   if(strcmp(_argv[1],"-thresholds")==0){
     if (_argc < 7) {
       usage(argv0);
       return EXIT_FAILURE;
     }
     tot_threshold=atof(_argv[2]);
     max_threshold=atof(_argv[3]);
     pitch_threshold=atof(_argv[4]);
     compare_thresholds=1;
     _argv+=4;
     _argc-=4;
   }
   if(_argc!=3) {
      usage(argv0);
      return EXIT_FAILURE;
   }
   fin1=fopen(_argv[1],"rb");
   if(fin1==NULL){
     fprintf(stderr,"Error opening '%s'.\n",_argv[1]);
     return EXIT_FAILURE;
   }
   fin2=fopen(_argv[2],"rb");
   if(fin2==NULL){
     fprintf(stderr,"Error opening '%s'.\n",_argv[2]);
     fclose(fin1);
     return EXIT_FAILURE;
   }
   while (1) {
      int ret;
      float x[NB_FEATURES], y[NB_FEATURES];
      unsigned char buf[NB_FEATURES*4];
      ret = fread(buf, NB_FEATURES*4, 1, fin1);
      if (ret != 1) break;
      buf_to_float(buf, x, NB_FEATURES);
      ret = fread(buf, NB_FEATURES*4, 1, fin2);
      if (ret != 1) {
         fprintf(stderr, "Error: truncated test file\n");
         return EXIT_FAILURE;
      }
      buf_to_float(buf, y, NB_FEATURES);
      for (i=0;i<NB_FEATURES;i++) {
         float e = (x[i]-y[i]);
         mse[i] += e*e;
      }
      if (x[NB_FEATURES-1] > .2) {
         pitch_error += fabs(x[NB_FEATURES-2]-y[NB_FEATURES-2]);
         pitch_count++;
      }
      count++;
   }
   pitch_error /= pitch_count;
   for(i=0;i<NB_FEATURES;i++) {
      mse[i] /= count;
      if (i != NB_FEATURES-2) {
         tot_error += mse[i];
         max_error = MAX(max_error, mse[i]);
      }
   }
   tot_error = sqrt(tot_error);
   max_error = sqrt(max_error);
   fprintf(stderr, "total = %f, max = %f, pitch = %f\n", tot_error, max_error, pitch_error);
   if (compare_thresholds) {
     if (tot_error <= tot_threshold && max_error <= max_threshold && pitch_error <= pitch_threshold) {
       fprintf(stderr, "Comparison PASSED\n");
     } else {
       fprintf(stderr, "*** Comparison FAILED ***"
       " (thresholds were %f %f %f)\n", tot_threshold, max_threshold, pitch_threshold);
       return EXIT_FAILURE;
     }
   }
   if (fin1) fclose(fin1);
   if (fin2) fclose(fin2);
   return EXIT_SUCCESS;
}

int main(int _argc,const char **_argv){
  if (_argc<3) {
     usage(_argv[0]);
  } else if (strcmp(_argv[1], "-audio")==0) {
     return compare_audio(_argc-1, _argv+1, _argv[0]);
  } else if (strcmp(_argv[1], "-features")==0) {
     return compare_features(_argc-1, _argv+1, _argv[0]);
  } else {
     fprintf(stderr, "%s: First argument must be either -audio or -features\n\n", _argv[0]);
     usage(_argv[0]);
  }
  return EXIT_FAILURE;
}
