#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define OPUS_PI (3.14159265F)

#define OPUS_MIN(_x,_y)      ((_x)<(_y)?(_x):(_y))
#define OPUS_MAX(_x,_y)      ((_x)>(_y)?(_x):(_y))
#define OPUS_CLAMP(_a,_b,_c) OPUS_MAX(_a,OPUS_MIN(_b,_c))
#define OPUS_COSF(_x)        ((float)cos(_x))
#define OPUS_SINF(_x)        ((float)sin(_x))
#define OPUS_SQRTF(_x)       ((float)sqrt(_x))
#define OPUS_LOG10F(_x)      ((float)log10(_x))

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

static size_t read_pcm16(float **_samples,FILE *_fin,
                         int _nchannels){
  unsigned char  buf[1024];
  float         *samples;
  size_t         nsamples;
  size_t         csamples;
  size_t         xi;
  size_t         nread;
  samples=NULL;
  nsamples=csamples=0;
  for(;;){
    nread=fread(buf,2*_nchannels,1024/(2*_nchannels),_fin);
    if(nread<=0)break;
    if(nsamples+nread>csamples){
      do csamples=csamples<<1|1;
      while(nsamples+nread>csamples);
      samples=(float *)opus_realloc(samples,
       _nchannels*csamples*sizeof(*samples));
    }
    for(xi=0;xi<nread;xi++){
      int ci;
      for(ci=0;ci<_nchannels;ci++){
        int s;
        s=buf[2*(xi*_nchannels+ci)+1]<<8|buf[2*(xi*_nchannels+ci)];
        s=((s&0xFFFF)^0x8000)-0x8000;
        samples[(nsamples+xi)*_nchannels+ci]=s;
      }
    }
    nsamples+=nread;
  }
  *_samples=(float *)opus_realloc(samples,
                     _nchannels*nsamples*sizeof(*samples));
  return nsamples;
}

static void band_energy(float *_out,const int *_bands,int _nbands,
 const float *_in,int _nchannels,size_t _nframes,int _window_sz,
 int _step){
  float *window;
  float *x;
  float *c;
  float *s;
  size_t xi;
  int    xj;
  window=(float *)opus_malloc((3+_nchannels)*_window_sz
          *sizeof(*window));
  c=window+_window_sz;
  s=c+_window_sz;
  x=s+_window_sz;
  for(xj=0;xj<_window_sz;xj++){
    window[xj]=0.5F-0.5F*OPUS_COSF((2*OPUS_PI/(_window_sz-1))*xj);
  }
  for(xj=0;xj<_window_sz;xj++)
      c[xj]=OPUS_COSF((2*OPUS_PI/_window_sz)*xj);
  for(xj=0;xj<_window_sz;xj++)
      s[xj]=OPUS_SINF((2*OPUS_PI/_window_sz)*xj);
  for(xi=0;xi<_nframes;xi++){
    int ci;
    int xk;
    int bi;
    for(ci=0;ci<_nchannels;ci++){
      for(xk=0;xk<_window_sz;xk++){
        x[ci*_window_sz+xk]=window[xk]
                           *_in[(xi*_step+xk)*_nchannels+ci];
      }
    }
    for(bi=xj=0;bi<_nbands;bi++){
      float e2;
      e2=0;
      for(;xj<_bands[bi+1];xj++){
        float p;
        p=0;
        for(ci=0;ci<_nchannels;ci++){
          float re;
          float im;
          int   ti;
          ti=0;
          re=im=0;
          for(xk=0;xk<_window_sz;xk++){
            re+=c[ti]*x[ci*_window_sz+xk];
            im-=s[ti]*x[ci*_window_sz+xk];
            ti+=xj;
            if(ti>=_window_sz)ti-=_window_sz;
          }
          p+=OPUS_SQRTF(re*re+im*im);
        }
        p*=(1.0F/_nchannels);
        e2+=p*p;
      }
      _out[xi*_nbands+bi]=e2/(_bands[bi+1]-_bands[bi])+1;
    }
  }
  free(window);
}

static int cmp_float(const void *_a,const void *_b){
  float a;
  float b;
  a=*(const float *)_a;
  b=*(const float *)_b;
  return (a>b)-(a<b);
}

#define NBANDS (21)

/*Bands on which we compute the pseudo-NMR (Bark-derived
  CELT bands).*/
static const int BANDS[NBANDS+1]={
  0,2,4,6,8,10,12,14,16,20,24,28,32,40,48,56,68,80,96,120,156,200
};

/*Per-band NMR threshold.*/
static const float NMR_THRESH[NBANDS]={
85113.804F,72443.596F,61659.5F,  52480.746F,44668.359F,38018.940F,
32359.366F,27542.287F,23442.288F,19952.623F,16982.437F,14454.398F,
12302.688F,10471.285F, 8912.5094F,7585.7758F,6456.5423F,5495.4087F,
 4677.3514F,3981.0717F,3388.4416F
};

/*Noise floor.*/
static const float NOISE_FLOOR[NBANDS]={
8.7096359F,7.5857758F,6.6069345F,5.7543994F,5.0118723F,4.3651583F,
3.8018940F,3.3113112F,2.8840315F,2.5118864F,2.1877616F,1.9054607F,
1.6595869F,1.4454398F,1.2589254F,1.0964782F,0.95499259F,0.83176377F,
0.72443596F,0.63095734F,0.54954087F
};

#define TEST_WIN_SIZE (480)
#define TEST_WIN_STEP (TEST_WIN_SIZE>>1)

int main(int _argc,const char **_argv){
  FILE   *fin1;
  FILE   *fin2;
  float  *x;
  float  *y;
  float  *xb;
  float  *eb;
  float  *nmr;
  float   thresh;
  float   mismatch;
  float   err;
  float   nmr_sum;
  size_t  weight;
  size_t  xlength;
  size_t  ylength;
  size_t  nframes;
  size_t  xi;
  int     bi;
  int     nchannels;
  if(_argc<3||_argc>4){
    fprintf(stderr,"Usage: %s [-s] <file1.sw> <file2.sw>\n",
            _argv[0]);
    return EXIT_FAILURE;
  }
  nchannels=1;
  if(strcmp(_argv[1],"-s")==0)nchannels=2;
  fin1=fopen(_argv[nchannels],"rb");
  if(fin1==NULL){
    fprintf(stderr,"Error opening '%s'.\n",_argv[nchannels]);
    return EXIT_FAILURE;
  }
  fin2=fopen(_argv[nchannels+1],"rb");
  if(fin2==NULL){
    fprintf(stderr,"Error opening '%s'.\n",_argv[nchannels+1]);
    fclose(fin1);
    return EXIT_FAILURE;
  }
  /*Read in the data and allocate scratch space.*/
  xlength=read_pcm16(&x,fin1,nchannels);
  fclose(fin1);
  ylength=read_pcm16(&y,fin2,nchannels);
  fclose(fin2);
  if(xlength!=ylength){
    fprintf(stderr,"Sample counts do not match (%lu!=%lu).\n",
     (unsigned long)xlength,(unsigned long)ylength);
    return EXIT_FAILURE;
  }
  if(xlength<TEST_WIN_SIZE){
    fprintf(stderr,"Insufficient sample data (%lu<%i).\n",
     (unsigned long)xlength,TEST_WIN_SIZE);
    return EXIT_FAILURE;
  }
  nframes=(xlength-TEST_WIN_SIZE+TEST_WIN_STEP)/TEST_WIN_STEP;
  xb=(float *)opus_malloc(nframes*NBANDS*sizeof(*xb));
  eb=(float *)opus_malloc(nframes*NBANDS*sizeof(*eb));
  nmr=(float *)opus_malloc(nframes*NBANDS*sizeof(*nmr));
  /*Compute the error signal.*/
  for(xi=0;xi<xlength*nchannels;xi++){
    err=x[xi]-y[xi];
    y[xi]=err-OPUS_CLAMP(-1,err,1);
  }
  /*Compute the per-band spectral energy of the original signal
    and the error.*/
  band_energy(xb,BANDS,NBANDS,x,nchannels,nframes,
          TEST_WIN_SIZE,TEST_WIN_STEP);
  free(x);
  band_energy(eb,BANDS,NBANDS,y,nchannels,nframes,
          TEST_WIN_SIZE,TEST_WIN_STEP);
  free(y);
  nmr_sum=0;
  for(xi=0;xi<nframes;xi++){
    /*Frequency masking (low to high): 10 dB/Bark slope.*/
    for(bi=1;bi<NBANDS;bi++)
        xb[xi*NBANDS+bi]+=0.1F*xb[xi*NBANDS+bi-1];
    /*Frequency masking (high to low): 15 dB/Bark slope.*/
    for(bi=NBANDS-1;bi-->0;)
        xb[xi*NBANDS+bi]+=0.03F*xb[xi*NBANDS+bi+1];
    if(xi>0){
      /*Temporal masking: 5 dB/5ms slope.*/
      for(bi=0;bi<NBANDS;bi++)
          xb[xi*NBANDS+bi]+=0.3F*xb[(xi-1)*NBANDS+bi];
    }
    /*Compute NMR.*/
    for(bi=0;bi<NBANDS;bi++){
      nmr[xi*NBANDS+bi]=xb[xi*NBANDS+bi]/eb[xi*NBANDS+bi];
      nmr_sum+=10*OPUS_LOG10F(nmr[xi*NBANDS+bi]);
    }
  }
  /*Find the 90th percentile of the errors.*/
  memcpy(xb,eb,nframes*NBANDS*sizeof(*xb));
  qsort(xb,nframes*NBANDS,sizeof(*xb),cmp_float);
  thresh=xb[(9*nframes*NBANDS+5)/10];
  free(xb);
  /*Compute the mismatch.*/
  mismatch=0;
  weight=0;
  for(xi=0;xi<nframes;xi++){
    for(bi=0;bi<NBANDS;bi++){
      if(eb[xi*NBANDS+bi]>thresh){
        mismatch+=NMR_THRESH[bi]/nmr[xi*NBANDS+bi];
        weight++;
      }
    }
  }
  free(nmr);
  free(eb);
  printf("Average pseudo-NMR: %3.2f dB\n",nmr_sum/(nframes*NBANDS));
  if(weight<=0){
    err=-100;
    printf("Mismatch level: below noise floor\n");
  }
  else{
    err=10*OPUS_LOG10F(mismatch/weight);
    printf("Weighted mismatch: %3.2f dB\n",err);
  }
  printf("\n");
  if(err<0){
    printf("**Decoder PASSES test (mismatch < 0 dB)\n");
    return EXIT_SUCCESS;
  }
  printf("**Decoder FAILS test (mismatch >= 0 dB)\n");
  return EXIT_FAILURE;
}
