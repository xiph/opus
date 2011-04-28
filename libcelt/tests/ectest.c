#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>       
#include "entcode.h"
#include "entenc.h"
#include "entdec.h"
#include <string.h>

#include "../libcelt/entenc.c"
#include "../libcelt/entdec.c"
#include "../libcelt/entcode.c"

#ifndef M_LOG2E
# define M_LOG2E    1.4426950408889634074
#endif
#define DATA_SIZE 10000000
#define DATA_SIZE2 10000

int main(int _argc,char **_argv){
  ec_enc         enc;
  ec_dec         dec;
  long           nbits;
  long           nbits2;
  double         entropy;
  int            ft;
  int            ftb;
  int            sym;
  int            sz;
  int            i;
  int            ret;
  unsigned int   seed;
  unsigned char *ptr;
  ret=0;
  entropy=0;
    if (_argc > 2) {
	fprintf(stderr, "Usage: %s [<seed>]\n", _argv[0]);
	return 1;
    }
    if (_argc > 1)
	seed = atoi(_argv[1]);
    else
	seed = time(NULL);
  /*Testing encoding of raw bit values.*/
  ptr = malloc(DATA_SIZE);
  ec_enc_init(&enc,ptr, DATA_SIZE);
  for(ft=2;ft<1024;ft++){
    for(i=0;i<ft;i++){
      entropy+=log(ft)*M_LOG2E;
      ec_enc_uint(&enc,i,ft);
    }
  }
  /*Testing encoding of raw bit values.*/
  for(ftb=0;ftb<16;ftb++){
    for(i=0;i<(1<<ftb);i++){
      entropy+=ftb;
      nbits=ec_tell(&enc);
      ec_enc_bits(&enc,i,ftb);
      nbits2=ec_tell(&enc);
      if(nbits2-nbits!=ftb){
        fprintf(stderr,"Used %li bits to encode %i bits directly.\n",
         nbits2-nbits,ftb);
        ret=-1;
      }
    }
  }
  nbits=ec_tell_frac(&enc);
  ec_enc_done(&enc);
  fprintf(stderr,
   "Encoded %0.2lf bits of entropy to %0.2lf bits (%0.3lf%% wasted).\n",
   entropy,ldexp(nbits,-3),100*(nbits-ldexp(entropy,3))/nbits);
  fprintf(stderr,"Packed to %li bytes.\n",(long)ec_range_bytes(&enc));
  ec_dec_init(&dec,ptr,DATA_SIZE);
  for(ft=2;ft<1024;ft++){
    for(i=0;i<ft;i++){
      sym=ec_dec_uint(&dec,ft);
      if(sym!=i){
        fprintf(stderr,"Decoded %i instead of %i with ft of %i.\n",sym,i,ft);
        ret=-1;
      }
    }
  }
  for(ftb=0;ftb<16;ftb++){
    for(i=0;i<(1<<ftb);i++){
      sym=ec_dec_bits(&dec,ftb);
      if(sym!=i){
        fprintf(stderr,"Decoded %i instead of %i with ftb of %i.\n",sym,i,ftb);
        ret=-1;
      }
    }
  }
  nbits2=ec_tell_frac(&dec);
  if(nbits!=nbits2){
    fprintf(stderr,
     "Reported number of bits used was %0.2lf, should be %0.2lf.\n",
     ldexp(nbits2,-3),ldexp(nbits,-3));
    ret=-1;
  }
  srand(seed);
  fprintf(stderr,"Testing random streams... Random seed: %u (%.4X)\n", seed, rand() % 65536);
  for(i=0;i<409600;i++){
    unsigned *data;
    unsigned *tell;
    int       j;
    int tell_bits;
    int zeros;
    ft=rand()/((RAND_MAX>>(rand()%11))+1)+10;
    sz=rand()/((RAND_MAX>>(rand()%9))+1);
    data=(unsigned *)malloc(sz*sizeof(*data));
    tell=(unsigned *)malloc((sz+1)*sizeof(*tell));
    ec_enc_init(&enc,ptr,DATA_SIZE2);
    zeros = rand()%13==0;
    tell[0]=ec_tell_frac(&enc);
    for(j=0;j<sz;j++){
      if (zeros)
        data[j]=0;
      else
        data[j]=rand()%ft;
      ec_enc_uint(&enc,data[j],ft);
      tell[j+1]=ec_tell_frac(&enc);
    }
    if (rand()%2==0)
      while(ec_tell(&enc)%8 != 0)
        ec_enc_uint(&enc, rand()%2, 2);
    tell_bits = ec_tell(&enc);
    ec_enc_done(&enc);
    if(tell_bits!=ec_tell(&enc)){
      fprintf(stderr,"ec_tell() changed after ec_enc_done(): %i instead of %i (Random seed: %u)\n",
       ec_tell(&enc),tell_bits,seed);
      ret=-1;
    }
    if ((tell_bits+7)/8 < ec_range_bytes(&enc))
    {
      fprintf (stderr, "ec_tell() lied, there's %i bytes instead of %d (Random seed: %u)\n",
               ec_range_bytes(&enc), (tell_bits+7)/8,seed);
      ret=-1;
    }
    tell_bits -= 8*ec_range_bytes(&enc);
    ec_dec_init(&dec,ptr,DATA_SIZE2);
    if(ec_tell_frac(&dec)!=tell[0]){
      fprintf(stderr,
       "Tell mismatch between encoder and decoder at symbol %i: %i instead of %i (Random seed: %u).\n",
       0,ec_tell_frac(&dec),tell[0],seed);
    }
    for(j=0;j<sz;j++){
      sym=ec_dec_uint(&dec,ft);
      if(sym!=data[j]){
        fprintf(stderr,
         "Decoded %i instead of %i with ft of %i at position %i of %i (Random seed: %u).\n",
         sym,data[j],ft,j,sz,seed);
        ret=-1;
      }
      if(ec_tell_frac(&dec)!=tell[j+1]){
        fprintf(stderr,
         "Tell mismatch between encoder and decoder at symbol %i: %i instead of %i (Random seed: %u).\n",
         j+1,ec_tell_frac(&dec),tell[j+1],seed);
      }
    }
    free(tell);
    free(data);
  }
  /*Test compatibility between multiple different encode/decode routines.*/
  for(i=0;i<409600;i++){
    unsigned *logp1;
    unsigned *data;
    unsigned *tell;
    unsigned *enc_method;
    int       j;
    sz=rand()/((RAND_MAX>>(rand()%9))+1);
    logp1=(unsigned *)malloc(sz*sizeof(*logp1));
    data=(unsigned *)malloc(sz*sizeof(*data));
    tell=(unsigned *)malloc((sz+1)*sizeof(*tell));
    enc_method=(unsigned *)malloc(sz*sizeof(*enc_method));
    ec_enc_init(&enc,ptr,DATA_SIZE2);
    tell[0]=ec_tell_frac(&enc);
    for(j=0;j<sz;j++){
      data[j]=rand()/((RAND_MAX>>1)+1);
      logp1[j]=(rand()%15)+1;
      enc_method[j]=rand()/((RAND_MAX>>2)+1);
      switch(enc_method[j]){
        case 0:{
          ec_encode(&enc,data[j]?(1<<logp1[j])-1:0,
           (1<<logp1[j])-(data[j]?0:1),1<<logp1[j]);
        }break;
        case 1:{
          ec_encode_bin(&enc,data[j]?(1<<logp1[j])-1:0,
           (1<<logp1[j])-(data[j]?0:1),logp1[j]);
        }break;
        case 2:{
          ec_enc_bit_logp(&enc,data[j],logp1[j]);
        }break;
        case 3:{
          unsigned char icdf[2];
          icdf[0]=1;
          icdf[1]=0;
          ec_enc_icdf(&enc,data[j],icdf,logp1[j]);
        }break;
      }
      tell[j+1]=ec_tell_frac(&enc);
    }
    ec_enc_done(&enc);
    if((ec_tell(&enc)+7)/8<ec_range_bytes(&enc)){
      fprintf(stderr,"tell() lied, there's %i bytes instead of %d (Random seed: %u)\n",
       ec_range_bytes(&enc),(ec_tell(&enc)+7)/8,seed);
      ret=-1;
    }
    ec_dec_init(&dec,ptr,DATA_SIZE2);
    if(ec_tell_frac(&dec)!=tell[0]){
      fprintf(stderr,
       "Tell mismatch between encoder and decoder at symbol %i: %i instead of %i (Random seed: %u).\n",
       0,ec_tell_frac(&dec),tell[0],seed);
    }
    for(j=0;j<sz;j++){
      int fs;
      int dec_method;
      dec_method=rand()/((RAND_MAX>>2)+1);
      switch(dec_method){
        case 0:{
          fs=ec_decode(&dec,1<<logp1[j]);
          sym=fs>=(1<<logp1[j])-1;
          ec_dec_update(&dec,sym?(1<<logp1[j])-1:0,
           (1<<logp1[j])-(sym?0:1),1<<logp1[j]);
        }break;
        case 1:{
          fs=ec_decode_bin(&dec,logp1[j]);
          sym=fs>=(1<<logp1[j])-1;
          ec_dec_update(&dec,sym?(1<<logp1[j])-1:0,
           (1<<logp1[j])-(sym?0:1),1<<logp1[j]);
        }break;
        case 2:{
          sym=ec_dec_bit_logp(&dec,logp1[j]);
        }break;
        case 3:{
          unsigned char icdf[2];
          icdf[0]=1;
          icdf[1]=0;
          sym=ec_dec_icdf(&dec,icdf,logp1[j]);
        }break;
      }
      if(sym!=data[j]){
        fprintf(stderr,
         "Decoded %i instead of %i with logp1 of %i at position %i of %i (Random seed: %u).\n",
         sym,data[j],logp1[j],j,sz,seed);
        fprintf(stderr,"Encoding method: %i, decoding method: %i\n",
         enc_method[j],dec_method);
        ret=-1;
      }
      if(ec_tell_frac(&dec)!=tell[j+1]){
        fprintf(stderr,
         "Tell mismatch between encoder and decoder at symbol %i: %i instead of %i (Random seed: %u).\n",
         j+1,ec_tell_frac(&dec),tell[j+1],seed);
      }
    }
    free(enc_method);
    free(tell);
    free(data);
    free(logp1);
  }
  free(ptr);
  return ret;
}
