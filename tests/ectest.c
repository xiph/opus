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

#include "../libcelt/rangeenc.c"
#include "../libcelt/rangedec.c"
#include "../libcelt/entenc.c"
#include "../libcelt/entdec.c"
#include "../libcelt/entcode.c"

#ifndef M_LOG2E
# define M_LOG2E    1.4426950408889634074
#endif
#define DATA_SIZE 10000000
#define DATA_SIZE2 10000

int main(int _argc,char **_argv){
  ec_byte_buffer buf;
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
  ret=0;
  entropy=0;
  unsigned char *ptr;
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
  ec_byte_writeinit_buffer(&buf, ptr, DATA_SIZE);
  ec_enc_init(&enc,&buf);
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
      nbits=ec_enc_tell(&enc,0);
      ec_enc_bits(&enc,i,ftb);
      nbits2=ec_enc_tell(&enc,0);
      if(nbits2-nbits!=ftb){
        fprintf(stderr,"Used %li bits to encode %i bits directly.\n",
         nbits2-nbits,ftb);
        ret=-1;
      }
    }
  }
  nbits=ec_enc_tell(&enc,4);
  ec_enc_done(&enc);
  fprintf(stderr,
   "Encoded %0.2lf bits of entropy to %0.2lf bits (%0.3lf%% wasted).\n",
   entropy,ldexp(nbits,-4),100*(nbits-ldexp(entropy,4))/nbits);
  fprintf(stderr,"Packed to %li bytes.\n",(long)(buf.ptr-buf.buf));
  ec_byte_readinit(&buf,ptr,DATA_SIZE);
  ec_dec_init(&dec,&buf);
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
  nbits2=ec_dec_tell(&dec,4);
  if(nbits!=nbits2){
    fprintf(stderr,
     "Reported number of bits used was %0.2lf, should be %0.2lf.\n",
     ldexp(nbits2,-4),ldexp(nbits,-4));
    ret=-1;
  }
  srand(seed);
  fprintf(stderr,"Testing random streams... Random seed: %u (%.4X)\n", seed, rand() % 65536);
  for(i=0;i<409600;i++){
    unsigned *data;
    int       j;
    int tell_bits;
    int zeros;
    ft=rand()/((RAND_MAX>>(rand()%11))+1)+10;
    sz=rand()/((RAND_MAX>>(rand()%9))+1);
    data=(unsigned *)malloc(sz*sizeof(*data));
    ec_byte_writeinit_buffer(&buf, ptr, DATA_SIZE2);
    ec_enc_init(&enc,&buf);
    zeros = rand()%13==0;
    for(j=0;j<sz;j++){
      if (zeros)
        data[j]=0;
      else
        data[j]=rand()%ft;
      ec_enc_uint(&enc,data[j],ft);
    }
    if (rand()%2==0)
      while(ec_enc_tell(&enc, 0)%8 != 0)
        ec_enc_uint(&enc, rand()%2, 2);
    tell_bits = ec_enc_tell(&enc, 0);
    ec_enc_done(&enc);
    if ((tell_bits+7)/8 < ec_byte_bytes(&buf))
    {
      fprintf (stderr, "tell() lied, there's %li bytes instead of %d (Random seed: %u)\n", 
               ec_byte_bytes(&buf), (tell_bits+7)/8,seed);
      ret=-1;
    }
    tell_bits -= 8*ec_byte_bytes(&buf);
    ec_byte_readinit(&buf,ptr,DATA_SIZE2);
    ec_dec_init(&dec,&buf);
    for(j=0;j<sz;j++){
      sym=ec_dec_uint(&dec,ft);
      if(sym!=data[j]){
        fprintf(stderr,
         "Decoded %i instead of %i with ft of %i at position %i of %i (Random seed: %u).\n",
         sym,data[j],ft,j,sz,seed);
        ret=-1;
      }
    }
    free(data);
  }
  free(ptr);
  return ret;
}
