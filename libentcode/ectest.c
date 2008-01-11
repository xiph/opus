#include <stdio.h>
#include <math.h>
#include "probenc.h"
#include "probdec.h"
#include "bitrenc.h"

int main(int _argc,char **_argv){
  ec_byte_buffer buf;
  ec_enc         enc;
  ec_dec         dec;
  ec_probmod     mod;
  ec_uint64      sym64;
  long           nbits;
  double         entropy;
  int            ft;
  int            ftb;
  int            sym;
  int            sz;
  int            s;
  int            i;
  entropy=0;
  /*Testing encoding of raw bit values.*/
  ec_byte_writeinit(&buf);
  ec_enc_init(&enc,&buf);
  for(ft=0;ft<1024;ft++){
    for(i=0;i<ft;i++){
      entropy+=log(ft)*M_LOG2E;
      ec_enc_uint(&enc,i,ft);
      entropy+=log(ft)*M_LOG2E+30;
      ec_enc_uint64(&enc,(ec_uint64)i<<30|i,(ec_uint64)ft<<30);
    }
  }
  /*Testing encoding of raw bit values.*/
  for(ftb=0;ftb<16;ftb++){
    for(i=0;i<(1<<ftb);i++){
      long nbits;
      long nbits2;
      entropy+=ftb;
      nbits=ec_enc_tell(&enc);
      ec_enc_bits(&enc,i,ftb);
      nbits2=ec_enc_tell(&enc);
      if(nbits2-nbits!=ftb){
        fprintf(stderr,"Used %li bits to encode %i bits directly.\n",
         nbits2-nbits,ftb);
      }
      entropy+=ftb+30;
      nbits=nbits2;
      ec_enc_bits64(&enc,(ec_uint64)i<<30|i,ftb+30);
      nbits2=ec_enc_tell(&enc);
      if(nbits2-nbits!=ftb+30){
        fprintf(stderr,"Used %li bits to encode %i bits directly.\n",
         nbits2-nbits,ftb+30);
      }
    }
  }
  for(sz=1;sz<256;sz++){
    ec_probmod_init_full(&mod,sz,1,sz+(sz>>1),NULL);
    for(i=0;i<sz;i++){
      s=((unsigned)(i*45678901+7))%sz;
      entropy+=(log(mod.ft)-log(ec_bitree_get_freq(mod.bitree,s)))*M_LOG2E;
      ec_probmod_write(&mod,&enc,s);
    }
    ec_probmod_clear(&mod);
  }
  for(sz=11;sz<256;sz++){
    ec_probmod_init_full(&mod,sz,1,sz+(sz>>1),NULL);
    for(i=0;i<sz;i++){
      s=((unsigned)(i*45678901+7))%sz;
      entropy+=(log(ec_bitree_get_cumul(mod.bitree,EC_MINI(s+6,sz))-
       ec_bitree_get_cumul(mod.bitree,EC_MAXI(s-5,0)))-
       log(ec_bitree_get_freq(mod.bitree,s)))*M_LOG2E;
      ec_probmod_write_range(&mod,&enc,s,EC_MAXI(s-5,0),EC_MINI(s+6,sz));
    }
    ec_probmod_clear(&mod);
  }
  nbits=ec_enc_tell(&enc);
  ec_enc_done(&enc);
  fprintf(stderr,
   "Encoded %0.2lf bits of entropy to %li bits (%0.3lf%% wasted).\n",
   entropy,nbits,100*(nbits-entropy)/nbits);
  fprintf(stderr,"Packed to %li bytes.\n",(long)(buf.ptr-buf.buf));
  ec_byte_readinit(&buf,ec_byte_get_buffer(&buf),ec_byte_bytes(&buf));
  ec_dec_init(&dec,&buf);
  for(ft=0;ft<1024;ft++){
    for(i=0;i<ft;i++){
      sym=ec_dec_uint(&dec,ft);
      if(sym!=i){
        fprintf(stderr,"Decoded %i instead of %i with ft of %i.\n",sym,i,ft);
        return -1;
      }
      sym64=ec_dec_uint64(&dec,(ec_uint64)ft<<30);
      if(sym64!=((ec_uint64)i<<30|i)){
        fprintf(stderr,"Decoded %lli instead of %lli with ft of %lli.\n",sym64,
         (ec_uint64)i<<30|i,(ec_uint64)ft<<30);
      }
    }
  }
  for(ftb=0;ftb<16;ftb++){
    for(i=0;i<(1<<ftb);i++){
      sym=ec_dec_bits(&dec,ftb);
      if(sym!=i){
        fprintf(stderr,"Decoded %i instead of %i with ftb of %i.\n",sym,i,ftb);
        return -1;
      }
      sym64=ec_dec_bits64(&dec,ftb+30);
      if(sym64!=((ec_uint64)i<<30|i)){
        fprintf(stderr,"Decoded %lli instead of %lli with ftb of %i.\n",
         sym64,(ec_uint64)i<<30|i,ftb+30);
      }
    }
  }
  for(sz=1;sz<256;sz++){
    ec_probmod_init_full(&mod,sz,1,sz+(sz>>1),NULL);
    for(i=0;i<sz;i++){
      s=((unsigned)(i*45678901+7))%sz;
      sym=ec_probmod_read(&mod,&dec);
      if(sym!=s){
        fprintf(stderr,"Decoded %i instead of %i with sz of %i.\n",sym,s,sz);
        return -1;
      }
    }
    ec_probmod_clear(&mod);
  }
  for(sz=11;sz<256;sz++){
    ec_probmod_init_full(&mod,sz,1,sz+(sz>>1),NULL);
    for(i=0;i<sz;i++){
      s=((unsigned)(i*45678901+7))%sz;
      sym=ec_probmod_read_range(&mod,&dec,EC_MAXI(s-5,0),EC_MINI(s+6,sz));
      if(sym!=s){
        fprintf(stderr,"Decoded %i instead of %i with sz of %i.\n",sym,s,sz);
        return -1;
      }
    }
    ec_probmod_clear(&mod);
  }
  ec_byte_writeclear(&buf);
  fprintf(stderr,"All tests passed.\n");
  return 0;
}
