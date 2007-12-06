#include <stdio.h>
#include "probenc.h"
#include "probdec.h"

int main(int _argc,char **_argv){
  ec_byte_buffer buf;
  ec_enc         enc;
  ec_dec         dec;
  ec_probmod     mod;
  int            ft;
  int            ftb;
  int            sym;
  int            sz;
  int            s;
  int            i;
  /*Testing encoding of raw bit values.*/
  ec_byte_writeinit(&buf);
  ec_enc_init(&enc,&buf);
  for(ft=0;ft<1024;ft++){
    for(i=0;i<ft;i++){
      ec_enc_uint(&enc,i,ft);
    }
  }
  /*Testing encoding of raw bit values.*/
  for(ftb=0;ftb<16;ftb++){
    for(i=0;i<(1<<ftb);i++){
      ec_enc_bits(&enc,i,ftb);
    }
  }
  for(sz=1;sz<256;sz++){
    ec_probmod_init_full(&mod,sz,1,sz+(sz>>1),NULL);
    for(i=0;i<sz;i++){
      s=((unsigned)(i*45678901+7))%sz;
      ec_probmod_write(&mod,&enc,s);
    }
    ec_probmod_clear(&mod);
  }
  for(sz=11;sz<256;sz++){
    ec_probmod_init_full(&mod,sz,1,sz+(sz>>1),NULL);
    for(i=0;i<sz;i++){
      s=((unsigned)(i*45678901+7))%sz;
      ec_probmod_write_range(&mod,&enc,s,EC_MAXI(s-5,0),EC_MINI(s+6,sz));
    }
    ec_probmod_clear(&mod);
  }
  ec_enc_done(&enc);
  fprintf(stderr,"Encoded to %li bytes.\n",(long)(buf.ptr-buf.buf));
  ec_byte_readinit(&buf,ec_byte_get_buffer(&buf),ec_byte_bytes(&buf));
  ec_dec_init(&dec,&buf);
  for(ft=0;ft<1024;ft++){
    for(i=0;i<ft;i++){
      sym=ec_dec_uint(&dec,ft);
      if(sym!=i){
        fprintf(stderr,"Decoded %i instead of %i with ft of %i.\n",sym,i,ft);
        return -1;
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
