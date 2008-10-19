#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "arch.h"
#include "entenc.h"
#include "mfrngcod.h"



/*A range encoder.
  See rangedec.c and the references for implementation details
   \cite{Mar79,MNW98}.

  @INPROCEEDINGS{Mar79,
   author="Martin, G.N.N.",
   title="Range encoding: an algorithm for removing redundancy from a digitised
    message",
   booktitle="Video \& Data Recording Conference",
   year=1979,
   address="Southampton",
   month=Jul
  }
  @ARTICLE{MNW98,
   author="Alistair Moffat and Radford Neal and Ian H. Witten",
   title="Arithmetic Coding Revisited",
   journal="{ACM} Transactions on Information Systems",
   year=1998,
   volume=16,
   number=3,
   pages="256--294",
   month=Jul,
   URL="http://www.stanford.edu/class/ee398/handouts/papers/Moffat98ArithmCoding.pdf"
  }*/



/*Outputs a symbol, with a carry bit.
  If there is a potential to propagate a carry over several symbols, they are
   buffered until it can be determined whether or not an actual carry will
   occur.
  If the counter for the buffered symbols overflows, then the stream becomes
   undecodable.
  This gives a theoretical limit of a few billion symbols in a single packet on
   32-bit systems.
  The alternative is to truncate the range in order to force a carry, but
   requires similar carry tracking in the decoder, needlessly slowing it down.*/
static void ec_enc_carry_out(ec_enc *_this,int _c){
  if(_c!=EC_SYM_MAX){
    /*No further carry propagation possible, flush buffer.*/
    int carry;
    carry=_c>>EC_SYM_BITS;
    /*Don't output a byte on the first write.
      This compare should be taken care of by branch-prediction thereafter.*/
    if(_this->rem>=0)ec_byte_write1(_this->buf,_this->rem+carry);
    if(_this->ext>0){
      unsigned sym;
      sym=EC_SYM_MAX+carry&EC_SYM_MAX;
      do ec_byte_write1(_this->buf,sym);
      while(--(_this->ext)>0);
    }
    _this->rem=_c&EC_SYM_MAX;
  }
  else _this->ext++;
}

static inline void ec_enc_normalize(ec_enc *_this){
  /*If the range is too small, output some bits and rescale it.*/
  while(_this->rng<=EC_CODE_BOT){
    ec_enc_carry_out(_this,(int)(_this->low>>EC_CODE_SHIFT));
    /*Move the next-to-high-order symbol into the high-order position.*/
    _this->low=_this->low<<EC_SYM_BITS&EC_CODE_TOP-1;
    _this->rng<<=EC_SYM_BITS;
  }
}

void ec_enc_init(ec_enc *_this,ec_byte_buffer *_buf){
  _this->buf=_buf;
  _this->rem=-1;
  _this->ext=0;
  _this->low=0;
  _this->rng=EC_CODE_TOP;
}

void ec_encode(ec_enc *_this,unsigned _fl,unsigned _fh,unsigned _ft){
  ec_uint32 r;
  r=_this->rng/_ft;
  if(_fl>0){
    _this->low+=_this->rng-IMUL32(r,(_ft-_fl));
    _this->rng=IMUL32(r,(_fh-_fl));
  }
  else _this->rng-=IMUL32(r,(_ft-_fh));
  ec_enc_normalize(_this);
}

void ec_encode_bin(ec_enc *_this,unsigned _fl,unsigned _fh,unsigned bits){
   ec_uint32 r, ft;
   r=_this->rng>>bits;
   ft = (ec_uint32)1<<bits;
   if(_fl>0){
     _this->low+=_this->rng-IMUL32(r,(ft-_fl));
     _this->rng=IMUL32(r,(_fh-_fl));
   }
   else _this->rng-=IMUL32(r,(ft-_fh));
   ec_enc_normalize(_this);
}

long ec_enc_tell(ec_enc *_this,int _b){
  ec_uint32 r;
  int       l;
  long      nbits;
  nbits=(ec_byte_bytes(_this->buf)+(_this->rem>=0)+_this->ext)*EC_SYM_BITS;
  /*To handle the non-integral number of bits still left in the encoder state,
     we compute the number of bits of low that must be encoded to ensure that
     the value is inside the range for any possible subsequent bits.
    Note that this is subtly different than the actual value we would end the
     stream with, which tries to make as many of the trailing bits zeros as
     possible.*/
  nbits+=EC_CODE_BITS;
  nbits<<=_b;
  l=EC_ILOG(_this->rng);
  r=_this->rng>>l-16;
  while(_b-->0){
    int b;
    r=r*r>>15;
    b=(int)(r>>16);
    l=l<<1|b;
    r>>=b;
  }
  return nbits-l;
}

void ec_enc_done(ec_enc *_this){
  /*We compute the integer in the current interval that has the largest number
     of trailing zeros, and write that to the stream.
    This is guaranteed to yield the smallest possible encoding.*/
  if(_this->low){
    ec_uint32 end;
    end=EC_CODE_TOP;
    /*Ensure that the end value is in the range.*/
    if(end-_this->low>=_this->rng){
      ec_uint32 msk;
      msk=EC_CODE_TOP-1;
      do{
        msk>>=1;
        end=_this->low+msk&~msk|msk+1;
      }
      while(end-_this->low>=_this->rng);
    }
    /*The remaining output is the next free end.*/
    while(end){
      ec_enc_carry_out(_this,end>>EC_CODE_SHIFT);
      end=end<<EC_SYM_BITS&EC_CODE_TOP-1;
    }
  }
  /*If we have a buffered byte flush it into the output buffer.*/
  if(_this->rem>0||_this->ext>0){
    ec_enc_carry_out(_this,0);
    _this->rem=-1;
  }
}
