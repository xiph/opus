/* Copyright (c) 2001-2008 Timothy B. Terriberry
   Copyright (c) 2008-2009 Xiph.Org Foundation */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

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

#include "arch.h"
#include "entdec.h"
#include "mfrngcod.h"



/*A range decoder.
  This is an entropy decoder based upon \cite{Mar79}, which is itself a
   rediscovery of the FIFO arithmetic code introduced by \cite{Pas76}.
  It is very similar to arithmetic encoding, except that encoding is done with
   digits in any base, instead of with bits, and so it is faster when using
   larger bases (i.e.: a byte).
  The author claims an average waste of $\frac{1}{2}\log_b(2b)$ bits, where $b$
   is the base, longer than the theoretical optimum, but to my knowledge there
   is no published justification for this claim.
  This only seems true when using near-infinite precision arithmetic so that
   the process is carried out with no rounding errors.

  IBM (the author's employer) never sought to patent the idea, and to my
   knowledge the algorithm is unencumbered by any patents, though its
   performance is very competitive with proprietary arithmetic coding.
  The two are based on very similar ideas, however.
  An excellent description of implementation details is available at
   http://www.arturocampos.com/ac_range.html
  A recent work \cite{MNW98} which proposes several changes to arithmetic
   encoding for efficiency actually re-discovers many of the principles
   behind range encoding, and presents a good theoretical analysis of them.

  End of stream is handled by writing out the smallest number of bits that
   ensures that the stream will be correctly decoded regardless of the value of
   any subsequent bits.
  ec_dec_tell() can be used to determine how many bits were needed to decode
   all the symbols thus far; other data can be packed in the remaining bits of
   the input buffer.
  @PHDTHESIS{Pas76,
    author="Richard Clark Pasco",
    title="Source coding algorithms for fast data compression",
    school="Dept. of Electrical Engineering, Stanford University",
    address="Stanford, CA",
    month=May,
    year=1976
  }
  @INPROCEEDINGS{Mar79,
   author="Martin, G.N.N.",
   title="Range encoding: an algorithm for removing redundancy from a digitised
    message",
   booktitle="Video & Data Recording Conference",
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


/*Gets the next byte of input.
  After all the bytes in the current packet have been consumed, and the extra
   end code returned if needed, this function will continue to return zero each
   time it is called.
  Return: The next byte of input.*/
static int ec_dec_in(ec_dec *_this){
  int ret;
  ret=ec_byte_read1(_this->buf);
  if(ret<0){
    ret=0;
    /*Needed to keep oc_dec_tell() operating correctly.*/
    ec_byte_adv1(_this->buf);
  }
  return ret;
}

/*Normalizes the contents of dif and rng so that rng lies entirely in the
   high-order symbol.*/
static inline void ec_dec_normalize(ec_dec *_this){
  /*If the range is too small, rescale it and input some bits.*/
  while(_this->rng<=EC_CODE_BOT){
    int sym;
    _this->rng<<=EC_SYM_BITS;
    /*Use up the remaining bits from our last symbol.*/
    sym=_this->rem<<EC_CODE_EXTRA;
    /*Read the next value from the input.*/
    _this->rem=ec_dec_in(_this);
    /*Take the rest of the bits we need from this new symbol.*/
    sym|=_this->rem>>EC_SYM_BITS-EC_CODE_EXTRA;
    /*And subtract them from dif, capped to be less than EC_CODE_TOP.*/
    _this->dif=(_this->dif<<EC_SYM_BITS)+(EC_SYM_MAX&~sym)&EC_CODE_TOP-1;
  }
}

void ec_dec_init(ec_dec *_this,ec_byte_buffer *_buf){
  _this->buf=_buf;
  _this->rem=ec_dec_in(_this);
  _this->rng=1U<<EC_CODE_EXTRA;
  _this->dif=_this->rng-1-(_this->rem>>EC_SYM_BITS-EC_CODE_EXTRA);
  /*Normalize the interval.*/
  ec_dec_normalize(_this);
  _this->end_byte=0; /* Required for platforms that have chars > 8 bits */
  _this->end_bits_left=0;
  _this->nb_end_bits=0;
  _this->error=0;
}


unsigned ec_decode(ec_dec *_this,unsigned _ft){
  unsigned s;
  _this->nrm=_this->rng/_ft;
  s=(unsigned)(_this->dif/_this->nrm);
  return _ft-EC_MINI(s+1,_ft);
}

unsigned ec_decode_bin(ec_dec *_this,unsigned _bits){
   unsigned s;
   _this->nrm=_this->rng>>_bits;
   s=(unsigned)(_this->dif/_this->nrm);
   return (1<<_bits)-EC_MINI(s+1,1<<_bits);
}

unsigned ec_dec_bits(ec_dec *_this,unsigned bits){
  unsigned value=0;
  int count=0;
  _this->nb_end_bits += bits;
  while (bits>=_this->end_bits_left)
  {
    value |= _this->end_byte>>(8-_this->end_bits_left)<<count;
    count += _this->end_bits_left;
    bits -= _this->end_bits_left;
    _this->end_byte=ec_byte_look_at_end(_this->buf);
    _this->end_bits_left = 8;
  }
  value |= ((_this->end_byte>>(8-_this->end_bits_left))&((1<<bits)-1))<<count;
  _this->end_bits_left -= bits;
  return value;
}

void ec_dec_update(ec_dec *_this,unsigned _fl,unsigned _fh,unsigned _ft){
  ec_uint32 s;
  s=IMUL32(_this->nrm,(_ft-_fh));
  _this->dif-=s;
  _this->rng=_fl>0?IMUL32(_this->nrm,(_fh-_fl)):_this->rng-s;
  ec_dec_normalize(_this);
}

/*The probability of having a "one" is given in 1/65536.*/
int ec_dec_bit_prob(ec_dec *_this,unsigned _prob){
  ec_uint32 r;
  ec_uint32 d;
  ec_uint32 s;
  int       val;
  r=_this->rng;
  d=_this->dif;
  s=(r>>16)*_prob;
  val=d<s;
  if(!val)_this->dif=d-s;
  _this->rng=val?s:r-s;
  ec_dec_normalize(_this);
  return val;
}

/*The probability of having a "one" is 1/(1<<_logp).*/
int ec_dec_bit_logp(ec_dec *_this,unsigned _logp){
  ec_uint32 r;
  ec_uint32 d;
  ec_uint32 s;
  int       val;
  r=_this->rng;
  d=_this->dif;
  s=r>>_logp;
  val=d<s;
  if(!val)_this->dif=d-s;
  _this->rng=val?s:r-s;
  ec_dec_normalize(_this);
  return val;
}

int ec_dec_cdf(ec_dec *_this,const int *_cdf,unsigned _ftb){
  ec_uint32 r;
  ec_uint32 d;
  ec_uint32 s;
  ec_uint32 t;
  int       val;
  s=_this->rng;
  d=_this->dif;
  r=s>>_ftb;
  val=0;
  do{
    t=s;
    s=IMUL32(r,(1<<_ftb)-_cdf[++val]);
  }
  while(d<s);
  _this->dif=d-s;
  _this->rng=t-s;
  ec_dec_normalize(_this);
  return val-1;
}

ec_uint32 ec_dec_tell(ec_dec *_this,int _b){
  ec_uint32 r;
  int       l;
  ec_uint32      nbits;
  nbits=(ec_byte_bytes(_this->buf)-(EC_CODE_BITS+EC_SYM_BITS-1)/EC_SYM_BITS)*
   EC_SYM_BITS;
  /*To handle the non-integral number of bits still left in the decoder state,
     we compute the number of bits of low that must be encoded to ensure that
     the value is inside the range for any possible subsequent bits.*/
  nbits+=EC_CODE_BITS+1+_this->nb_end_bits;
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
