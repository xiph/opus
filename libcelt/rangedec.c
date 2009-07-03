/* (C) 2001-2008 Timothy B. Terriberry
   (C) 2008 Jean-Marc Valin */
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
    sym=_this->rem<<EC_CODE_EXTRA&EC_SYM_MAX;
    /*Read the next value from the input.*/
    _this->rem=ec_dec_in(_this);
    /*Take the rest of the bits we need from this new symbol.*/
    sym|=_this->rem>>EC_SYM_BITS-EC_CODE_EXTRA;
    _this->dif=(_this->dif<<EC_SYM_BITS)-sym&EC_CODE_MASK;
    /*dif can never be larger than EC_CODE_TOP.
      This is equivalent to the slightly more readable:
      if(_this->dif>EC_CODE_TOP)_this->dif-=EC_CODE_TOP;*/
    _this->dif^=_this->dif&_this->dif-1&EC_CODE_TOP;
  }
}

void ec_dec_init(ec_dec *_this,ec_byte_buffer *_buf){
  _this->buf=_buf;
  _this->rem=ec_dec_in(_this);
  _this->rng=1U<<EC_CODE_EXTRA;
  _this->dif=_this->rng-(_this->rem>>EC_SYM_BITS-EC_CODE_EXTRA);
  /*Normalize the interval.*/
  ec_dec_normalize(_this);
}


unsigned ec_decode(ec_dec *_this,unsigned _ft){
  unsigned s;
  _this->nrm=_this->rng/_ft;
  s=(unsigned)((_this->dif-1)/_this->nrm);
  return _ft-EC_MINI(s+1,_ft);
}

unsigned ec_decode_bin(ec_dec *_this,unsigned bits){
   unsigned s;
   ec_uint32 ft;
   ft = (ec_uint32)1<<bits;
   _this->nrm=_this->rng>>bits;
   s=(unsigned)((_this->dif-1)/_this->nrm);
   return ft-EC_MINI(s+1,ft);
}

void ec_dec_update(ec_dec *_this,unsigned _fl,unsigned _fh,unsigned _ft){
  ec_uint32 s;
  s=IMUL32(_this->nrm,(_ft-_fh));
  _this->dif-=s;
  _this->rng=_fl>0?IMUL32(_this->nrm,(_fh-_fl)):_this->rng-s;
  ec_dec_normalize(_this);
}

long ec_dec_tell(ec_dec *_this,int _b){
  ec_uint32 r;
  int       l;
  long      nbits;
  nbits=(ec_byte_bytes(_this->buf)-(EC_CODE_BITS+EC_SYM_BITS-1)/EC_SYM_BITS)*
   EC_SYM_BITS;
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

#if 0
int ec_dec_done(ec_dec *_this){
  unsigned low;
  int      ret;
  /*Check to make sure we've used all the input bytes.
    This ensures that no more ones would ever be inserted into the decoder.*/
  if(_this->buf->ptr-ec_byte_get_buffer(_this->buf)<=
   ec_byte_bytes(_this->buf)){
    return 0;
  }
  /*We compute the smallest finitely odd fraction that fits inside the current
     range, and write that to the stream.
    This is guaranteed to yield the smallest possible encoding.*/
  /*TODO: Fix this line, as it is wrong.
    It doesn't seem worth being able to make this check to do an extra
     subtraction for every symbol decoded.*/
  low=/*What we want: _this->top-_this->rng; What we have:*/_this->dif
  if(low){
    unsigned end;
    end=EC_CODE_TOP;
    /*Ensure that the next free end is in the range.*/
    if(end-low>=_this->rng){
      unsigned msk;
      msk=EC_CODE_TOP-1;
      do{
        msk>>=1;
        end=(low+msk)&~msk|msk+1;
      }
      while(end-low>=_this->rng);
    }
    /*The remaining input should have been the next free end.*/
    return end-low!=_this->dif;
  }
  return 1;
}
#endif
