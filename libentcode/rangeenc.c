#include <stddef.h>
#include "entenc.h"
#include "mfrngcod.h"



/*A multiply-free range encoder.
  See mfrngdec.c and the references for implementation details
   \cite{Mar79,MNW98,SM98}.

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
   URL="http://dev.acm.org/pubs/citations/journals/tois/1998-16-3/p256-moffat/"
  }
  @INPROCEEDINGS{SM98,
   author="Lang Stuiver and Alistair Moffat",
   title="Piecewise Integer Mapping for Arithmetic Coding",
   booktitle="Proceedings of the {IEEE} Data Compression Conference",
   pages="1--10",
   address="Snowbird, UT",
   month="Mar./Apr.",
   year=1998
  }*/



/*Outputs a symbol, with a carry bit.
  If there is a potential to propogate a carry over several symbols, they are
   buffered until it can be determined whether or not an actual carry will
   occur.
  If the counter for the buffered symbols overflows, then the range is
   truncated to force a carry to occur, towards whichever side maximizes the
   remaining range.*/
static void ec_enc_carry_out(ec_enc *_this,int _c){
  if(_c!=EC_SYM_MAX){
    /*No further carry propogation possible, flush buffer.*/
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

static void ec_enc_normalize(ec_enc *_this){
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
  unsigned r;
  unsigned s;
  r=_this->rng/_ft;
  if(_fl>0){
    s=r*(_ft-_fl);
    _this->low+=_this->rng-s;
    _this->rng=r*(_fh-_fl);
  }
  else _this->rng-=r*(_ft-_fh);
  ec_enc_normalize(_this);
}

void ec_enc_done(ec_enc *_this){
  /*We compute the integer in the current interval that has the largest number
     of trailing zeros, and write that to the stream.
    This is guaranteed to yield the smallest possible encoding.*/
  if(_this->low){
    unsigned end;
    end=EC_CODE_TOP;
    /*Ensure that the end value is in the range.*/
    if(end-_this->low>=_this->rng){
      unsigned msk;
      msk=EC_CODE_TOP-1;
      do{
        msk>>=1;
        end=(_this->low+msk)&~msk|msk+1;
      }
      while(end-_this->low>=_this->rng);
    }
    /*The remaining output is the next free end.*/
    while(end){
      ec_enc_carry_out(_this,end>>EC_CODE_SHIFT);
      end=end<<EC_SYM_BITS&EC_CODE_TOP-1;
    }
  }
  /*If we have a buffered byte...*/
  if(_this->rem>=0){
    unsigned char *p;
    unsigned char *buf;
    /*Flush it into the output buffer.*/
    ec_enc_carry_out(_this,0);
    /*We may be able to drop some redundant bytes from the end.*/
    buf=ec_byte_get_buffer(_this->buf);
    p=buf+ec_byte_bytes(_this->buf)-1;
    /*Strip trailing zeros.*/
    while(p>=buf&&!p[0])p--;
    /*Strip one trailing EC_FOF_RSV1 byte if the buffer ends in a string of
       consecutive EC_FOF_RSV1 bytes preceded by one (or more) zeros.*/
    if(p>buf&&p[0]==EC_FOF_RSV1){
      unsigned char *q;
      q=p;
      do q--;
      while(q>buf&&q[0]==EC_FOF_RSV1);
      if(!q[0])p--;
    }
    ec_byte_writetrunc(_this->buf,p+1-buf);
  }
}
