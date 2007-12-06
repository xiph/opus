#if !defined(_entenc_H)
# define _entenc_H (1)
# include <stddef.h>
# include "entcode.h"



typedef struct ec_enc ec_enc;



/*The entropy encoder.*/
struct ec_enc{
   /*Buffered output.*/
   ec_byte_buffer *buf;
   /*A buffered output symbol, awaiting carry propagation.*/
   int             rem;
   /*Number of extra carry propogating symbols.*/
   size_t          ext;
   /*The number of values in the current range.*/
   ec_uint32       rng;
   /*The low end of the current range (inclusive).*/
   ec_uint32       low;
};


/*Initializes the encoder.
  _buf: The buffer to store output bytes in.
        This must have already been initialized for writing and reset.*/
void ec_enc_init(ec_enc *_this,ec_byte_buffer *_buf);
/*Encodes a symbol given its frequency information.
  The frequency information must be discernable by the decoder, assuming it
   has read only the previous symbols from the stream.
  It is allowable to change the frequency information, or even the entire
   source alphabet, so long as the decoder can tell from the context of the
   previously encoded information that it is supposed to do so as well.
  _fl: The sum of the frequencies of symbols before the one to be encoded.
  _fs: The frequency of the symbol to be encoded.
  _ft: The sum of the frequencies of all the symbols*/
void ec_encode(ec_enc *_this,unsigned _fl,unsigned _fs,unsigned _ft);
/*Encodes a sequence of raw bits in the stream.
  _fl:  The bits to encode.
  _ftb: The number of bits to encode.
        This must be at least one, and no more than 32.*/
void ec_enc_bits(ec_enc *_this,ec_uint32 _fl,int _ftb);
/*Encodes a raw unsigned integer in the stream.
  _fl: The integer to encode.
  _ft: The number of integers that can be encoded (one more than the max).
       This must be at least one, and no more than 2**32-1.*/
void ec_enc_uint(ec_enc *_this,ec_uint32 _fl,ec_uint32 _ft);

/*Indicates that there are no more symbols to encode.
  All reamining output bytes are flushed to the output buffer.
  ec_enc_init() must be called before the encoder can be used again.*/
void ec_enc_done(ec_enc *_this);

#endif
