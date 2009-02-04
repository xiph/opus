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

#if !defined(_entdec_H)
# define _entdec_H (1)
# include "entcode.h"



typedef struct ec_dec ec_dec;



/*The entropy decoder.*/
struct ec_dec{
   /*The buffer to decode.*/
   ec_byte_buffer *buf;
   /*The remainder of a buffered input symbol.*/
   int             rem;
   /*The number of values in the current range.*/
   ec_uint32       rng;
   /*The difference between the input value and the lowest value in the current
      range.*/
   ec_uint32       dif;
   /*Normalization factor.*/
   ec_uint32       nrm;
};


/*Initializes the decoder.
  _buf: The input buffer to use.
  Return: 0 on success, or a negative value on error.*/
void ec_dec_init(ec_dec *_this,ec_byte_buffer *_buf);
/*Calculates the cumulative frequency for the next symbol.
  This can then be fed into the probability model to determine what that
   symbol is, and the additional frequency information required to advance to
   the next symbol.
  This function cannot be called more than once without a corresponding call to
   ec_dec_update(), or decoding will not proceed correctly.
  _ft: The total frequency of the symbols in the alphabet the next symbol was
        encoded with.
  Return: A cumulative frequency representing the encoded symbol.
          If the cumulative frequency of all the symbols before the one that
           was encoded was fl, and the cumulative frequency of all the symbols
           up to and including the one encoded is fh, then the returned value
           will fall in the range [fl,fh).*/
unsigned ec_decode(ec_dec *_this,unsigned _ft);
unsigned ec_decode_bin(ec_dec *_this,unsigned bits);
/*Advance the decoder past the next symbol using the frequency information the
   symbol was encoded with.
  Exactly one call to ec_decode() must have been made so that all necessary
   intermediate calculations are performed.
  _fl:  The cumulative frequency of all symbols that come before the symbol
         decoded.
  _fh:  The cumulative frequency of all symbols up to and including the symbol
         decoded.
        Together with _fl, this defines the range [_fl,_fh) in which the value
         returned above must fall.
  _ft:  The total frequency of the symbols in the alphabet the symbol decoded
         was encoded in.
        This must be the same as passed to the preceding call to ec_decode().*/
void ec_dec_update(ec_dec *_this,unsigned _fl,unsigned _fh,
 unsigned _ft);
/*Extracts a sequence of raw bits from the stream.
  The bits must have been encoded with ec_enc_bits().
  No call to ec_dec_update() is necessary after this call.
  _ftb: The number of bits to extract.
        This must be at least one, and no more than 32.
  Return: The decoded bits.*/
ec_uint32 ec_dec_bits(ec_dec *_this,int _ftb);
/*Extracts a sequence of raw bits from the stream.
  The bits must have been encoded with ec_enc_bits64().
  No call to ec_dec_update() is necessary after this call.
  _ftb: The number of bits to extract.
        This must be at least one, and no more than 64.
  Return: The decoded bits.*/
ec_uint64 ec_dec_bits64(ec_dec *_this,int _ftb);
/*Extracts a raw unsigned integer with a non-power-of-2 range from the stream.
  The bits must have been encoded with ec_enc_uint().
  No call to ec_dec_update() is necessary after this call.
  _ft: The number of integers that can be decoded (one more than the max).
       This must be at least one, and no more than 2**32-1.
  Return: The decoded bits.*/
ec_uint32 ec_dec_uint(ec_dec *_this,ec_uint32 _ft);
/*Extracts a raw unsigned integer with a non-power-of-2 range from the stream.
  The bits must have been encoded with ec_enc_uint64().
  No call to ec_dec_update() is necessary after this call.
  _ft: The number of integers that can be decoded (one more than the max).
       This must be at least one, and no more than 2**64-1.
  Return: The decoded bits.*/
ec_uint64 ec_dec_uint64(ec_dec *_this,ec_uint64 _ft);

/*Returns the number of bits "used" by the decoded symbols so far.
  The actual number of bits may be larger, due to rounding to whole bytes, or
   smaller, due to trailing zeros that were be stripped, so this is not an
   estimate of the true packet size.
  This same number can be computed by the encoder, and is suitable for making
   coding decisions.
  _b: The number of extra bits of precision to include.
      At most 16 will be accurate.
  Return: The number of bits scaled by 2**_b.
          This will always be slightly larger than the exact value (e.g., all
           rounding error is in the positive direction).*/
long ec_dec_tell(ec_dec *_this,int _b);

#endif
