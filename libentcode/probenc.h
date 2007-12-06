#if !defined(_probenc_H)
# define _probenc_H (1)
# include "probmod.h"
# include "entenc.h"

/*Encodes a single symbol using the given probability model and entropy
   encoder.
  _sym: The symbol to encode.*/
void ec_probmod_write(ec_probmod *_this,ec_enc *_enc,int _sym);
/*Encodes a single symbol using the given probability model and entropy
   encoder, restricted to a given subrange of the available symbols.
  This effectively sets the frequency counts of all the symbols outside this
   range to zero, encodes the symbol, then restores the counts to their
   original values, and updates the models.
  _sym: The symbol to encode.
        The caller must ensure this falls in the range _lo<=_sym<_hi.
  _lo:  The first legal symbol to encode.
  _hi:  One greater than the last legal symbol to encode.
        This must be greater than _lo.*/
void ec_probmod_write_range(ec_probmod *_this,ec_enc *_enc,int _sym,
 int _lo,int _hi);

#endif
