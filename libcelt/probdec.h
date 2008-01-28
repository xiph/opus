#if !defined(_probdec_H)
# define _probdec_H (1)
# include "probmod.h"
# include "entdec.h"



/*Decodes a single symbol using the given probability model and entropy
   decoder.
  Return: The decoded symbol.*/
int ec_probmod_read(ec_probmod *_this,ec_dec *_dec);
/*Decodes a single symbol using the given probability model and entropy
   decoder, restricted to a given subrange of the available symbols.
  This effectively sets the frequency counts of all the symbols outside this
   range to zero, decodes the symbol, then restores the counts to their
   original values, and updates the model.
  _lo: The first legal symbol to decode.
  _hi: One greater than the last legal symbol to decode.
       This must be greater than _lo.
  Return: The decoded symbol.*/
int ec_probmod_read_range(ec_probmod *_this,ec_dec *_dec,int _lo,int _hi);

#endif
