#if !defined(_bitrenc_H)
# define _bitrenc_H (1)
# include "bitree.h"

/*Encoder-specific functions for Binary Indexed Trees.
  See bitree.h for more detailed documentation.*/

/*Updates the frequency of a given symbol.
  _sz:  The size of the table.
  _sym: The symbol to update.
  _val: The amount to add to this symbol's frequency.*/
void ec_bitree_update(unsigned *_this,int _sz,int _sym,int _val);

#endif
