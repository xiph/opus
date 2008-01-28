#if !defined(_bitrenc_H)
# define _bitredec_H (1)
# include "bitree.h"

/*Decoder-specific functions for Binary Indexed Trees.
  See bitree.h for more detailed documentation.*/

/*Gets the symbol that corresponds with a given frequency.
  This is an omnibus function that also computes the cumulative frequency of
   the symbols before the one returned, and updates the count of that symbol by
   the given amount.
  _sz:    The size of the table.
  _split: The largest power of two less than OR EQUAL to the table size.
  _freq:  A frequency in the range of one of the symbols in the alphabet.
  _fl:    Returns the sum of the frequencies of the symbols less than that of
           the returned symbol.
  _val:   The amount to add to returned symbol's frequency.
  Return: The smallest symbol whose cumulative frequency is greater than freq.*/
int ec_bitree_find_and_update(unsigned *_this,int _sz,int _split,
 unsigned _freq,unsigned *_fl,int _val);

#endif
