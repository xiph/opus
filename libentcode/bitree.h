/*Implements Binary Indexed Trees for cumulative probability tables, based
   upon a combination of the techniques described in \cite{Fen93,Fen95,Mof99}.
  This is a really, amazingly elegant data structure, that maintains
   cumulative frequency tables in logarithmic time, using exactly the same
   space as an ordinary frequency table.
  In addition, the non-cumulative frequency can be retrieved in constant
   amortized time (under 2 memory references per symbol on average).

  We are dealing primarily with relatively small alphabets and are not sorting
   symbols by frequency, and so we adopt Fenwick's tree organization strategy.
  It's complexity has better constant factors, and although it is logarithmic
   in n (the alphabet size) instead of s (the index of the symbol), the latter
   is not expected to be appreciably smaller.
  We modify it however, to remove the special cases surrounding the element 0,
   which greatly streamlines the code.
  Our scheme has the added benefit that for alphabet sizes that are a power of
   2, the last element of the array is the total cumulative frequency count.

  We choose Moffat's approach to halving the entire frequency table, which is
   over twice as fast in practice as that suggested by Fenwick, even though
   they have the same number of memory accesses.
  We also adapt Moffat's suggestion for an omnibus decode function that updates
   the count of the symbol being decoded while it is searching for it.
  We also have it retain and return the cumulative frequency count needed to
   update the arithmetic decoder.

  See bitrenc.h and bitrdec.h for encoding- and decoding-specific functions.

  @TECHREPORT{Fen93,
   author     ="Peter Fenwick",
   title      ="A new data structure for cumulative probability tables",
   institution="The University of Auckland, Department of Computer Science",
   year       =1993,
   number     =88,
   month      =May,
   URL        ="http://www.cs.auckland.ac.nz/~peter-f/ftplink/TechRep88.ps"
  }
  @TECHREPORT{Fen95,
   author     ="Peter Fenwick",
   title      ="A new data structure for cumulative probability tables: an
                improved frequency to symbol algorithm",
   institution="The University of Auckland, Department of Computer Science",
   year       =1995,
   number     =110,
   month      =Feb,
   URL        ="http://www.cs.auckland.ac.nz/~peter-f/ftplink/TechRep110.ps"
  }
  @ARTICLE{Mof99,
    author    ="Alistair Moffat",
    title     ="An improved data structure for cumulative probability tables",
    journal   ="Software Practice and Experience",
    volume    =29,
    number    =7,
    pages     ="647--659",
    year      =1999
  }*/
#if !defined(_bitree_H)
# define _bitree_H (1)

/*Converts an array of frequency counts to our cumulative tree representation.
  _sz: The size of the table.*/
void ec_bitree_from_counts(unsigned *_this,int _sz);

/*Converts our cumulative tree representation to an array of frequency counts.
  _sz:    The size of the table.
  _split: The largest power of two less than OR EQUAL to the table size.*/
void ec_bitree_to_counts(unsigned *_this,int _sz,int _split);

/*Gets the cumulative frequency of the symbols less than the given one.
  _sym: The symbol to obtain the cumulative frequency for.
  Return: The sum of the frequencies of the symbols less than _sym.*/
unsigned ec_bitree_get_cumul(const unsigned *_this,int _sym);

/*Gets the frequency of a single symbol.
  _sym: The symbol to obtain the frequency for.
  Return: The frequency of _sym.*/
unsigned ec_bitree_get_freq(const unsigned *_this,int _sym);

/*Halves the frequency of each symbol, rounding up.
  _sz: The size of the table.
  _split: The largest power of two less than OR EQUAL to the table size.*/
void ec_bitree_halve(unsigned *_this,int _sz,int _split);

#endif
