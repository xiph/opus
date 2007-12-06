#if !defined(_probmod_H)
# define _probmod_H (1)
# include "entcode.h"

typedef struct ec_probsamp     ec_probsamp;
typedef struct ec_probmod      ec_probmod;

/*A sample from a probability distribution.
  This is the information needed to encode a symbol or update the decoder
   state.*/
struct ec_probsamp{
  /*The cumulative frequency of all symbols preceding this one in the
     alphabet.*/
  unsigned fl;
  /*The frequency of the symbol coded.*/
  unsigned fs;
  /*The total frequency of all symbols in the alphabet.*/
  unsigned ft;
};


/*A simple frequency-count probability model.*/
struct ec_probmod{
  /*The number of symbols in this context.*/
  int       sz;
  /*The largest power of two less than or equal to sz.*/
  int       split;
  /*The amount by which to increment the frequency count of an observed
     symbol.*/
  unsigned  inc;
  /*The current total frequency count.*/
  unsigned  ft;
  /*The maximum total frequency count allowed before the counts are rescaled.
    Note that this should be larger than (inc+1>>1)+sz-1, since at most one
     rescaling is done per decoded symbol.
    Otherwise, this threshold might be exceeded.
    This must be less than 2**23 for a range coder, and 2**31 for an
     arithmetic coder.*/
  unsigned  thresh;
  /*The binary indexed tree used to keep track of the frequency counts.*/
  unsigned *bitree;
};


/*Initializes a probability model of the given size.
  The amount to increment and all frequency counts are initialized to 1.
  The rescaling threshold is initialized to 2**23.
  _sz: The number of symbols in this context.*/
void ec_probmod_init(ec_probmod *_this,unsigned _sz);
/*Initializes a probability model of the given size.
  The amount to increment is initialized to 1.
  The rescaling threshold is initialized to 2**23.
  _sz:     The number of symbols in this context.
  _counts: The initial frequency count of each symbol.*/
void ec_probmod_init_from_counts(ec_probmod *_this,unsigned _sz,
 const unsigned *_counts);
/*Initializes a probability model of the given size.
  _sz:     The number of symbols in this context.
  _inc:    The amount by which to increment the frequency count of an observed
            symbol.
  _thresh: The maximum total frequency count allowed before the counts are
            rescaled.
           See above for restrictions on this value.
  _counts: The initial frequency count of each symbol, or NULL to initialize
            each frequency count to 1.*/
void ec_probmod_init_full(ec_probmod *_this,unsigned _sz,unsigned _inc,
 unsigned _thresh,const unsigned *_counts);
/*Frees all memory used by this probability model.*/
void ec_probmod_clear(ec_probmod *_this);



#endif
