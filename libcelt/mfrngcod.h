#if !defined(_mfrngcode_H)
# define _mfrngcode_H (1)
# include "entcode.h"

/*Constants used by the entropy encoder/decoder.*/

/*The number of bits to output at a time.*/
# define EC_SYM_BITS   (8)
/*The total number of bits in each of the state registers.*/
# define EC_CODE_BITS  (32)
/*The maximum symbol value.*/
# define EC_SYM_MAX    ((1U<<EC_SYM_BITS)-1)
/*Bits to shift by to move a symbol into the high-order position.*/
# define EC_CODE_SHIFT (EC_CODE_BITS-EC_SYM_BITS-1)
/*Carry bit of the high-order range symbol.*/
# define EC_CODE_TOP   (((ec_uint32)1U)<<EC_CODE_BITS-1)
/*Low-order bit of the high-order range symbol.*/
# define EC_CODE_BOT   (EC_CODE_TOP>>EC_SYM_BITS)
/*Code for which propagating carries are possible.*/
# define EC_CODE_CARRY (((ec_uint32)EC_SYM_MAX)<<EC_CODE_SHIFT)
/*The number of bits available for the last, partial symbol in the code field.*/
# define EC_CODE_EXTRA ((EC_CODE_BITS-2)%EC_SYM_BITS+1)
/*A mask for the bits available in the coding buffer.
  This allows different platforms to use a variable with more bits, if it is
   convenient.
  We will only use EC_CODE_BITS of it.*/
# define EC_CODE_MASK  ((((ec_uint32)1U)<<EC_CODE_BITS-1)-1<<1|1)


/*The non-zero symbol of the second possible reserved ending.
  This must be the high-bit.*/
# define EC_FOF_RSV1      (1<<EC_SYM_BITS-1)
/*A mask for all the other bits.*/
# define EC_FOF_RSV1_MASK (EC_FOF_RSV1-1)

#endif
