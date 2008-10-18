#include "celt_types.h"

#if !defined(_entcode_H)
# define _entcode_H (1)
# include <limits.h>
# include "ecintrin.h"



typedef celt_int32_t ec_int32;
typedef celt_uint32_t ec_uint32;
typedef celt_uint64_t ec_uint64;
typedef struct ec_byte_buffer ec_byte_buffer;



/*The number of bits to code at a time when coding bits directly.*/
# define EC_UNIT_BITS  (8)
/*The mask for the given bits.*/
# define EC_UNIT_MASK  ((1U<<EC_UNIT_BITS)-1)



/*Simple libogg1-style buffer.*/
struct ec_byte_buffer{
  unsigned char *buf;
  unsigned char *ptr;
  long           storage;
  int            resizable;
};

/*Encoding functions.*/
void ec_byte_writeinit_buffer(ec_byte_buffer *_b, unsigned char *_buf, long _size);
void ec_byte_writeinit(ec_byte_buffer *_b);
void ec_byte_writetrunc(ec_byte_buffer *_b,long _bytes);
void ec_byte_write1(ec_byte_buffer *_b,unsigned _value);
void ec_byte_write4(ec_byte_buffer *_b,ec_uint32 _value);
void ec_byte_writecopy(ec_byte_buffer *_b,void *_source,long _bytes);
void ec_byte_writeclear(ec_byte_buffer *_b);
/*Decoding functions.*/
void ec_byte_readinit(ec_byte_buffer *_b,unsigned char *_buf,long _bytes);
int ec_byte_look1(ec_byte_buffer *_b);
int ec_byte_look4(ec_byte_buffer *_b,ec_uint32 *_val);
void ec_byte_adv1(ec_byte_buffer *_b);
void ec_byte_adv4(ec_byte_buffer *_b);
int ec_byte_read1(ec_byte_buffer *_b);
int ec_byte_read4(ec_byte_buffer *_b,ec_uint32 *_val);
/*Shared functions.*/
static inline void ec_byte_reset(ec_byte_buffer *_b){
   _b->ptr=_b->buf;
}

static inline long ec_byte_bytes(ec_byte_buffer *_b){
   return _b->ptr-_b->buf;
}

static inline unsigned char *ec_byte_get_buffer(ec_byte_buffer *_b){
   return _b->buf;
}

int ec_ilog(ec_uint32 _v);
int ec_ilog64(ec_uint64 _v);

#endif
