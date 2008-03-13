#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "celt_types.h"
#include <stdio.h>

int main(void)
{
   celt_int16_t i = 1;
   i <<= 14;
   if (i>>14 != 1)
   {
      fprintf(stderr, "celt_int16_t isn't 16 bits\n");
      return 1;
   }
   if (sizeof(celt_int16_t)*2 != sizeof(celt_int32_t))
   {
      fprintf(stderr, "16*2 != 32\n");
      return 1;
   }
   if (sizeof(celt_int32_t)*2 != sizeof(celt_int64_t))
   {
      fprintf(stderr, "32*2 != 64\n");
      return 1;
   }
   return 0;
}
