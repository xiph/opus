#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "opus_types.h"
#include <stdio.h>

int main(void)
{
   opus_int16 i = 1;
   i <<= 14;
   if (i>>14 != 1)
   {
      fprintf(stderr, "opus_int16 isn't 16 bits\n");
      return 1;
   }
   if (sizeof(opus_int16)*2 != sizeof(opus_int32))
   {
      fprintf(stderr, "16*2 != 32\n");
      return 1;
   }
   return 0;
}
