#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mathops.h"
#include <stdio.h>
#include <math.h>

#ifdef FIXED_POINT
#define WORD "%d"
#else
#define WORD "%f"
#endif

int ret = 0;

void testdiv(void)
{
   celt_int32_t i;
   for (i=1;i<=327670;i++)
   {
      double prod;
      celt_word32_t val;
      val = celt_rcp(i);
#ifdef FIXED_POINT
      prod = (1./32768./65526.)*val*i;
#else
      prod = val*i;
#endif
      if (fabs(prod-1) > .001)
      {
         fprintf (stderr, "div failed: 1/%d="WORD" (product = %f)\n", i, val, prod);
         ret = 1;
      }
   }
}

void testsqrt(void)
{
   celt_int32_t i;
   for (i=1;i<=1000000000;i++)
   {
      double ratio;
      celt_word16_t val;
      val = celt_sqrt(i);
      ratio = val/sqrt(i);
      if (fabs(ratio - 1) > .001 && fabs(val-sqrt(i)) > 2)
      {
         fprintf (stderr, "sqrt failed: sqrt(%d)="WORD" (ratio = %f)\n", i, val, ratio);
         ret = 1;
      }
      i+= i>>10;
   }
}

void testrsqrt(void)
{
   celt_int32_t i;
   for (i=1;i<=2000000;i++)
   {
      double ratio;
      celt_word16_t val;
      val = celt_rsqrt(i);
      ratio = val*sqrt(i)/Q15ONE;
      if (fabs(ratio - 1) > .05)
      {
         fprintf (stderr, "rsqrt failed: rsqrt(%d)="WORD" (ratio = %f)\n", i, val, ratio);
         ret = 1;
      }
      i+= i>>10;
   }
}

int main(void)
{
   testdiv();
   testsqrt();
   testrsqrt();
   return ret;
}
