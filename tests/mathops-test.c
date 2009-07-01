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

#ifdef FIXED_DEBUG  
long long celt_mips=0;
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

#ifndef FIXED_POINT
void testlog2(void)
{
   float x;
   for (x=0.001;x<1677700.0;x+=(x/8.0))
   {
      float error = fabs((1.442695040888963387*log(x))-celt_log2(x));
      if (error>0.001)
      {
         fprintf (stderr, "celt_log2 failed: fabs((1.442695040888963387*log(x))-celt_log2(x))>0.001 (x = %f, error = %f)\n", x,error);
         ret = 1;    
      }
   }
}

void testexp2(void)
{
   float x;
   for (x=-11.0;x<24.0;x+=0.0007)
   {
      float error = fabs(x-(1.442695040888963387*log(celt_exp2(x))));
      if (error>0.0005)
      {
         fprintf (stderr, "celt_exp2 failed: fabs(x-(1.442695040888963387*log(celt_exp2(x))))>0.0005 (x = %f, error = %f)\n", x,error);
         ret = 1;    
      }
   }
}

void testexp2log2(void)
{
   float x;
   for (x=-11.0;x<24.0;x+=0.0007)
   {
      float error = fabs(x-(celt_log2(celt_exp2(x))));
      if (error>0.001)
      {
         fprintf (stderr, "celt_log2/celt_exp2 failed: fabs(x-(celt_log2(celt_exp2(x))))>0.001 (x = %f, error = %f)\n", x,error);
         ret = 1;    
      }
   }
}
#else
void testilog2(void)
{
   celt_word32_t x;
   for (x=1;x<=268435455;x+=127)
   {
      celt_word32_t error = abs(celt_ilog2(x)-(int)floor(log2(x)));
      if (error!=0)
      {
         printf("celt_ilog2 failed: celt_ilog2(x)!=floor(log2(x)) (x = %d, error = %d)\n",x,error);
         ret = 1;
      }
   }
}
#endif

int main(void)
{
   testdiv();
   testsqrt();
   testrsqrt();
#ifndef FIXED_POINT
   testlog2();
   testexp2();
   testexp2log2();
#else
   testilog2();
#endif
   return ret;
}
