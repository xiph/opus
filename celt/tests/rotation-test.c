#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef CUSTOM_MODES
#define CUSTOM_MODES
#endif

#define CELT_C

#include <stdio.h>
#include <stdlib.h>
#include "vq.c"
#include "cwrs.c"
#include "entcode.c"
#include "entenc.c"
#include "entdec.c"
#include "mathops.c"
#include "bands.h"
#include <math.h>
#define MAX_SIZE 100

int ret=0;
void test_rotation(int N, int K)
{
   int i;
   double err = 0, ener = 0, snr, snr0;
   opus_val16 x0[MAX_SIZE];
   opus_val16 x1[MAX_SIZE];
   for (i=0;i<N;i++)
      x1[i] = x0[i] = rand()%32767-16384;
   exp_rotation(x1, N, 1, 1, K, SPREAD_NORMAL);
   for (i=0;i<N;i++)
   {
      err += (x0[i]-(double)x1[i])*(x0[i]-(double)x1[i]);
      ener += x0[i]*(double)x0[i];
   }
   snr0 = 20*log10(ener/err);
   err = ener = 0;
   exp_rotation(x1, N, -1, 1, K, SPREAD_NORMAL);
   for (i=0;i<N;i++)
   {
      err += (x0[i]-(double)x1[i])*(x0[i]-(double)x1[i]);
      ener += x0[i]*(double)x0[i];
   }
   snr = 20*log10(ener/err);
   printf ("SNR for size %d (%d pulses) is %f (was %f without inverse)\n", N, K, snr, snr0);
   if (snr < 60 || snr0 > 20)
   {
      fprintf(stderr, "FAIL!\n");
      ret = 1;
   }
}

int main(void)
{
   ALLOC_STACK;
   test_rotation(15, 3);
   test_rotation(23, 5);
   test_rotation(50, 3);
   test_rotation(80, 1);
   return ret;
}
