#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include "celt_types.h"
#include "bands.h"
#include <math.h>
#define MAX_SIZE 100

int ret=0;
void test_rotation(int N, int K)
{
   int i;
   double err = 0, ener = 0, snr;
   celt_word16_t theta = Q15_ONE*.007*N/K;
   celt_word16_t x0[MAX_SIZE];
   celt_word16_t x1[MAX_SIZE];
   for (i=0;i<N;i++)
      x1[i] = x0[i] = rand()%32767-16384;
   exp_rotation(x1, N, theta, 1, 1, 8);
   exp_rotation(x1, N, theta, -1, 1, 8);
   for (i=0;i<N;i++)
   {
      err += (x0[i]-(double)x1[i])*(x0[i]-(double)x1[i]);
      ener += x0[i]*(double)x0[i];
   }
   snr = 20*log10(ener/err);
   printf ("SNR for size %d (%d pulses) is %f\n", N, K, snr);
   if (snr < 60)
      ret = 1;
}

int main()
{
   test_rotation(4, 40);
   test_rotation(7, 20);
   test_rotation(10, 10);
   test_rotation(23, 5);
   test_rotation(50, 3);
   return ret;
}
