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
   double err = 0, ener = 0, snr, snr0;
   celt_word16 x0[MAX_SIZE];
   celt_word16 x1[MAX_SIZE];
   int nb_rotations = (N+4*K)/(8*K);
   for (i=0;i<N;i++)
      x1[i] = x0[i] = rand()%32767-16384;
   exp_rotation(x1, N, 1, 1, nb_rotations);
   for (i=0;i<N;i++)
   {
      err += (x0[i]-(double)x1[i])*(x0[i]-(double)x1[i]);
      ener += x0[i]*(double)x0[i];
   }
   snr0 = 20*log10(ener/err);
   err = ener = 0;
   exp_rotation(x1, N, -1, 1, nb_rotations);
   for (i=0;i<N;i++)
   {
      err += (x0[i]-(double)x1[i])*(x0[i]-(double)x1[i]);
      ener += x0[i]*(double)x0[i];
   }
   snr = 20*log10(ener/err);
   printf ("SNR for size %d (%d pulses) is %f (was %f without inverse)\n", N, K, snr, snr0);
   if (snr < 60 || snr0 > 20)
      ret = 1;
}

int main(void)
{
   test_rotation(15, 3);
   test_rotation(23, 5);
   test_rotation(50, 3);
   test_rotation(80, 1);
   return ret;
}
