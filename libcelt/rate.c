/* (C) 2007 Jean-Marc Valin, CSIRO
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <math.h>
#include "modes.h"
#include "cwrs.h"
#include "arch.h"

#include "entcode.h"

int log2_frac(ec_uint32 val, int frac)
{
   int i;
   /* EC_ILOG() actually returns log2()+1, go figure */
   int L = EC_ILOG(val)-1;
   //printf ("in: %d %d ", val, L);
   if (L>14)
      val >>= L-14;
   else if (L<14)
      val <<= 14-L;
   L <<= frac;
   //printf ("%d\n", val);
   for (i=0;i<frac;i++)
   {
      val = (val*val) >> 15;
      //printf ("%d\n", val);
      if (val > 16384)
         L |= (1<<(frac-i-1));
      else   
         val <<= 1;
   }
   return L;
}

int log2_frac64(ec_uint64 val, int frac)
{
   int i;
   /* EC_ILOG64() actually returns log2()+1, go figure */
   int L = EC_ILOG64(val)-1;
   //printf ("in: %d %d ", val, L);
   if (L>14)
      val >>= L-14;
   else if (L<14)
      val <<= 14-L;
   L <<= frac;
   //printf ("%d\n", val);
   for (i=0;i<frac;i++)
   {
      val = (val*val) >> 15;
      //printf ("%d\n", val);
      if (val > 16384)
         L |= (1<<(frac-i-1));
      else   
         val <<= 1;
   }
   return L;
}

int compute_allocation(const CELTMode *m, int *pulses)
{
   int i, N, BC, bits;
   const int *eBands = m->eBands;
   BC = m->nbMdctBlocks*m->nbChannels;
   bits = 0;
   for (i=0;i<m->nbEBands;i++)
   {
      int q;
      N = BC*(eBands[i+1]-eBands[i]);
      q = pulses[i];
      if (q<=0)
      {
         bits += log2_frac64(eBands[i] - (eBands[i+1]-eBands[i]), 8) + (1<<8);
         q = -q;
      }
      if (q != 0)
         bits += log2_frac64(ncwrs64(N, pulses[i]), 8);
   }
   return (bits+255)>>8;
}

int bits2pulses(int bits, int N)
{
   int i, b, prev;
   /* FIXME: This is terribly inefficient */
   prev = 0;
   i=1;
   b = log2_frac64(ncwrs(N, i),0);
   while (b<bits)
   {
      prev=b;
      i++;
      b = log2_frac64(ncwrs(N, i),0);
   }
   if (bits-prev < b-bits)
      i--;
   return i;
}

#if 0
int main()
{
   int i;
   /*for(i=1;i<2000000000;i+=1738)
   {
      printf ("%d %d\n", i, frac_log2(i, 10));
   }*/
   for (i=4;i<=32;i*=2)
   {
      int j;
      for (j=0;j<30;j++)
      {
         printf ("%d %d %d\n", i, j, bits2pulses(j,i));
      }
   }
   return 0;
}
#endif
