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
#include "os_support.h"

#include "entcode.h"

#define BITRES 4
#define BITOVERFLOW 1000

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

int bits2pulses(int bits, int N)
{
   int i, b, prev;
   /* FIXME: This is terribly inefficient. Do a bisection instead
   but be careful about overflows */
   prev = 0;
   i=1;
   b = log2_frac64(ncwrs64(N, i),0);
   while (b<bits)
   {
      prev=b;
      i++;
      b = log2_frac64(ncwrs64(N, i),0);
   }
   if (bits-prev < b-bits)
      i--;
   return i;
}


struct alloc_data {
   int len;
   int **bits;
   int **rev_bits;
};

void alloc_init(struct alloc_data *alloc, const CELTMode *m)
{
   int i, prevN, BC;
   const int *eBands = m->eBands;
   alloc->bits = celt_alloc(m->nbEBands*sizeof(int*));
   alloc->rev_bits = celt_alloc(m->nbEBands*sizeof(int*));
   alloc->len = m->nbEBands;
   BC = m->nbMdctBlocks*m->nbChannels;
   prevN = -1;
   for (i=0;i<alloc->len;i++)
   {
      int N = BC*(eBands[i+1]-eBands[i]);
      if (N == prevN)
      {
         alloc->bits[i] = alloc->bits[i-1];
         alloc->rev_bits[i] = alloc->rev_bits[i-1];
      } else {
         int j;
         /* FIXME: We could save memory here */
         alloc->bits[i] = celt_alloc(64*sizeof(int));
         alloc->rev_bits[i] = celt_alloc(64*sizeof(int));
         for (j=0;j<64;j++)
         {
            alloc->bits[i][j] = log2_frac64(ncwrs64(N, j),BITRES);
            /* We could just update rev_bits here */
            if (alloc->bits[i][j] > (60>>BITRES))
               break;
         }
         for (;j<64;j++)
            alloc->bits[i][j] = BITOVERFLOW;
         for (j=0;j<32;j++)
            alloc->rev_bits[i][j] = bits2pulses(j, N);
         prevN = N;
      }
   }
}

void alloc_clear(struct alloc_data *alloc)
{
   int i;
   int *prevPtr = NULL;
   for (i=0;i<alloc->len;i++)
   {
      if (alloc->bits[i] != prevPtr)
      {
         prevPtr = alloc->bits[i];
         celt_free(alloc->bits[i]);
         celt_free(alloc->rev_bits[i]);
      }
   }
   celt_free(alloc->bits);
   celt_free(alloc->rev_bits);
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


int vec_bits2pulses(int *bands, int *bits, int *pulses, int len, int B)
{
   int i;
   int sum=0;
   for (i=0;i<len;i++)
   {
      int N = (bands[i+1]-bands[i])*B;
      pulses[i] = bits2pulses(bits[i], N);
      sum += log2_frac64(ncwrs(N, pulses[i]),8);
   }
   return (sum+255)>>8;
}

int interp_bits2pulses(int *bands, int *bits1, int *bits2, int total, int *pulses, int len, int B)
{
   int i;
   /* FIXME: This too is terribly inefficient. We should do a bisection instead */
   for (i=0;i<16;i++)
   {
      int j;
      int bits[len];
      for (j=0;j<len;j++)
         bits[j] = ((16-i)*bits1[j] + i*bits2[j]) >> 4;
      if (vec_bits2pulses(bands, bits, pulses, len, B) > total)
         break;
   }
   if (i==0)
      return -1;
   else {
      int j;
      int bits[len];
      /* Get the previous one (that didn't bust). Should rewrite that anyway */
      i--;
      for (j=0;j<len;j++)
         bits[j] = ((16-i)*bits1[j] + i*bits2[j]) >> 4;      
      return vec_bits2pulses(bands, bits, pulses, len, B);
   }
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
#if 0
int main()
{
   int i;
   int bits[18] = {10, 9, 9, 8, 8, 8, 8, 8, 8, 8, 9, 10, 8, 9, 10, 11, 6, 7};
   int bits1[18] = {8, 7, 7, 6, 6, 6, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
   int bits2[18] = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
   int bank[20] = {0,  4,  8, 12, 16, 20, 24, 28, 32, 38, 44, 52, 62, 74, 90,112,142,182, 232,256};
   int pulses[18];
   struct alloc_data alloc;
   
   alloc_init(&alloc, celt_mode0);
   int b = vec_bits2pulses(bank, bits, pulses, 18, 1);
   printf ("total: %d bits\n", b);
   for (i=0;i<18;i++)
      printf ("%d ", pulses[i]);
   printf ("\n");
   b = interp_bits2pulses(bank, bits1, bits2, 160, pulses, 18, 1);
   printf ("total: %d bits\n", b);
   for (i=0;i<18;i++)
      printf ("%d ", pulses[i]);
   printf ("\n");

   alloc_clear(&alloc);
   return 0;
}
#endif
