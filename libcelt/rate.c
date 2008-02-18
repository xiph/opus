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
#include "rate.h"

#define BITRES 4
#define BITROUND 8
#define BITOVERFLOW 10000

#define MAX_PULSES 64

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


void alloc_init(struct alloc_data *alloc, const CELTMode *m)
{
   int i, prevN, BC;
   const int *eBands = m->eBands;
   
   alloc->mode = m;
   alloc->len = m->nbEBands;
   alloc->bands = m->eBands;
   alloc->bits = celt_alloc(m->nbEBands*sizeof(int*));
   
   BC = m->nbMdctBlocks*m->nbChannels;
   prevN = -1;
   for (i=0;i<alloc->len;i++)
   {
      int N = BC*(eBands[i+1]-eBands[i]);
      if (N == prevN && eBands[i] < m->pitchEnd)
      {
         alloc->bits[i] = alloc->bits[i-1];
      } else {
         int j;
         /* FIXME: We could save memory here */
         alloc->bits[i] = celt_alloc(MAX_PULSES*sizeof(int));
         for (j=0;j<MAX_PULSES;j++)
         {
            int done = 0;
            int pulses = j;
            /* For bands where there's no pitch, id 1 corresponds to intra prediction 
               with no pulse. id 2 means intra prediction with one pulse, and so on.*/
            if (eBands[i] >= m->pitchEnd)
               pulses -= 1;
            if (pulses < 0)
               alloc->bits[i][j] = 0;
            else {
               alloc->bits[i][j] = log2_frac64(ncwrs64(N, pulses),BITRES);
               /* FIXME: Could there be a better test for the max number of pulses that fit in 64 bits? */
               if (alloc->bits[i][j] > (60<<BITRES))
                  done = 1;
               /* Add the intra-frame prediction bits */
               if (eBands[i] >= m->pitchEnd)
               {
                  int max_pos = 2*eBands[i]-eBands[i+1];
                  if (max_pos > 32)
                     max_pos = 32;
                  alloc->bits[i][j] += (1<<BITRES) + log2_frac(max_pos,BITRES);
               }
            }
            if (done)
               break;
         }
         for (;j<MAX_PULSES;j++)
            alloc->bits[i][j] = BITOVERFLOW;
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
      }
   }
   celt_free(alloc->bits);
}

int bits2pulses(const struct alloc_data *alloc, int band, int bits)
{
   int lo, hi;
   lo = 0;
   hi = MAX_PULSES-1;
   
   while (hi-lo != 1)
   {
      int mid = (lo+hi)>>1;
      if (alloc->bits[band][mid] >= bits)
         hi = mid;
      else
         lo = mid;
   }
   if (bits-alloc->bits[band][lo] <= alloc->bits[band][hi]-bits)
      return lo;
   else
      return hi;
}

int vec_bits2pulses(const struct alloc_data *alloc, const int *bands, int *bits, int *pulses, int len)
{
   int i, BC;
   int sum=0;
   BC = alloc->mode->nbMdctBlocks*alloc->mode->nbChannels;

   for (i=0;i<len;i++)
   {
      pulses[i] = bits2pulses(alloc, i, bits[i]);
      sum += alloc->bits[i][pulses[i]];
   }
   //printf ("sum = %d\n", sum);
   return sum;
}

int interp_bits2pulses(const struct alloc_data *alloc, int *bits1, int *bits2, int total, int *pulses, int len)
{
   int lo, hi, out;
   int j;
   int bits[len];
   const int *bands = alloc->bands;
   lo = 0;
   hi = 1<<BITRES;
   while (hi-lo != 1)
   {
      int mid = (lo+hi)>>1;
      for (j=0;j<len;j++)
         bits[j] = ((1<<BITRES)-mid)*bits1[j] + mid*bits2[j];
      if (vec_bits2pulses(alloc, bands, bits, pulses, len) > total<<BITRES)
         hi = mid;
      else
         lo = mid;
   }
   //printf ("interp bisection gave %d\n", lo);
   for (j=0;j<len;j++)
      bits[j] = ((1<<BITRES)-lo)*bits1[j] + lo*bits2[j];
   out = vec_bits2pulses(alloc, bands, bits, pulses, len);
   /* Do some refinement to use up all bits. In the first pass, we can only add pulses to 
      bands that are under their allocated budget. In the second pass, anything goes */
   int firstpass = 1;
   while(1)
   {
      int incremented = 0;
      for (j=0;j<len;j++)
      {
         if ((!firstpass || alloc->bits[j][pulses[j]] < bits[j]) && pulses[j]<MAX_PULSES-1)
         {
            if (out+alloc->bits[j][pulses[j]+1]-alloc->bits[j][pulses[j]] <= total<<BITRES)
            {
               out = out+alloc->bits[j][pulses[j]+1]-alloc->bits[j][pulses[j]];
               pulses[j] += 1;
               incremented = 1;
               //printf ("INCREMENT %d\n", j);
            }
         }
      }
      if (!incremented)
      {
         if (firstpass)
            firstpass = 0;
         else
            break;
      }
   }
   return (out+BITROUND) >> BITRES;
}

int compute_allocation(const struct alloc_data *alloc, int *offsets, int total, int *pulses)
{
   int lo, hi, len;
   const CELTMode *m;

   m = alloc->mode;
   len = m->nbEBands;
   lo = 0;
   hi = m->nbAllocVectors - 1;
   while (hi-lo != 1)
   {
      int j;
      int bits[len];
      int pulses[len];
      int mid = (lo+hi) >> 1;
      for (j=0;j<len;j++)
      {
         bits[j] = (m->allocVectors[mid*len+j] + offsets[j])<<BITRES;
         if (bits[j] < 0)
            bits[j] = 0;
         //printf ("%d ", bits[j]);
      }
      //printf ("\n");
      if (vec_bits2pulses(alloc, alloc->bands, bits, pulses, len) > total<<BITRES)
         hi = mid;
      else
         lo = mid;
      //printf ("lo = %d, hi = %d\n", lo, hi);
   }
   {
      int bits1[len];
      int bits2[len];
      int j;
      for (j=0;j<len;j++)
      {
         bits1[j] = m->allocVectors[lo*len+j] + offsets[j];
         bits2[j] = m->allocVectors[hi*len+j] + offsets[j];
         if (bits1[j] < 0)
            bits1[j] = 0;
         if (bits2[j] < 0)
            bits2[j] = 0;
      }
      return interp_bits2pulses(alloc, bits1, bits2, total, pulses, len);
   }
}

#if 0
int main()
{
   int i;
   printf ("log(128) = %d\n", EC_ILOG(128));
   for(i=1;i<2000000000;i+=1738)
   {
      printf ("%d %d\n", i, log2_frac(i, 10));
   }
   return 0;
}
#endif
#if 0
int main()
{
   int i;
   int offsets[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   int bits[18] = {10, 9, 9, 8, 8, 8, 8, 8, 8, 8, 9, 10, 8, 9, 10, 11, 6, 7};
   int bits1[18] = {8, 7, 7, 6, 6, 6, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
   int bits2[18] = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
   int bank[20] = {0,  4,  8, 12, 16, 20, 24, 28, 32, 38, 44, 52, 62, 74, 90,112,142,182, 232,256};
   int pulses[18];
   struct alloc_data alloc;
   
   alloc_init(&alloc, celt_mode0);
   int b;
   //b = vec_bits2pulses(&alloc, bank, bits, pulses, 18);
   //printf ("total: %d bits\n", b);
   //for (i=0;i<18;i++)
   //   printf ("%d ", pulses[i]);
   //printf ("\n");
   //b = interp_bits2pulses(&alloc, bits1, bits2, 162, pulses, 18);
   b = compute_allocation(&alloc, offsets, 190, pulses);
   printf ("total: %d bits\n", b);
   for (i=0;i<18;i++)
      printf ("%d ", pulses[i]);
   printf ("\n");

   alloc_clear(&alloc);
   return 0;
}
#endif
