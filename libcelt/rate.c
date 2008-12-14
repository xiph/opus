/* (C) 2007-2008 Jean-Marc Valin, CSIRO
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include "modes.h"
#include "cwrs.h"
#include "arch.h"
#include "os_support.h"

#include "entcode.h"
#include "rate.h"


#ifndef STATIC_MODES

celt_int16_t **compute_alloc_cache(CELTMode *m, int C)
{
   int i, prevN;
   celt_int16_t **bits;
   const celt_int16_t *eBands = m->eBands;

   bits = celt_alloc(m->nbEBands*sizeof(celt_int16_t*));
   
   prevN = -1;
   for (i=0;i<m->nbEBands;i++)
   {
      int N = C*(eBands[i+1]-eBands[i]);
      if (N == prevN && eBands[i] < m->pitchEnd)
      {
         bits[i] = bits[i-1];
      } else {
         bits[i] = celt_alloc(MAX_PULSES*sizeof(celt_int16_t));
         get_required_bits(bits[i], N, MAX_PULSES, BITRES);
         prevN = N;
      }
   }
   return bits;
}

#endif /* !STATIC_MODES */



static int interp_bits2pulses(const CELTMode *m, int *bits1, int *bits2, int *ebits1, int *ebits2, int total, int *bits, int *ebits, int len)
{
   int esum, psum;
   int lo, hi;
   int j;
   const int C = CHANNELS(m);
   SAVE_STACK;
   lo = 0;
   hi = 1<<BITRES;
   while (hi-lo != 1)
   {
      int mid = (lo+hi)>>1;
      psum = 0;
      esum = 0;
      for (j=0;j<len;j++)
      {
         esum += (((1<<BITRES)-mid)*ebits1[j] + mid*ebits2[j] + (1<<(BITRES-1)))>>BITRES;
         psum += ((1<<BITRES)-mid)*bits1[j] + mid*bits2[j];
      }
      if (psum > (total-C*esum)<<BITRES)
         hi = mid;
      else
         lo = mid;
   }
   esum = 0;
   psum = 0;
   /*printf ("interp bisection gave %d\n", lo);*/
   for (j=0;j<len;j++)
   {
      ebits[j] = (((1<<BITRES)-lo)*ebits1[j] + lo*ebits2[j] + (1<<(BITRES-1)))>>BITRES;
      esum += ebits[j];
   }
   for (j=0;j<len;j++)
   {
      bits[j] = ((1<<BITRES)-lo)*bits1[j] + lo*bits2[j];
      psum += bits[j];
   }
   /* Allocate the remaining bits */
   {
      int left, perband;
      left = ((total-C*esum)<<BITRES)-psum;
      perband = left/len;
      for (j=0;j<len;j++)
         bits[j] += perband;
      left = left-len*perband;
      for (j=0;j<left;j++)
         bits[j]++;
   }
   RESTORE_STACK;
   return (total-C*esum)<<BITRES;
}

void compute_allocation(const CELTMode *m, int *offsets, const int *stereo_mode, int total, int *pulses, int *ebits)
{
   int lo, hi, len, j;
   int remaining_bits;
   VARDECL(int, bits1);
   VARDECL(int, bits2);
   VARDECL(int, ebits1);
   VARDECL(int, ebits2);
   const int C = CHANNELS(m);
   SAVE_STACK;
   
   len = m->nbEBands;
   ALLOC(bits1, len, int);
   ALLOC(bits2, len, int);
   ALLOC(ebits1, len, int);
   ALLOC(ebits2, len, int);

   lo = 0;
   hi = m->nbAllocVectors - 1;
   while (hi-lo != 1)
   {
      int psum = 0;
      int mid = (lo+hi) >> 1;
      for (j=0;j<len;j++)
      {
         bits1[j] = (m->allocVectors[mid*len+j] + offsets[j])<<BITRES;
         if (bits1[j] < 0)
            bits1[j] = 0;
         psum += bits1[j];
         /*printf ("%d ", bits[j]);*/
      }
      /*printf ("\n");*/
      if (psum > (total-C*m->energy_alloc[mid*(len+1)+len])<<BITRES)
         hi = mid;
      else
         lo = mid;
      /*printf ("lo = %d, hi = %d\n", lo, hi);*/
   }
   /*printf ("interp between %d and %d\n", lo, hi);*/
   for (j=0;j<len;j++)
   {
      ebits1[j] = m->energy_alloc[lo*(len+1)+j];
      ebits2[j] = m->energy_alloc[hi*(len+1)+j];
      bits1[j] = m->allocVectors[lo*len+j] + offsets[j];
      bits2[j] = m->allocVectors[hi*len+j] + offsets[j];
      if (bits1[j] < 0)
         bits1[j] = 0;
      if (bits2[j] < 0)
         bits2[j] = 0;
   }
   remaining_bits = interp_bits2pulses(m, bits1, bits2, ebits1, ebits2, total, pulses, ebits, len);
   RESTORE_STACK;
}

