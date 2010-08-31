/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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

/*Determines if V(N,K) fits in a 32-bit unsigned integer.
  N and K are themselves limited to 15 bits.*/
static int fits_in32(int _n, int _k)
{
   static const celt_int16 maxN[15] = {
      32767, 32767, 32767, 1476, 283, 109,  60,  40,
       29,  24,  20,  18,  16,  14,  13};
   static const celt_int16 maxK[15] = {
      32767, 32767, 32767, 32767, 1172, 238,  95,  53,
       36,  27,  22,  18,  16,  15,  13};
   if (_n>=14)
   {
      if (_k>=14)
         return 0;
      else
         return _n <= maxN[_k];
   } else {
      return _k <= maxK[_n];
   }
}

void compute_pulse_cache(CELTMode *m, int LM)
{
   int i;
   int curr=0;
   int nbEntries=0;
   int entryN[100], entryK[100], entryI[100];
   const celt_int16 *eBands = m->eBands;
   PulseCache *cache = &m->cache;
   celt_int16 *cindex;
   unsigned char *bits;

   cindex = celt_alloc(sizeof(cache->index[0])*m->nbEBands*(LM+2));
   cache->index = cindex;

   for (i=0;i<=LM+1;i++)
   {
      int j;
      for (j=0;j<m->nbEBands;j++)
      {
         int k;
         int N = (eBands[j+1]-eBands[j])<<i>>1;
         cindex[i*m->nbEBands+j] = -1;
         for (k=0;k<=i;k++)
         {
            int n;
            for (n=0;n<m->nbEBands && (k!=i || n<j);n++)
            {
               if (N == (eBands[n+1]-eBands[n])<<k>>1)
               {
                  cindex[i*m->nbEBands+j] = cindex[k*m->nbEBands+n];
                  break;
               }
            }
         }
         if (cache->index[i*m->nbEBands+j] == -1 && N!=0)
         {
            int K;
            entryN[nbEntries] = N;
            K = 0;
            while (fits_in32(N,get_pulses(K+1)) && K<MAX_PSEUDO-1)
               K++;
            entryK[nbEntries] = K;
            cindex[i*m->nbEBands+j] = curr;
            entryI[nbEntries] = curr;

            curr += K+1;
            nbEntries++;
         }
      }
   }
   bits = celt_alloc(sizeof(unsigned char)*curr);
   cache->bits = bits;
   cache->size = curr;
   for (i=0;i<nbEntries;i++)
   {
      int j;
      unsigned char *ptr = bits+entryI[i];
      celt_int16 tmp[MAX_PULSES];
      get_required_bits(tmp, entryN[i], get_pulses(entryK[i]), BITRES);
      for (j=1;j<=entryK[i];j++)
         ptr[j] = tmp[get_pulses(j)]-1;
      ptr[0] = entryK[i];
   }
}

#endif /* !STATIC_MODES */



static inline void interp_bits2pulses(const CELTMode *m, int start, int end, int *bits1, int *bits2, int total, int *bits, int *ebits, int *fine_priority, int len, int _C, int M)
{
   int psum;
   int lo, hi;
   int j;
   int logM;
   const int C = CHANNELS(_C);
   SAVE_STACK;

   logM = log2_frac(M, BITRES);
   lo = 0;
   hi = 1<<BITRES;
   while (hi-lo != 1)
   {
      int mid = (lo+hi)>>1;
      psum = 0;
      for (j=start;j<end;j++)
         psum += (((1<<BITRES)-mid)*bits1[j] + mid*bits2[j])>>BITRES;
      if (psum > (total<<BITRES))
         hi = mid;
      else
         lo = mid;
   }
   psum = 0;
   /*printf ("interp bisection gave %d\n", lo);*/
   for (j=start;j<end;j++)
   {
      bits[j] = (((1<<BITRES)-lo)*bits1[j] + lo*bits2[j])>>BITRES;
      psum += bits[j];
   }
   /* Allocate the remaining bits */
   {
      int left, perband;
      left = (total<<BITRES)-psum;
      perband = left/(end-start);
      for (j=start;j<end;j++)
         bits[j] += perband;
      left = left-end*perband;
      for (j=start;j<start+left;j++)
         bits[j]++;
   }
   for (j=start;j<end;j++)
   {
      int N0, N, den;
      int offset;
      N0 = m->eBands[j+1]-m->eBands[j];
      N=M*N0;
      /* Compensate for the extra DoF in stereo */
      den=(C*N+ ((C==2 && N>2) ? 1 : 0));

      /* Offset for the number of fine bits compared to their "fair share" of total/N */
      offset = N*C*(((m->logN[j] + logM)>>1)-FINE_OFFSET);

      /* N=2 is the only point that doesn't match the curve */
      if (N==2)
         offset += N*C<<BITRES>>2;

      /* Changing the offset for allocating the second and third fine energy bit */
      if (bits[j] + offset < den*2<<BITRES)
         offset += (m->logN[j] + logM)*N*C>>BITRES-1;
      else if (bits[j] + offset < den*3<<BITRES)
         offset += (m->logN[j] + logM)*N*C>>BITRES;

      ebits[j] = (bits[j] + offset + (den<<(BITRES-1))) / (den<<BITRES);

      /* If we rounded down, make it a candidate for final fine energy pass */
      fine_priority[j] = ebits[j]*(den<<BITRES) >= bits[j]+offset;

      /* For N=1, all bits go to fine energy except for a single sign bit */
      if (N==1)
         ebits[j] = (bits[j]/C >> BITRES)-1;
      /* Make sure the first bit is spent on fine energy */
      if (ebits[j] < 1)
         ebits[j] = 1;

      /* Make sure not to bust */
      if (C*ebits[j] > (bits[j]>>BITRES))
         ebits[j] = bits[j]/C >> BITRES;

      if (ebits[j]>7)
         ebits[j]=7;
      if (ebits[j]<0)
         ebits[j]=0;

      /* The bits used for fine allocation can't be used for pulses */
      bits[j] -= C*ebits[j]<<BITRES;
      if (bits[j] < 0)
         bits[j] = 0;
   }
   RESTORE_STACK;
}

void compute_allocation(const CELTMode *m, int start, int end, int *offsets, int total, int *pulses, int *ebits, int *fine_priority, int _C, int M)
{
   int lo, hi, len, j;
   const int C = CHANNELS(_C);
   VARDECL(int, bits1);
   VARDECL(int, bits2);
   SAVE_STACK;
   
   len = m->nbEBands;
   ALLOC(bits1, len, int);
   ALLOC(bits2, len, int);

   lo = 0;
   hi = m->nbAllocVectors - 1;
   while (hi-lo != 1)
   {
      int psum = 0;
      int mid = (lo+hi) >> 1;
      for (j=start;j<end;j++)
      {
         int N = m->eBands[j+1]-m->eBands[j];
         bits1[j] = (C*M*N*m->allocVectors[mid*len+j] + offsets[j]);
         if (bits1[j] < 0)
            bits1[j] = 0;
         psum += bits1[j];
         /*printf ("%d ", bits[j]);*/
      }
      /*printf ("\n");*/
      if (psum > (total<<BITRES))
         hi = mid;
      else
         lo = mid;
      /*printf ("lo = %d, hi = %d\n", lo, hi);*/
   }
   /*printf ("interp between %d and %d\n", lo, hi);*/
   for (j=start;j<end;j++)
   {
      int N = m->eBands[j+1]-m->eBands[j];
      bits1[j] = C*M*N*m->allocVectors[lo*len+j] + offsets[j];
      bits2[j] = C*M*N*m->allocVectors[hi*len+j] + offsets[j];
      if (bits1[j] < 0)
         bits1[j] = 0;
      if (bits2[j] < 0)
         bits2[j] = 0;
   }
   interp_bits2pulses(m, start, end, bits1, bits2, total, pulses, ebits, fine_priority, len, C, M);
   RESTORE_STACK;
}

