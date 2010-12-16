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

   /* Scan for all unique band sizes */
   for (i=0;i<=LM+1;i++)
   {
      int j;
      for (j=0;j<m->nbEBands;j++)
      {
         int k;
         int N = (eBands[j+1]-eBands[j])<<i>>1;
         cindex[i*m->nbEBands+j] = -1;
         /* Find other bands that have the same size */
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
            while (fits_in32(N,get_pulses(K+1)) && K<MAX_PSEUDO)
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
   /* Compute the cache for all unique sizes */
   for (i=0;i<nbEntries;i++)
   {
      int j;
      unsigned char *ptr = bits+entryI[i];
      celt_int16 tmp[MAX_PULSES+1];
      get_required_bits(tmp, entryN[i], get_pulses(entryK[i]), BITRES);
      for (j=1;j<=entryK[i];j++)
         ptr[j] = tmp[get_pulses(j)]-1;
      ptr[0] = entryK[i];
   }
}

#endif /* !STATIC_MODES */


#define ALLOC_STEPS 6

static inline int interp_bits2pulses(const CELTMode *m, int start, int end, int skip_start,
      const int *bits1, const int *bits2, const int *thresh, int total, int skip_rsv,int *bits,
      int *ebits, int *fine_priority, int len, int _C, int LM, void *ec, int encode, int prev)
{
   int psum;
   int lo, hi;
   int i, j;
   int logM;
   const int C = CHANNELS(_C);
   int codedBands=-1;
   int alloc_floor;
   int left, percoeff;
   int done;
   int balance;
   SAVE_STACK;

   alloc_floor = C<<BITRES;

   logM = LM<<BITRES;
   lo = 0;
   hi = 1<<ALLOC_STEPS;
   for (i=0;i<ALLOC_STEPS;i++)
   {
      int mid = (lo+hi)>>1;
      psum = 0;
      done = 0;
      for (j=end;j-->start;)
      {
         int tmp = bits1[j] + (mid*bits2[j]>>ALLOC_STEPS);
         if (tmp >= thresh[j] || done)
         {
            done = 1;
            /* Don't allocate more than we can actually use */
            psum += IMIN(tmp, 64*C<<BITRES<<LM);
         } else {
            if (tmp >= alloc_floor)
               psum += alloc_floor;
         }
      }
      if (psum > total)
         hi = mid;
      else
         lo = mid;
   }
   psum = 0;
   /*printf ("interp bisection gave %d\n", lo);*/
   done = 0;
   for (j=end;j-->start;)
   {
      int tmp = bits1[j] + (lo*bits2[j]>>ALLOC_STEPS);
      if (tmp < thresh[j] && !done)
      {
         if (tmp >= alloc_floor)
            tmp = alloc_floor;
         else
            tmp = 0;
      } else
         done = 1;
      /* Don't allocate more than we can actually use */
      tmp = IMIN(tmp, 64*C<<BITRES<<LM);
      bits[j] = tmp;
      psum += tmp;
   }

   /* Decide which bands to skip, working backwards from the end. */
   for (codedBands=end;;codedBands--)
   {
      int band_width;
      int band_bits;
      int rem;
      j = codedBands-1;
      /*Figure out how many left-over bits we would be adding to this band.
        This can include bits we've stolen back from higher, skipped bands.*/
      left = total-psum;
      percoeff = left/(m->eBands[codedBands]-m->eBands[start]);
      left -= (m->eBands[codedBands]-m->eBands[start])*percoeff;
      /* Never skip the first band, nor a band that has been boosted by
          dynalloc.
         In the first case, we'd be coding a bit to signal we're going to waste
          all the other bits.
         In the second case, we'd be coding a bit to redistribute all the bits
          we just signaled should be cocentrated in this band. */
      if (j<=skip_start)
      {
         /* Give the bit we reserved to end skipping back to this band. */
         bits[j] += skip_rsv;
         break;
      }
      rem = IMAX(left-(m->eBands[j]-m->eBands[start]),0);
      band_width = m->eBands[codedBands]-m->eBands[j];
      band_bits = bits[j] + percoeff*band_width + rem;
      /*Only code a skip decision if we're above the threshold for this band.
        Otherwise it is force-skipped.
        This ensures that we have enough bits to code the skip flag.*/
      if (band_bits >= IMAX(thresh[j], alloc_floor+(1<<BITRES)))
      {
         if (encode)
         {
            /*This if() block is the only part of the allocation function that
               is not a mandatory part of the bitstream: any bands we choose to
               skip here must be explicitly signaled.*/
            /*Choose a threshold with some hysteresis to keep bands from
               fluctuating in and out.*/
            if (band_bits > ((j<prev?7:9)*band_width<<LM<<BITRES)>>4)
            {
               ec_enc_bit_prob((ec_enc *)ec, 1, 32768);
               break;
            }
            ec_enc_bit_prob((ec_enc *)ec, 0, 32768);
         } else if (ec_dec_bit_prob((ec_dec *)ec, 32768)) {
            break;
         }
         /*We used a bit to skip this band.*/
         psum += 1<<BITRES;
         band_bits -= 1<<BITRES;
      }
      /*Reclaim the bits originally allocated to this band.*/
      psum -= bits[j];
      if (band_bits >= alloc_floor)
      {
         /*If we have enough for a fine energy bit per channel, use it.*/
         psum += alloc_floor;
         bits[j] = alloc_floor;
      } else {
         /*Otherwise this band gets nothing at all.*/
         bits[j] = 0;
      }
   }

   /* Allocate the remaining bits */
   if (codedBands>start) {
      for (j=start;j<codedBands;j++)
         bits[j] += percoeff*(m->eBands[j+1]-m->eBands[j]);
      for (j=start;j<codedBands;j++)
      {
         int tmp = IMIN(left, m->eBands[j+1]-m->eBands[j]);
         bits[j] += tmp;
         left -= tmp;
      }
   }
   /*for (j=0;j<end;j++)printf("%d ", bits[j]);printf("\n");*/

   balance = 0;
   for (j=start;j<codedBands;j++)
   {
      int N0, N, den;
      int offset;
      int NClogN;

      celt_assert(bits[j] >= 0);
      N0 = m->eBands[j+1]-m->eBands[j];
      N=N0<<LM;

      if (N>1)
      {
         NClogN = N*C*(m->logN[j] + logM);

         /* Compensate for the extra DoF in stereo */
         den=(C*N+ ((C==2 && N>2) ? 1 : 0));

         /* Offset for the number of fine bits by log2(N)/2 + FINE_OFFSET
            compared to their "fair share" of total/N */
         offset = (NClogN>>1)-N*C*FINE_OFFSET;

         /* N=2 is the only point that doesn't match the curve */
         if (N==2)
            offset += N*C<<BITRES>>2;

         /* Changing the offset for allocating the second and third
             fine energy bit */
         if (bits[j] + offset < den*2<<BITRES)
            offset += NClogN>>2;
         else if (bits[j] + offset < den*3<<BITRES)
            offset += NClogN>>3;

         /* Divide with rounding */
         ebits[j] = IMAX(0, (bits[j] + offset + (den<<(BITRES-1))) / (den<<BITRES));

         /* If we rounded down, make it a candidate for final
             fine energy pass */
         fine_priority[j] = ebits[j]*(den<<BITRES) >= bits[j]+offset;

         /* Make sure not to bust */
         if (C*ebits[j] > (bits[j]>>BITRES))
            ebits[j] = bits[j]/C >> BITRES;

         /* More than 7 is useless because that's about as far as PVQ can go */
         if (ebits[j]>7)
            ebits[j]=7;

      } else {
         /* For N=1, all bits go to fine energy except for a single sign bit */
         ebits[j] = IMIN(IMAX(0,(bits[j]/C >> BITRES)-1),7);
         fine_priority[j] = (ebits[j]+1)*C<<BITRES >= (bits[j]-balance);
         /* N=1 bands can't take advantage of the re-balancing in
             quant_all_bands() because they don't have shape, only fine energy.
            Instead, do the re-balancing here.*/
         balance = IMAX(0,bits[j] - ((ebits[j]+1)*C<<BITRES));
         if (j+1<codedBands)
         {
            bits[j] -= balance;
            bits[j+1] += balance;
         }
      }

      /* The other bits are assigned to PVQ */
      bits[j] -= C*ebits[j]<<BITRES;
      celt_assert(bits[j] >= 0);
      celt_assert(ebits[j] >= 0);
   }
   /* The skipped bands use all their bits for fine energy. */
   for (;j<end;j++)
   {
      ebits[j] = bits[j]/C >> BITRES;
      celt_assert(C*ebits[j]<<BITRES == bits[j]);
      bits[j] = 0;
      fine_priority[j] = ebits[j]<1;
   }
   RESTORE_STACK;
   return codedBands;
}

int compute_allocation(const CELTMode *m, int start, int end, const int *offsets, int alloc_trim,
      int total, int *pulses, int *ebits, int *fine_priority, int _C, int LM, void *ec, int encode, int prev)
{
   int lo, hi, len, j;
   const int C = CHANNELS(_C);
   int codedBands;
   int skip_start;
   int skip_rsv;
   VARDECL(int, bits1);
   VARDECL(int, bits2);
   VARDECL(int, thresh);
   VARDECL(int, trim_offset);
   SAVE_STACK;
   
   total = IMAX(total, 0);
   len = m->nbEBands;
   skip_start = start;
   /* Reserve a bit to signal the end of manually skipped bands. */
   skip_rsv = total >= 1<<BITRES ? 1<<BITRES : 0;
   total -= skip_rsv;
   ALLOC(bits1, len, int);
   ALLOC(bits2, len, int);
   ALLOC(thresh, len, int);
   ALLOC(trim_offset, len, int);

   /* Below this threshold, we're sure not to allocate any PVQ bits */
   for (j=start;j<end;j++)
      thresh[j] = IMAX((C)<<BITRES, (3*(m->eBands[j+1]-m->eBands[j])<<LM<<BITRES)>>4);
   /* Tilt of the allocation curve */
   for (j=start;j<end;j++)
      trim_offset[j] = C*(m->eBands[j+1]-m->eBands[j])*(alloc_trim-5-LM)*(m->nbEBands-j-1)
            <<(LM+BITRES)>>6;

   lo = 0;
   hi = m->nbAllocVectors - 1;
   while (hi-lo != 1)
   {
      int psum = 0;
      int mid = (lo+hi) >> 1;
      for (j=start;j<end;j++)
      {
         int N = m->eBands[j+1]-m->eBands[j];
         bits1[j] = C*N*m->allocVectors[mid*len+j]<<LM>>2;
         if (bits1[j] > 0)
            bits1[j] += trim_offset[j];
         if (bits1[j] < 0)
            bits1[j] = 0;
         bits1[j] += offsets[j];
         if (bits1[j] >= thresh[j])
            psum += bits1[j];
         else if (bits1[j] >= C<<BITRES)
            psum += C<<BITRES;

         /*printf ("%d ", bits[j]);*/
      }
      /*printf ("\n");*/
      if (psum > total)
         hi = mid;
      else
         lo = mid;
      /*printf ("lo = %d, hi = %d\n", lo, hi);*/
   }
   /*printf ("interp between %d and %d\n", lo, hi);*/
   for (j=start;j<end;j++)
   {
      int N = m->eBands[j+1]-m->eBands[j];
      bits1[j] = (C*N*m->allocVectors[lo*len+j]<<LM>>2);
      bits2[j] = (C*N*m->allocVectors[hi*len+j]<<LM>>2) - bits1[j];
      if (bits1[j] > 0)
         bits1[j] += trim_offset[j];
      if (bits1[j] < 0)
         bits1[j] = 0;
      bits1[j] += offsets[j];
      if (offsets[j]>0)
         skip_start = j;
   }
   codedBands = interp_bits2pulses(m, start, end, skip_start, bits1, bits2, thresh,
         total, skip_rsv, pulses, ebits, fine_priority, len, C, LM, ec, encode, prev);
   RESTORE_STACK;
   return codedBands;
}

