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

#define BITRES 4
#define BITROUND 8
#define BITOVERFLOW 10000

#ifndef STATIC_MODES
#if 0
static int log2_frac(ec_uint32 val, int frac)
{
   int i;
   /* EC_ILOG() actually returns log2()+1, go figure */
   int L = EC_ILOG(val)-1;
   /*printf ("in: %d %d ", val, L);*/
   if (L>14)
      val >>= L-14;
   else if (L<14)
      val <<= 14-L;
   L <<= frac;
   /*printf ("%d\n", val);*/
   for (i=0;i<frac;i++)
   {
      val = (val*val) >> 15;
      /*printf ("%d\n", val);*/
      if (val > 16384)
         L |= (1<<(frac-i-1));
      else   
         val <<= 1;
   }
   return L;
}
#endif

static int log2_frac64(ec_uint64 val, int frac)
{
   int i;
   /* EC_ILOG64() actually returns log2()+1, go figure */
   int L = EC_ILOG64(val)-1;
   /*printf ("in: %d %d ", val, L);*/
   if (L>14)
      val >>= L-14;
   else if (L<14)
      val <<= 14-L;
   L <<= frac;
   /*printf ("%d\n", val);*/
   for (i=0;i<frac;i++)
   {
      val = (val*val) >> 15;
      /*printf ("%d\n", val);*/
      if (val > 16384)
         L |= (1<<(frac-i-1));
      else   
         val <<= 1;
   }
   return L;
}

void compute_alloc_cache(CELTMode *m)
{
   int i, prevN, BC;
   celt_int16_t **bits;
   const celt_int16_t *eBands = m->eBands;

   bits = celt_alloc(m->nbEBands*sizeof(celt_int16_t*));
   
   BC = m->nbChannels;
   prevN = -1;
   for (i=0;i<m->nbEBands;i++)
   {
      int N = BC*(eBands[i+1]-eBands[i]);
      if (N == prevN && eBands[i] < m->pitchEnd)
      {
         bits[i] = bits[i-1];
      } else {
         int j;
         VARDECL(celt_uint64_t, u);
         SAVE_STACK;
         ALLOC(u, N, celt_uint64_t);
         /* FIXME: We could save memory here */
         bits[i] = celt_alloc(MAX_PULSES*sizeof(celt_int16_t));
         for (j=0;j<MAX_PULSES;j++)
         {
            int done = 0;
            int pulses = j;
            /* For bands where there's no pitch, id 1 corresponds to intra prediction 
            with no pulse. id 2 means intra prediction with one pulse, and so on.*/
            if (eBands[i] >= m->pitchEnd)
               pulses -= 1;
            if (pulses < 0)
               bits[i][j] = 0;
            else {
               celt_uint64_t nc;
               nc=pulses?ncwrs_unext64(N, u):ncwrs_u64(N, 0, u);
               bits[i][j] = log2_frac64(nc,BITRES);
               /* FIXME: Could there be a better test for the max number of pulses that fit in 64 bits? */
               if (bits[i][j] > (60<<BITRES))
                  done = 1;
               /* Add the intra-frame prediction sign bit */
               if (eBands[i] >= m->pitchEnd)
                  bits[i][j] += (1<<BITRES);
            }
            if (done)
               break;
         }
         for (;j<MAX_PULSES;j++)
            bits[i][j] = BITOVERFLOW;
         prevN = N;
         RESTORE_STACK;
      }
   }
   m->bits = (const celt_int16_t * const *)bits;
}

#endif /* !STATIC_MODES */

static inline int bits2pulses(const CELTMode *m, int band, int bits)
{
   int i;
   int lo, hi;
   lo = 0;
   hi = MAX_PULSES-1;
   
   /* Instead of using the "bisection condition" we use a fixed number of 
      iterations because it should be faster */
   /*while (hi-lo != 1)*/
   for (i=0;i<LOG_MAX_PULSES;i++)
   {
      int mid = (lo+hi)>>1;
      /* OPT: Make sure this is implemented with a conditional move */
      if (m->bits[band][mid] >= bits)
         hi = mid;
      else
         lo = mid;
   }
   if (bits-m->bits[band][lo] <= m->bits[band][hi]-bits)
      return lo;
   else
      return hi;
}

static int vec_bits2pulses(const CELTMode *m, int *bits, int *pulses, int len)
{
   int i;
   int sum=0;

   for (i=0;i<len;i++)
   {
      pulses[i] = bits2pulses(m, i, bits[i]);
      sum += m->bits[i][pulses[i]];
   }
   /*printf ("sum = %d\n", sum);*/
   return sum;
}

static int interp_bits2pulses(const CELTMode *m, int *bits1, int *bits2, int total, int *pulses, int len)
{
   int lo, hi, out;
   int j;
   VARDECL(int, bits);
   SAVE_STACK;
   ALLOC(bits, len, int);
   lo = 0;
   hi = 1<<BITRES;
   while (hi-lo != 1)
   {
      int mid = (lo+hi)>>1;
      for (j=0;j<len;j++)
         bits[j] = ((1<<BITRES)-mid)*bits1[j] + mid*bits2[j];
      if (vec_bits2pulses(m, bits, pulses, len) > total<<BITRES)
         hi = mid;
      else
         lo = mid;
   }
   /*printf ("interp bisection gave %d\n", lo);*/
   for (j=0;j<len;j++)
      bits[j] = ((1<<BITRES)-lo)*bits1[j] + lo*bits2[j];
   out = vec_bits2pulses(m, bits, pulses, len);
   /* Do some refinement to use up all bits. In the first pass, we can only add pulses to 
      bands that are under their allocated budget. In the second pass, anything goes */
   for (j=0;j<len;j++)
   {
      if (m->bits[j][pulses[j]] < bits[j] && pulses[j]<MAX_PULSES-1)
      {
         if (out+m->bits[j][pulses[j]+1]-m->bits[j][pulses[j]] <= total<<BITRES)
         {
            out = out+m->bits[j][pulses[j]+1]-m->bits[j][pulses[j]];
            pulses[j] += 1;
         }
      }
   }
   while(1)
   {
      int incremented = 0;
      for (j=0;j<len;j++)
      {
         if (pulses[j]<MAX_PULSES-1)
         {
            if (out+m->bits[j][pulses[j]+1]-m->bits[j][pulses[j]] <= total<<BITRES)
            {
               out = out+m->bits[j][pulses[j]+1]-m->bits[j][pulses[j]];
               pulses[j] += 1;
               incremented = 1;
            }
         }
      }
      if (!incremented)
            break;
   }
   RESTORE_STACK;
   return (out+BITROUND) >> BITRES;
}

int compute_allocation(const CELTMode *m, int *offsets, int total, int *pulses)
{
   int lo, hi, len, ret;
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
      int j;
      int mid = (lo+hi) >> 1;
      for (j=0;j<len;j++)
      {
         bits1[j] = (m->allocVectors[mid*len+j] + offsets[j])<<BITRES;
         if (bits1[j] < 0)
            bits1[j] = 0;
         /*printf ("%d ", bits[j]);*/
      }
      /*printf ("\n");*/
      if (vec_bits2pulses(m, bits1, pulses, len) > total<<BITRES)
         hi = mid;
      else
         lo = mid;
      /*printf ("lo = %d, hi = %d\n", lo, hi);*/
   }
   {
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
      ret = interp_bits2pulses(m, bits1, bits2, total, pulses, len);
      RESTORE_STACK;
      return ret;
   }
}

