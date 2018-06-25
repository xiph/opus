/* Copyright (c) 2009-2010 Xiph.Org Foundation
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

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
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

#include "celt_lpc.h"
#include "arch.h"
#include "common.h"
#include "pitch.h"

float _celt_lpc(
      opus_val16       *_lpc, /* out: [0...p-1] LPC coefficients      */
      opus_val16 *rc,
const opus_val32 *ac,  /* in:  [0...p] autocorrelation values  */
int          p
)
{
   int i, j;
   opus_val32 r;
   opus_val32 error = ac[0];
#ifdef FIXED_POINT
   opus_val32 lpc[LPC_ORDER];
#else
   float *lpc = _lpc;
#endif

   RNN_CLEAR(lpc, p);
   RNN_CLEAR(rc, p);
   if (ac[0] != 0)
   {
      for (i = 0; i < p; i++) {
         /* Sum up this iteration's reflection coefficient */
         opus_val32 rr = 0;
         for (j = 0; j < i; j++)
            rr += MULT32_32_Q31(lpc[j],ac[i - j]);
         rr += SHR32(ac[i + 1],3);
         r = -SHL32(rr,3)/error;
         rc[i] = r;
         /*  Update LPC coefficients and total error */
         lpc[i] = SHR32(r,3);
         for (j = 0; j < (i+1)>>1; j++)
         {
            opus_val32 tmp1, tmp2;
            tmp1 = lpc[j];
            tmp2 = lpc[i-1-j];
            lpc[j]     = tmp1 + MULT32_32_Q31(r,tmp2);
            lpc[i-1-j] = tmp2 + MULT32_32_Q31(r,tmp1);
         }

         error = error - MULT32_32_Q31(MULT32_32_Q31(r,r),error);
         /* Bail out once we get 30 dB gain */
#ifdef FIXED_POINT
         if (error<SHR32(ac[0],10))
            break;
#else
         if (error<.001f*ac[0])
            break;
#endif
      }
   }
#ifdef FIXED_POINT
   for (i=0;i<p;i++)
      _lpc[i] = ROUND16(lpc[i],16);
#endif
   return error;
}


void celt_fir(
         const opus_val16 *x,
         const opus_val16 *num,
         opus_val16 *y,
         int N,
         int ord)
{
   int i,j;
   opus_val16 rnum[ord];
   for(i=0;i<ord;i++)
      rnum[i] = num[ord-i-1];
   for (i=0;i<N-3;i+=4)
   {
      opus_val32 sum[4];
      sum[0] = SHL32(EXTEND32(x[i  ]), SIG_SHIFT);
      sum[1] = SHL32(EXTEND32(x[i+1]), SIG_SHIFT),
      sum[2] = SHL32(EXTEND32(x[i+2]), SIG_SHIFT);
      sum[3] = SHL32(EXTEND32(x[i+3]), SIG_SHIFT);
      xcorr_kernel(rnum, x+i-ord, sum, ord);
      y[i  ] = ROUND16(sum[0], SIG_SHIFT);
      y[i+1] = ROUND16(sum[1], SIG_SHIFT);
      y[i+2] = ROUND16(sum[2], SIG_SHIFT);
      y[i+3] = ROUND16(sum[3], SIG_SHIFT);
   }
   for (;i<N;i++)
   {
      opus_val32 sum = SHL32(EXTEND32(x[i]), SIG_SHIFT);
      for (j=0;j<ord;j++)
         sum = MAC16_16(sum,rnum[j],x[i+j-ord]);
      y[i] = ROUND16(sum, SIG_SHIFT);
   }
}

void celt_iir(const opus_val32 *_x,
         const opus_val16 *den,
         opus_val32 *_y,
         int N,
         int ord,
         opus_val16 *mem)
{
#ifdef SMALL_FOOTPRINT
   int i,j;
   for (i=0;i<N;i++)
   {
      opus_val32 sum = _x[i];
      for (j=0;j<ord;j++)
      {
         sum -= MULT16_16(den[j],mem[j]);
      }
      for (j=ord-1;j>=1;j--)
      {
         mem[j]=mem[j-1];
      }
      mem[0] = SROUND16(sum, SIG_SHIFT);
      _y[i] = sum;
   }
#else
   int i,j;
   celt_assert((ord&3)==0);
   opus_val16 rden[ord];
   opus_val16 y[N+ord];
   for(i=0;i<ord;i++)
      rden[i] = den[ord-i-1];
   for(i=0;i<ord;i++)
      y[i] = -mem[ord-i-1];
   for(;i<N+ord;i++)
      y[i]=0;
   for (i=0;i<N-3;i+=4)
   {
      /* Unroll by 4 as if it were an FIR filter */
      opus_val32 sum[4];
      sum[0]=_x[i];
      sum[1]=_x[i+1];
      sum[2]=_x[i+2];
      sum[3]=_x[i+3];
      xcorr_kernel(rden, y+i, sum, ord);

      /* Patch up the result to compensate for the fact that this is an IIR */
      y[i+ord  ] = -SROUND16(sum[0],SIG_SHIFT);
      _y[i  ] = sum[0];
      sum[1] = MAC16_16(sum[1], y[i+ord  ], den[0]);
      y[i+ord+1] = -SROUND16(sum[1],SIG_SHIFT);
      _y[i+1] = sum[1];
      sum[2] = MAC16_16(sum[2], y[i+ord+1], den[0]);
      sum[2] = MAC16_16(sum[2], y[i+ord  ], den[1]);
      y[i+ord+2] = -SROUND16(sum[2],SIG_SHIFT);
      _y[i+2] = sum[2];

      sum[3] = MAC16_16(sum[3], y[i+ord+2], den[0]);
      sum[3] = MAC16_16(sum[3], y[i+ord+1], den[1]);
      sum[3] = MAC16_16(sum[3], y[i+ord  ], den[2]);
      y[i+ord+3] = -SROUND16(sum[3],SIG_SHIFT);
      _y[i+3] = sum[3];
   }
   for (;i<N;i++)
   {
      opus_val32 sum = _x[i];
      for (j=0;j<ord;j++)
         sum -= MULT16_16(rden[j],y[i+j]);
      y[i+ord] = SROUND16(sum,SIG_SHIFT);
      _y[i] = sum;
   }
   for(i=0;i<ord;i++)
      mem[i] = _y[N-i-1];
#endif
}

int _celt_autocorr(
                   const opus_val16 *x,   /*  in: [0...n-1] samples x   */
                   opus_val32       *ac,  /* out: [0...lag-1] ac values */
                   const opus_val16       *window,
                   int          overlap,
                   int          lag,
                   int          n)
{
   opus_val32 d;
   int i, k;
   int fastN=n-lag;
   int shift;
   const opus_val16 *xptr;
   opus_val16 xx[n];
   celt_assert(n>0);
   celt_assert(overlap>=0);
   if (overlap == 0)
   {
      xptr = x;
   } else {
      for (i=0;i<n;i++)
         xx[i] = x[i];
      for (i=0;i<overlap;i++)
      {
         xx[i] = MULT16_16_Q15(x[i],window[i]);
         xx[n-i-1] = MULT16_16_Q15(x[n-i-1],window[i]);
      }
      xptr = xx;
   }
   shift=0;
   celt_pitch_xcorr(xptr, xptr, ac, fastN, lag+1);
   for (k=0;k<=lag;k++)
   {
      for (i = k+fastN, d = 0; i < n; i++)
         d = MAC16_16(d, xptr[i], xptr[i-k]);
      ac[k] += d;
   }
   return shift;
}
