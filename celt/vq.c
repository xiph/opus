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

#include "mathops.h"
#include "cwrs.h"
#include "vq.h"
#include "arch.h"
#include "os_support.h"
#include "bands.h"
#include "rate.h"
#include "pitch.h"
#include "SigProc_FIX.h"

#if defined(MIPSr1_ASM)
#include "mips/vq_mipsr1.h"
#endif

#ifndef OVERRIDE_vq_exp_rotation1
static void exp_rotation1(celt_norm *X, int len, int stride, opus_val16 c, opus_val16 s)
{
   int i;
   opus_val16 ms;
   celt_norm *Xptr;
   Xptr = X;
   ms = NEG16(s);
   for (i=0;i<len-stride;i++)
   {
      celt_norm x1, x2;
      x1 = Xptr[0];
      x2 = Xptr[stride];
      Xptr[stride] = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, x2),  s, x1), 15));
      *Xptr++      = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, x1), ms, x2), 15));
   }
   Xptr = &X[len-2*stride-1];
   for (i=len-2*stride-1;i>=0;i--)
   {
      celt_norm x1, x2;
      x1 = Xptr[0];
      x2 = Xptr[stride];
      Xptr[stride] = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, x2),  s, x1), 15));
      *Xptr--      = EXTRACT16(PSHR32(MAC16_16(MULT16_16(c, x1), ms, x2), 15));
   }
}
#endif /* OVERRIDE_vq_exp_rotation1 */

void exp_rotation(celt_norm *X, int len, int dir, int stride, int K, int spread)
{
   static const int SPREAD_FACTOR[3]={15,10,5};
   int i;
   opus_val16 c, s;
   opus_val16 gain, theta;
   int stride2=0;
   int factor;

   if (2*K>=len || spread==SPREAD_NONE)
      return;
   factor = SPREAD_FACTOR[spread-1];

   gain = celt_div((opus_val32)MULT16_16(Q15_ONE,len),(opus_val32)(len+factor*K));
   theta = HALF16(MULT16_16_Q15(gain,gain));

   c = celt_cos_norm(EXTEND32(theta));
   s = celt_cos_norm(EXTEND32(SUB16(Q15ONE,theta))); /*  sin(theta) */

   if (len>=8*stride)
   {
      stride2 = 1;
      /* This is just a simple (equivalent) way of computing sqrt(len/stride) with rounding.
         It's basically incrementing long as (stride2+0.5)^2 < len/stride. */
      while ((stride2*stride2+stride2)*stride + (stride>>2) < len)
         stride2++;
   }
   /*NOTE: As a minor optimization, we could be passing around log2(B), not B, for both this and for
      extract_collapse_mask().*/
   len = celt_udiv(len, stride);
   for (i=0;i<stride;i++)
   {
      if (dir < 0)
      {
         if (stride2)
            exp_rotation1(X+i*len, len, stride2, s, c);
         exp_rotation1(X+i*len, len, 1, c, s);
      } else {
         exp_rotation1(X+i*len, len, 1, c, -s);
         if (stride2)
            exp_rotation1(X+i*len, len, stride2, s, -c);
      }
   }
}

/** Takes the pitch vector and the decoded residual vector, computes the gain
    that will give ||p+g*y||=1 and mixes the residual with the pitch. */
static void normalise_residual(int * OPUS_RESTRICT iy, celt_norm * OPUS_RESTRICT X,
      int N, opus_val32 Ryy, opus_val32 gain)
{
   int i;
#ifdef FIXED_POINT
   int k;
#endif
   opus_val32 t;
   opus_val16 g;

#ifdef FIXED_POINT
   k = celt_ilog2(Ryy)>>1;
#endif
   t = VSHR32(Ryy, 2*(k-7));
   g = MULT32_32_Q31(celt_rsqrt_norm(t),gain);

   i=0;
   do
      X[i] = EXTRACT16(PSHR32(MULT16_16(g, iy[i]), k+1));
   while (++i < N);
}

static unsigned extract_collapse_mask(int *iy, int N, int B)
{
   unsigned collapse_mask;
   int N0;
   int i;
   if (B<=1)
      return 1;
   /*NOTE: As a minor optimization, we could be passing around log2(B), not B, for both this and for
      exp_rotation().*/
   N0 = celt_udiv(N, B);
   collapse_mask = 0;
   i=0; do {
      int j;
      unsigned tmp=0;
      j=0; do {
         tmp |= iy[i*N0+j];
      } while (++j<N0);
      collapse_mask |= (tmp!=0)<<i;
   } while (++i<B);
   return collapse_mask;
}

opus_val16 op_pvq_search_c(celt_norm *X, int *iy, int K, int N, int arch)
{
   VARDECL(celt_norm, y);
   VARDECL(int, signx);
   int i, j;
   int pulsesLeft;
   opus_val32 sum;
   opus_val32 xy;
   opus_val16 yy;
   SAVE_STACK;

   (void)arch;
   ALLOC(y, N, celt_norm);
   ALLOC(signx, N, int);

   /* Get rid of the sign */
   sum = 0;
   j=0; do {
      signx[j] = X[j]<0;
      /* OPT: Make sure the compiler doesn't use a branch on ABS16(). */
      X[j] = ABS16(X[j]);
      iy[j] = 0;
      y[j] = 0;
   } while (++j<N);

   xy = yy = 0;

   pulsesLeft = K;

   /* Do a pre-search by projecting on the pyramid */
   if (K > (N>>1))
   {
      opus_val16 rcp;
      j=0; do {
         sum += X[j];
      }  while (++j<N);

      /* If X is too small, just replace it with a pulse at 0 */
#ifdef FIXED_POINT
      if (sum <= K)
#else
      /* Prevents infinities and NaNs from causing too many pulses
         to be allocated. 64 is an approximation of infinity here. */
      if (!(sum > EPSILON && sum < 64))
#endif
      {
         X[0] = QCONST16(1.f,14);
         j=1; do
            X[j]=0;
         while (++j<N);
         sum = QCONST16(1.f,14);
      }
#ifdef FIXED_POINT
      rcp = EXTRACT16(MULT16_32_Q16(K, celt_rcp(sum)));
#else
      /* Using K+e with e < 1 guarantees we cannot get more than K pulses. */
      rcp = EXTRACT16(MULT16_32_Q16(K+0.8f, celt_rcp(sum)));
#endif
      j=0; do {
#ifdef FIXED_POINT
         /* It's really important to round *towards zero* here */
         iy[j] = MULT16_16_Q15(X[j],rcp);
#else
         iy[j] = (int)floor(rcp*X[j]);
#endif
         y[j] = (celt_norm)iy[j];
         yy = MAC16_16(yy, y[j],y[j]);
         xy = MAC16_16(xy, X[j],y[j]);
         y[j] *= 2;
         pulsesLeft -= iy[j];
      }  while (++j<N);
   }
   celt_sig_assert(pulsesLeft>=0);

   /* This should never happen, but just in case it does (e.g. on silence)
      we fill the first bin with pulses. */
#ifdef FIXED_POINT_DEBUG
   celt_sig_assert(pulsesLeft<=N+3);
#endif
   if (pulsesLeft > N+3)
   {
      opus_val16 tmp = (opus_val16)pulsesLeft;
      yy = MAC16_16(yy, tmp, tmp);
      yy = MAC16_16(yy, tmp, y[0]);
      iy[0] += pulsesLeft;
      pulsesLeft=0;
   }

   for (i=0;i<pulsesLeft;i++)
   {
      opus_val16 Rxy, Ryy;
      int best_id;
      opus_val32 best_num;
      opus_val16 best_den;
#ifdef FIXED_POINT
      int rshift;
#endif
#ifdef FIXED_POINT
      rshift = 1+celt_ilog2(K-pulsesLeft+i+1);
#endif
      best_id = 0;
      /* The squared magnitude term gets added anyway, so we might as well
         add it outside the loop */
      yy = ADD16(yy, 1);

      /* Calculations for position 0 are out of the loop, in part to reduce
         mispredicted branches (since the if condition is usually false)
         in the loop. */
      /* Temporary sums of the new pulse(s) */
      Rxy = EXTRACT16(SHR32(ADD32(xy, EXTEND32(X[0])),rshift));
      /* We're multiplying y[j] by two so we don't have to do it here */
      Ryy = ADD16(yy, y[0]);

      /* Approximate score: we maximise Rxy/sqrt(Ryy) (we're guaranteed that
         Rxy is positive because the sign is pre-computed) */
      Rxy = MULT16_16_Q15(Rxy,Rxy);
      best_den = Ryy;
      best_num = Rxy;
      j=1;
      do {
         /* Temporary sums of the new pulse(s) */
         Rxy = EXTRACT16(SHR32(ADD32(xy, EXTEND32(X[j])),rshift));
         /* We're multiplying y[j] by two so we don't have to do it here */
         Ryy = ADD16(yy, y[j]);

         /* Approximate score: we maximise Rxy/sqrt(Ryy) (we're guaranteed that
            Rxy is positive because the sign is pre-computed) */
         Rxy = MULT16_16_Q15(Rxy,Rxy);
         /* The idea is to check for num/den >= best_num/best_den, but that way
            we can do it without any division */
         /* OPT: It's not clear whether a cmov is faster than a branch here
            since the condition is more often false than true and using
            a cmov introduces data dependencies across iterations. The optimal
            choice may be architecture-dependent. */
         if (opus_unlikely(MULT16_16(best_den, Rxy) > MULT16_16(Ryy, best_num)))
         {
            best_den = Ryy;
            best_num = Rxy;
            best_id = j;
         }
      } while (++j<N);

      /* Updating the sums of the new pulse(s) */
      xy = ADD32(xy, EXTEND32(X[best_id]));
      /* We're multiplying y[j] by two so we don't have to do it here */
      yy = ADD16(yy, y[best_id]);

      /* Only now that we've made the final choice, update y/iy */
      /* Multiplying y[j] by 2 so we don't have to do it everywhere else */
      y[best_id] += 2;
      iy[best_id]++;
   }

   /* Put the original sign back */
   j=0;
   do {
      /*iy[j] = signx[j] ? -iy[j] : iy[j];*/
      /* OPT: The is more likely to be compiled without a branch than the code above
         but has the same performance otherwise. */
      iy[j] = (iy[j]^-signx[j]) + signx[j];
   } while (++j<N);
   RESTORE_STACK;
   return yy;
}

#ifdef ENABLE_QEXT
#include "macros.h"

static opus_val32 op_pvq_search_N2(const celt_norm *X, int *iy, int *up_iy, int K, int up, int *refine) {
   opus_val32 sum;
   opus_val32 rcp_sum;
   int offset;
   sum = ABS16(X[0]) + ABS16(X[1]);
   if (sum == 0) {
      iy[0] = K;
      up_iy[0] = up*K;
      iy[1]=up_iy[1]=0;
      *refine=0;
      return K*K*up*up;
   }
#ifdef FIXED_POINT
   rcp_sum = celt_rcp(sum);
   iy[0] = PSHR32(silk_SMULWW(MULT16_16(K,X[0]), rcp_sum), 15);
   up_iy[0] = PSHR32(silk_SMULWW(MULT16_16(up*K,X[0]), rcp_sum), 15);
#else
   rcp_sum = 1.f/sum;
   iy[0] = (int)floor(.5f+K*X[0]*rcp_sum);
   up_iy[0] = (int)floor(.5f+up*K*X[0]*rcp_sum);
#endif
   up_iy[0] = IMAX(up*iy[0] - (up-1)/2, IMIN(up*iy[0] + (up-1)/2, up_iy[0]));
   offset = up_iy[0] - up*iy[0];
   iy[1] = K-abs(iy[0]);
   up_iy[1] = up*K-abs(up_iy[0]);
   if (X[1] < 0) {
      iy[1] = -iy[1];
      up_iy[1] = -up_iy[1];
      offset = -offset;
   }
   *refine = offset;
   return MULT16_16(up_iy[0],up_iy[0]) + MULT16_16(up_iy[1],up_iy[1]);
}

static int op_pvq_n4(const celt_norm *X, int *iy, int *iy0, int K, int up, int margin, opus_val32 rcp_sum) {
   int i;
   int dir;
   opus_val32 rounding[4];
   int iysum = 0;
   for (i=0;i<4;i++) {
      opus_val32 tmp;
#ifdef FIXED_POINT
      tmp = silk_SMULWW(MULT16_16(K,ABS16(X[i])), rcp_sum);
      iy[i] = PSHR32(tmp, 15);
#else
      tmp = K*ABS16(X[i])*rcp_sum;
      iy[i] = (int)floor(.5+tmp);
#endif
      iysum += iy[i];
      rounding[i] = tmp - SHL32(iy[i], 15);
   }
   if (abs(iysum - K) > 4) {
      return 1;
   }
   dir = iysum < K ? 1 : -1;
   while (iysum != K) {
      opus_val32 roundval=-100000*dir;
      int roundpos=0;
      for (i=0;i<4;i++) {
         if ((rounding[i]-roundval)*dir > 0 && abs(iy[i]-up*iy0[i]) < (margin-1) && !(dir==-1 && iy[i] == 0)) {
            roundval = rounding[i];
            roundpos = i;
         }
      }
      iy[roundpos] += dir;
      rounding[roundpos] -= SHL32(dir, 15);
      iysum+=dir;
   }
   return 0;
}

static opus_val32 op_pvq_search_N4(const celt_norm *X, int *iy, int *up_iy, int K, int up, int *refine) {
   opus_val32 rcp_sum;
   opus_val32 sum;
   int i;
   int failed=0;
   opus_val32 yy=0;
   sum = ABS16(X[0]) + ABS16(X[1]) + ABS16(X[2]) + ABS16(X[3]);
   if (sum == 0)
      failed = 1;
   else
      rcp_sum = celt_rcp(sum);
   failed = failed || op_pvq_n4(X, iy, iy, K, 1, K+1, rcp_sum);
   failed = failed || op_pvq_n4(X, up_iy, iy, up*K, up, up, rcp_sum);
   if (failed) {
      iy[0] = K;
      for (i=1;i<4;i++) iy[i] = 0;
      up_iy[0] = up*K;
      for (i=1;i<4;i++) up_iy[i] = 0;
   }
   for (i=0;i<4;i++) {
      yy += MULT16_16(up_iy[i], up_iy[i]);
      if (X[i] < 0) {
         iy[i] = -iy[i];
         up_iy[i] = -up_iy[i];
      }
      refine[i] = up_iy[i]-up*iy[i];
   }
   return yy;
}
#endif

unsigned alg_quant(celt_norm *X, int N, int K, int spread, int B, ec_enc *enc,
      opus_val32 gain, int resynth,
#ifdef ENABLE_QEXT
      ec_enc *ext_enc, int extra_bits,
#endif
      int arch)
{
   VARDECL(int, iy);
   opus_val32 yy;
   unsigned collapse_mask;
   SAVE_STACK;

   celt_assert2(K>0, "alg_quant() needs at least one pulse");
   celt_assert2(N>1, "alg_quant() needs at least two dimensions");

   /* Covers vectorization by up to 4. */
   ALLOC(iy, N+3, int);

   exp_rotation(X, N, 1, B, K, spread);

#ifdef ENABLE_QEXT
   if (N==2 && extra_bits >= 2) {
      int refine;
      int up_iy[2];
      int up;
      up = (1<<extra_bits)-1;
      yy = op_pvq_search_N2(X, iy, up_iy, K, up, &refine);
      collapse_mask = extract_collapse_mask(up_iy, N, B);
      ec_enc_uint(ext_enc, refine+(up-1)/2, up);
      if (resynth)
         normalise_residual(up_iy, X, N, yy, gain);
   } else if (N==4 && extra_bits >= 2) {
      int i;
      int up_iy[4];
      int refine[4];
      int up;
      up = (1<<extra_bits)-1;
      yy = op_pvq_search_N4(X, iy, up_iy, K, up, refine);
      collapse_mask = extract_collapse_mask(up_iy, N, B);
      for (i=0;i<3;i++) ec_enc_uint(ext_enc, refine[i]+up-1, 2*up-1);
      if (iy[3]==0) ec_enc_bits(ext_enc, up_iy[3]<0, 1);
      if (resynth)
         normalise_residual(up_iy, X, N, yy, gain);
   } else
#endif
   {
      yy = op_pvq_search(X, iy, K, N, arch);
      collapse_mask = extract_collapse_mask(iy, N, B);
      if (resynth)
         normalise_residual(iy, X, N, yy, gain);
   }

   encode_pulses(iy, N, K, enc);

   if (resynth)
      exp_rotation(X, N, -1, B, K, spread);

   RESTORE_STACK;
   return collapse_mask;
}

/** Decode pulse vector and combine the result with the pitch vector to produce
    the final normalised signal in the current band. */
unsigned alg_unquant(celt_norm *X, int N, int K, int spread, int B,
      ec_dec *dec, opus_val32 gain
#ifdef ENABLE_QEXT
      , ec_enc *ext_dec, int extra_bits
#endif
      )
{
   opus_val32 Ryy;
   unsigned collapse_mask;
   VARDECL(int, iy);
   SAVE_STACK;

   celt_assert2(K>0, "alg_unquant() needs at least one pulse");
   celt_assert2(N>1, "alg_unquant() needs at least two dimensions");
   ALLOC(iy, N, int);
   Ryy = decode_pulses(iy, N, K, dec);
#ifdef ENABLE_QEXT
   if (N==2 && extra_bits >= 2) {
      int up;
      int refine;
      up = (1<<extra_bits)-1;
      refine = ec_dec_uint(ext_dec, up) - (up-1)/2;
      iy[0] *= up;
      iy[1] *= up;
      if (iy[1] == 0) {
         iy[1] = (iy[0] > 0) ? -refine : refine;
         iy[0] += (refine*iy[0] > 0) ? -refine : refine;
      } else if (iy[1] > 0) {
         iy[0] += refine;
         iy[1] -= refine*(iy[0]>0?1:-1);
      } else {
         iy[0] -= refine;
         iy[1] -= refine*(iy[0]>0?1:-1);
      }
      Ryy = iy[0]*iy[0] + iy[1]*iy[1];
   } else if (N==4 && extra_bits >= 2) {
      int i;
      int refine[4];
      int up;
      int sign=0;
      up = (1<<extra_bits)-1;
      for (i=0;i<3;i++) refine[i] = ec_dec_uint(ext_dec, 2*up-1) - (up-1);
      if (iy[3]==0) sign = ec_dec_bits(ext_dec, 1);
      else sign = iy[3] < 0;
      for (i=0;i<3;i++) {
         iy[i] = iy[i]*up + refine[i];
      }
      iy[3] = up*K - abs(iy[0]) - abs(iy[1]) - abs(iy[2]);
      if (sign) iy[3] = -iy[3];
      Ryy = 0;
      for (i=0;i<4;i++) Ryy += iy[i]*iy[i];
   }
#endif
   normalise_residual(iy, X, N, Ryy, gain);
   exp_rotation(X, N, -1, B, K, spread);
   collapse_mask = extract_collapse_mask(iy, N, B);
   RESTORE_STACK;
   return collapse_mask;
}

#ifndef OVERRIDE_renormalise_vector
void renormalise_vector(celt_norm *X, int N, opus_val32 gain, int arch)
{
   int i;
#ifdef FIXED_POINT
   int k;
#endif
   opus_val32 E;
   opus_val16 g;
   opus_val32 t;
   celt_norm *xptr;
   E = EPSILON + celt_inner_prod(X, X, N, arch);
#ifdef FIXED_POINT
   k = celt_ilog2(E)>>1;
#endif
   t = VSHR32(E, 2*(k-7));
   g = MULT32_32_Q31(celt_rsqrt_norm(t),gain);

   xptr = X;
   for (i=0;i<N;i++)
   {
      *xptr = EXTRACT16(PSHR32(MULT16_16(g, *xptr), k+1));
      xptr++;
   }
   /*return celt_sqrt(E);*/
}
#endif /* OVERRIDE_renormalise_vector */

opus_int32 stereo_itheta(const celt_norm *X, const celt_norm *Y, int stereo, int N, int arch)
{
   int i;
   int itheta;
   opus_val16 mid, side;
   opus_val32 Emid, Eside;

   Emid = Eside = EPSILON;
   if (stereo)
   {
      for (i=0;i<N;i++)
      {
         celt_norm m, s;
         m = ADD16(SHR16(X[i],1),SHR16(Y[i],1));
         s = SUB16(SHR16(X[i],1),SHR16(Y[i],1));
         Emid = MAC16_16(Emid, m, m);
         Eside = MAC16_16(Eside, s, s);
      }
   } else {
      Emid += celt_inner_prod(X, X, N, arch);
      Eside += celt_inner_prod(Y, Y, N, arch);
   }
   mid = celt_sqrt(Emid);
   side = celt_sqrt(Eside);
#ifdef FIXED_POINT
   /* 0.63662 = 2/pi */
   itheta = MULT16_16(QCONST16(0.63662f,15),celt_atan2p(side, mid))<<1;
#else
   itheta = (int)floor(.5f+65536.f*16384*0.6366197724f*atan2(side,mid));
#endif

   return itheta;
}
