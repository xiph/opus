/* Copyright (c) 2002-2008 Jean-Marc Valin
   Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
/**
   @file mathops.h
   @brief Various math functions
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

#ifndef MATHOPS_H
#define MATHOPS_H

#include "arch.h"
#include "entcode.h"
#include "os_support.h"

#ifndef OVERRIDE_FIND_MAX16
static inline int find_max16(celt_word16 *x, int len)
{
   celt_word16 max_corr=-VERY_LARGE16;
   int i, id = 0;
   for (i=0;i<len;i++)
   {
      if (x[i] > max_corr)
      {
         id = i;
         max_corr = x[i];
      }
   }
   return id;
}
#endif

#ifndef OVERRIDE_FIND_MAX32
static inline int find_max32(celt_word32 *x, int len)
{
   celt_word32 max_corr=-VERY_LARGE32;
   int i, id = 0;
   for (i=0;i<len;i++)
   {
      if (x[i] > max_corr)
      {
         id = i;
         max_corr = x[i];
      }
   }
   return id;
}
#endif

#define FRAC_MUL16(a,b) ((16384+((celt_int32)(celt_int16)(a)*(celt_int16)(b)))>>15)
static inline celt_int16 bitexact_cos(celt_int16 x)
{
   celt_int32 tmp;
   celt_int16 x2;
   tmp = (4096+((celt_int32)(x)*(x)))>>13;
   if (tmp > 32767)
      tmp = 32767;
   x2 = tmp;
   x2 = (32767-x2) + FRAC_MUL16(x2, (-7651 + FRAC_MUL16(x2, (8277 + FRAC_MUL16(-626, x2)))));
   if (x2 > 32766)
      x2 = 32766;
   return 1+x2;
}


#ifndef FIXED_POINT

#define celt_sqrt(x) ((float)sqrt(x))
#define celt_psqrt(x) ((float)sqrt(x))
#define celt_rsqrt(x) (1.f/celt_sqrt(x))
#define celt_rsqrt_norm(x) (celt_rsqrt(x))
#define celt_acos acos
#define celt_exp exp
#define celt_cos_norm(x) (cos((.5f*M_PI)*(x)))
#define celt_atan atan
#define celt_rcp(x) (1.f/(x))
#define celt_div(a,b) ((a)/(b))

#ifdef FLOAT_APPROX

/* Note: This assumes radix-2 floating point with the exponent at bits 23..30 and an offset of 127
         denorm, +/- inf and NaN are *not* handled */

/** Base-2 log approximation (log2(x)). */
static inline float celt_log2(float x)
{
   int integer;
   float frac;
   union {
      float f;
      celt_uint32 i;
   } in;
   in.f = x;
   integer = (in.i>>23)-127;
   in.i -= integer<<23;
   frac = in.f - 1.5f;
   frac = -0.41445418f + frac*(0.95909232f
          + frac*(-0.33951290f + frac*0.16541097f));
   return 1+integer+frac;
}

/** Base-2 exponential approximation (2^x). */
static inline float celt_exp2(float x)
{
   int integer;
   float frac;
   union {
      float f;
      celt_uint32 i;
   } res;
   integer = floor(x);
   if (integer < -50)
      return 0;
   frac = x-integer;
   /* K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2 */
   res.f = 0.99992522f + frac * (0.69583354f
           + frac * (0.22606716f + 0.078024523f*frac));
   res.i = (res.i + (integer<<23)) & 0x7fffffff;
   return res.f;
}

#else
#define celt_log2(x) (1.442695040888963387*log(x))
#define celt_exp2(x) (exp(0.6931471805599453094*(x)))
#endif

#endif



#ifdef FIXED_POINT

#include "os_support.h"

#ifndef OVERRIDE_CELT_ILOG2
/** Integer log in base2. Undefined for zero and negative numbers */
static inline celt_int16 celt_ilog2(celt_int32 x)
{
   celt_assert2(x>0, "celt_ilog2() only defined for strictly positive numbers");
   return EC_ILOG(x)-1;
}
#endif


#ifndef OVERRIDE_CELT_MAXABS16
static inline celt_word16 celt_maxabs16(celt_word16 *x, int len)
{
   int i;
   celt_word16 maxval = 0;
   for (i=0;i<len;i++)
      maxval = MAX16(maxval, ABS16(x[i]));
   return maxval;
}
#endif

/** Integer log in base2. Defined for zero, but not for negative numbers */
static inline celt_int16 celt_zlog2(celt_word32 x)
{
   return x <= 0 ? 0 : celt_ilog2(x);
}

/** Reciprocal sqrt approximation in the range [0.25,1) (Q16 in, Q14 out) */
static inline celt_word16 celt_rsqrt_norm(celt_word32 x)
{
   celt_word16 n;
   celt_word16 r;
   celt_word16 r2;
   celt_word16 y;
   /* Range of n is [-16384,32767] ([-0.5,1) in Q15). */
   n = x-32768;
   /* Get a rough initial guess for the root.
      The optimal minimax quadratic approximation (using relative error) is
       r = 1.437799046117536+n*(-0.823394375837328+n*0.4096419668459485).
      Coefficients here, and the final result r, are Q14.*/
   r = ADD16(23557, MULT16_16_Q15(n, ADD16(-13490, MULT16_16_Q15(n, 6713))));
   /* We want y = x*r*r-1 in Q15, but x is 32-bit Q16 and r is Q14.
      We can compute the result from n and r using Q15 multiplies with some
       adjustment, carefully done to avoid overflow.
      Range of y is [-1564,1594]. */
   r2 = MULT16_16_Q15(r, r);
   y = SHL16(SUB16(ADD16(MULT16_16_Q15(r2, n), r2), 16384), 1);
   /* Apply a 2nd-order Householder iteration: r += r*y*(y*0.375-0.5).
      This yields the Q14 reciprocal square root of the Q16 x, with a maximum
       relative error of 1.04956E-4, a (relative) RMSE of 2.80979E-5, and a
       peak absolute error of 2.26591/16384. */
   return ADD16(r, MULT16_16_Q15(r, MULT16_16_Q15(y,
              SUB16(MULT16_16_Q15(y, 12288), 16384))));
}

/** Reciprocal sqrt approximation (Q30 input, Q0 output or equivalent) */
static inline celt_word32 celt_rsqrt(celt_word32 x)
{
   int k;
   k = celt_ilog2(x)>>1;
   x = VSHR32(x, (k-7)<<1);
   return PSHR32(celt_rsqrt_norm(x), k);
}

/** Sqrt approximation (QX input, QX/2 output) */
static inline celt_word32 celt_sqrt(celt_word32 x)
{
   int k;
   celt_word16 n;
   celt_word32 rt;
   const celt_word16 C[5] = {23175, 11561, -3011, 1699, -664};
   if (x==0)
      return 0;
   k = (celt_ilog2(x)>>1)-7;
   x = VSHR32(x, (k<<1));
   n = x-32768;
   rt = ADD16(C[0], MULT16_16_Q15(n, ADD16(C[1], MULT16_16_Q15(n, ADD16(C[2], 
              MULT16_16_Q15(n, ADD16(C[3], MULT16_16_Q15(n, (C[4])))))))));
   rt = VSHR32(rt,7-k);
   return rt;
}

/** Sqrt approximation (QX input, QX/2 output) that assumes that the input is
    strictly positive */
static inline celt_word32 celt_psqrt(celt_word32 x)
{
   int k;
   celt_word16 n;
   celt_word32 rt;
   const celt_word16 C[5] = {23175, 11561, -3011, 1699, -664};
   k = (celt_ilog2(x)>>1)-7;
   x = VSHR32(x, (k<<1));
   n = x-32768;
   rt = ADD16(C[0], MULT16_16_Q15(n, ADD16(C[1], MULT16_16_Q15(n, ADD16(C[2], 
              MULT16_16_Q15(n, ADD16(C[3], MULT16_16_Q15(n, (C[4])))))))));
   rt = VSHR32(rt,7-k);
   return rt;
}

#define L1 32767
#define L2 -7651
#define L3 8277
#define L4 -626

static inline celt_word16 _celt_cos_pi_2(celt_word16 x)
{
   celt_word16 x2;
   
   x2 = MULT16_16_P15(x,x);
   return ADD16(1,MIN16(32766,ADD32(SUB16(L1,x2), MULT16_16_P15(x2, ADD32(L2, MULT16_16_P15(x2, ADD32(L3, MULT16_16_P15(L4, x2
                                                                                ))))))));
}

#undef L1
#undef L2
#undef L3
#undef L4

static inline celt_word16 celt_cos_norm(celt_word32 x)
{
   x = x&0x0001ffff;
   if (x>SHL32(EXTEND32(1), 16))
      x = SUB32(SHL32(EXTEND32(1), 17),x);
   if (x&0x00007fff)
   {
      if (x<SHL32(EXTEND32(1), 15))
      {
         return _celt_cos_pi_2(EXTRACT16(x));
      } else {
         return NEG32(_celt_cos_pi_2(EXTRACT16(65536-x)));
      }
   } else {
      if (x&0x0000ffff)
         return 0;
      else if (x&0x0001ffff)
         return -32767;
      else
         return 32767;
   }
}

static inline celt_word16 celt_log2(celt_word32 x)
{
   int i;
   celt_word16 n, frac;
   /*-0.41446   0.96093  -0.33981   0.15600 */
   /* -0.4144541824871411+32/16384, 0.9590923197873218, -0.3395129038105771,
       0.16541096501128538 */
   const celt_word16 C[4] = {-6758, 15715, -5563, 2708};
   if (x==0)
      return -32767;
   i = celt_ilog2(x);
   n = VSHR32(x,i-15)-32768-16384;
   frac = ADD16(C[0], MULT16_16_Q15(n, ADD16(C[1], MULT16_16_Q15(n, ADD16(C[2], MULT16_16_Q15(n, (C[3])))))));
   return SHL16(i-13,8)+SHR16(frac,14-8);
}

/*
 K0 = 1
 K1 = log(2)
 K2 = 3-4*log(2)
 K3 = 3*log(2) - 2
*/
#define D0 16383
#define D1 22804
#define D2 14819
#define D3 10204
/** Base-2 exponential approximation (2^x). (Q11 input, Q16 output) */
static inline celt_word32 celt_exp2(celt_word16 x)
{
   int integer;
   celt_word16 frac;
   integer = SHR16(x,11);
   if (integer>14)
      return 0x7f000000;
   else if (integer < -15)
      return 0;
   frac = SHL16(x-SHL16(integer,11),3);
   frac = ADD16(D0, MULT16_16_Q15(frac, ADD16(D1, MULT16_16_Q15(frac, ADD16(D2 , MULT16_16_Q15(D3,frac))))));
   return VSHR32(EXTEND32(frac), -integer-2);
}

/** Reciprocal approximation (Q15 input, Q16 output) */
static inline celt_word32 celt_rcp(celt_word32 x)
{
   int i;
   celt_word16 n;
   celt_word16 r;
   celt_assert2(x>0, "celt_rcp() only defined for positive values");
   i = celt_ilog2(x);
   /* n is Q15 with range [0,1). */
   n = VSHR32(x,i-15)-32768;
   /* Start with a linear approximation:
      r = 1.8823529411764706-0.9411764705882353*n.
      The coefficients and the result are Q14 in the range [15420,30840].*/
   r = ADD16(30840, MULT16_16_Q15(-15420, n));
   /* Perform two Newton iterations:
      r -= r*((r*n)-1.Q15)
         = r*((r*n)+(r-1.Q15)). */
   r = SUB16(r, MULT16_16_Q15(r,
             ADD16(MULT16_16_Q15(r, n), ADD16(r, -32768))));
   /* We subtract an extra 1 in the second iteration to avoid overflow; it also
       neatly compensates for truncation error in the rest of the process. */
   r = SUB16(r, ADD16(1, MULT16_16_Q15(r,
             ADD16(MULT16_16_Q15(r, n), ADD16(r, -32768)))));
   /* r is now the Q15 solution to 2/(n+1), with a maximum relative error
       of 7.05346E-5, a (relative) RMSE of 2.14418E-5, and a peak absolute
       error of 1.24665/32768. */
   return VSHR32(EXTEND32(r),i-16);
}

#define celt_div(a,b) MULT32_32_Q31((celt_word32)(a),celt_rcp(b))


#define M1 32767
#define M2 -21
#define M3 -11943
#define M4 4936

static inline celt_word16 celt_atan01(celt_word16 x)
{
   return MULT16_16_P15(x, ADD32(M1, MULT16_16_P15(x, ADD32(M2, MULT16_16_P15(x, ADD32(M3, MULT16_16_P15(M4, x)))))));
}

#undef M1
#undef M2
#undef M3
#undef M4

static inline celt_word16 celt_atan2p(celt_word16 y, celt_word16 x)
{
   if (y < x)
   {
      celt_word32 arg;
      arg = celt_div(SHL32(EXTEND32(y),15),x);
      if (arg >= 32767)
         arg = 32767;
      return SHR16(celt_atan01(EXTRACT16(arg)),1);
   } else {
      celt_word32 arg;
      arg = celt_div(SHL32(EXTEND32(x),15),y);
      if (arg >= 32767)
         arg = 32767;
      return 25736-SHR16(celt_atan01(EXTRACT16(arg)),1);
   }
}

#endif /* FIXED_POINT */


#endif /* MATHOPS_H */
