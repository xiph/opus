/* Copyright (C) 2002-2008 Jean-Marc Valin */
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

#ifndef OVERRIDE_CELT_ILOG2
/** Integer log in base2. Undefined for zero and negative numbers */
static inline celt_int16_t celt_ilog2(celt_word32_t x)
{
   celt_assert2(x>0, "celt_ilog2() only defined for strictly positive numbers");
   return EC_ILOG(x)-1;
}
#endif

#ifndef OVERRIDE_FIND_MAX16
static inline int find_max16(celt_word16_t *x, int len)
{
   celt_word16_t max_corr=-VERY_LARGE16;
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
static inline int find_max32(celt_word32_t *x, int len)
{
   celt_word32_t max_corr=-VERY_LARGE32;
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


#ifndef FIXED_POINT

#define celt_sqrt(x) ((float)sqrt(x))
#define celt_psqrt(x) ((float)sqrt(x))
#define celt_rsqrt(x) (1.f/celt_sqrt(x))
#define celt_acos acos
#define celt_exp exp
#define celt_cos_norm(x) (cos((.5f*M_PI)*(x)))
#define celt_atan atan
#define celt_rcp(x) (1.f/(x))
#define celt_div(a,b) ((a)/(b))

#endif



#ifdef FIXED_POINT

#include "os_support.h"

#ifndef OVERRIDE_CELT_MAXABS16
static inline celt_word16_t celt_maxabs16(celt_word16_t *x, int len)
{
   int i;
   celt_word16_t maxval = 0;
   for (i=0;i<len;i++)
      maxval = MAX16(maxval, ABS16(x[i]));
   return maxval;
}
#endif

/** Integer log in base2. Defined for zero, but not for negative numbers */
static inline celt_int16_t celt_zlog2(celt_word32_t x)
{
   return x <= 0 ? 0 : celt_ilog2(x);
}

/** Reciprocal sqrt approximation (Q30 input, Q0 output or equivalent) */
static inline celt_word32_t celt_rsqrt(celt_word32_t x)
{
   int k;
   celt_word16_t n;
   celt_word32_t rt;
   const celt_word16_t C[5] = {23126, -11496, 9812, -9097, 4100};
   k = celt_ilog2(x)>>1;
   x = VSHR32(x, (k-7)<<1);
   /* Range of n is [-16384,32767] */
   n = x-32768;
   rt = ADD16(C[0], MULT16_16_Q15(n, ADD16(C[1], MULT16_16_Q15(n, ADD16(C[2], 
              MULT16_16_Q15(n, ADD16(C[3], MULT16_16_Q15(n, (C[4])))))))));
   rt = VSHR32(rt,k);
   return rt;
}

/** Sqrt approximation (QX input, QX/2 output) */
static inline celt_word32_t celt_sqrt(celt_word32_t x)
{
   int k;
   celt_word16_t n;
   celt_word32_t rt;
   const celt_word16_t C[5] = {23174, 11584, -3011, 1570, -557};
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
static inline celt_word32_t celt_psqrt(celt_word32_t x)
{
   int k;
   celt_word16_t n;
   celt_word32_t rt;
   const celt_word16_t C[5] = {23174, 11584, -3011, 1570, -557};
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

static inline celt_word16_t _celt_cos_pi_2(celt_word16_t x)
{
   celt_word16_t x2;
   
   x2 = MULT16_16_P15(x,x);
   return ADD16(1,MIN16(32766,ADD32(SUB16(L1,x2), MULT16_16_P15(x2, ADD32(L2, MULT16_16_P15(x2, ADD32(L3, MULT16_16_P15(L4, x2
                                                                                ))))))));
}

#undef L1
#undef L2
#undef L3
#undef L4

static inline celt_word16_t celt_cos_norm(celt_word32_t x)
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

static inline celt_word16_t celt_log2(celt_word32_t x)
{
   int i;
   celt_word16_t n, frac;
   /*-0.41446   0.96093  -0.33981   0.15600 */
   const celt_word16_t C[4] = {-6791, 7872, -1392, 319};
   if (x==0)
      return -32767;
   i = celt_ilog2(x);
   n = VSHR32(x,i-15)-32768-16384;
   frac = ADD16(C[0], MULT16_16_Q14(n, ADD16(C[1], MULT16_16_Q14(n, ADD16(C[2], MULT16_16_Q14(n, (C[3])))))));
   /*printf ("%d %d %d %d\n", x, n, ret, SHL16(i-13,8)+SHR16(ret,14-8));*/
   return SHL16(i-13,8)+SHR16(frac,14-8);
}

/*
 K0 = 1
 K1 = log(2)
 K2 = 3-4*log(2)
 K3 = 3*log(2) - 2
*/
#define D0 16384
#define D1 11356
#define D2 3726
#define D3 1301
/** Base-2 exponential approximation (2^x). (Q11 input, Q16 output) */
static inline celt_word32_t celt_exp2(celt_word16_t x)
{
   int integer;
   celt_word16_t frac;
   integer = SHR16(x,11);
   if (integer>14)
      return 0x7f000000;
   else if (integer < -15)
      return 0;
   frac = SHL16(x-SHL16(integer,11),3);
   frac = ADD16(D0, MULT16_16_Q14(frac, ADD16(D1, MULT16_16_Q14(frac, ADD16(D2 , MULT16_16_Q14(D3,frac))))));
   return VSHR32(EXTEND32(frac), -integer-2);
}

/** Reciprocal approximation (Q15 input, Q16 output) */
static inline celt_word32_t celt_rcp(celt_word32_t x)
{
   int i;
   celt_word16_t n, frac;
   const celt_word16_t C[5] = {21848, -7251, 2403, -934, 327};
   celt_assert2(x>0, "celt_rcp() only defined for positive values");
   i = celt_ilog2(x);
   n = VSHR32(x,i-16)-SHL32(EXTEND32(3),15);
   frac = ADD16(C[0], MULT16_16_Q15(n, ADD16(C[1], MULT16_16_Q15(n, ADD16(C[2], 
                MULT16_16_Q15(n, ADD16(C[3], MULT16_16_Q15(n, (C[4])))))))));
   return VSHR32(EXTEND32(frac),i-16);
}

#define celt_div(a,b) MULT32_32_Q31((celt_word32_t)(a),celt_rcp(b))

#endif /* FIXED_POINT */


#endif /* MATHOPS_H */
