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

#ifndef FIXED_POINT

#define celt_sqrt sqrt
#define celt_acos acos
#define celt_exp exp
#define celt_cos_norm(x) (cos((.5f*M_PI)*(x)))
#define celt_atan atan


#endif



#ifdef FIXED_POINT

#include "entcode.h"

static inline celt_int16_t celt_ilog2(celt_word32_t x)
{
   return EC_ILOG(x)-1;
}

#define C0 3634
#define C1 21173
#define C2 -12627
#define C3 4204

static inline celt_word32_t celt_sqrt(celt_word32_t x)
{
   int k;
   //printf ("%d ", x);
   celt_word32_t rt;
   /* ((EC_ILOG(x)-1)>>1) is just the int log4(x) (EC_ILOG returns log2 + 1) */
   k = ((EC_ILOG(x)-1)>>1)-6;
   x = VSHR32(x, (k<<1));
   rt = ADD16(C0, MULT16_16_Q14(x, ADD16(C1, MULT16_16_Q14(x, ADD16(C2, MULT16_16_Q14(x, (C3)))))));
   rt = VSHR32(rt,7-k);
   //printf ("%d\n", rt);
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
/* Input in Q11 format, output in Q16 */
static inline celt_word32_t celt_exp2(celt_word16_t x)
{
   int integer;
   celt_word16_t frac;
   integer = SHR16(x,11);
   if (integer>14)
      return 0x7fffffff;
   else if (integer < -15)
      return 0;
   frac = SHL16(x-SHL16(integer,11),3);
   frac = ADD16(D0, MULT16_16_Q14(frac, ADD16(D1, MULT16_16_Q14(frac, ADD16(D2 , MULT16_16_Q14(D3,frac))))));
   return VSHR32(EXTEND32(frac), -integer-2);
}


#endif


#endif
