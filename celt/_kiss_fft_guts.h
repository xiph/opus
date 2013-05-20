/*Copyright (c) 2003-2004, Mark Borgerding

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.*/

#ifndef KISS_FFT_GUTS_H
#define KISS_FFT_GUTS_H

#define MIN(a,b) ((a)<(b) ? (a):(b))
#define MAX(a,b) ((a)>(b) ? (a):(b))

/* kiss_fft.h
   defines kiss_fft_scalar as either short or a float type
   and defines
   typedef struct { kiss_fft_scalar r; kiss_fft_scalar i; }kiss_fft_cpx; */
#include "kiss_fft.h"

/*
  Explanation of macros dealing with complex math:

   C_MUL(m,a,b)         : m = a*b
   C_FIXDIV( c , div )  : if a fixed point impl., c /= div. noop otherwise
   C_SUB( res, a,b)     : res = a - b
   C_SUBFROM( res , a)  : res -= a
   C_ADDTO( res , a)    : res += a
 * */
#ifdef FIXED_POINT
#include "arch.h"


#define SAMP_MAX 2147483647
#define TWID_MAX 32767
#define TRIG_UPSCALE 1

#define SAMP_MIN -SAMP_MAX


#   define S_MUL(a,b) MULT16_32_Q15(b, a)

#   define C_MUL(m,a,b) \
      do{ (m).r = SUB32(S_MUL((a).r,(b).r) , S_MUL((a).i,(b).i)); \
          (m).i = ADD32(S_MUL((a).r,(b).i) , S_MUL((a).i,(b).r)); }while(0)

#   define C_MULC(m,a,b) \
      do{ (m).r = ADD32(S_MUL((a).r,(b).r) , S_MUL((a).i,(b).i)); \
          (m).i = SUB32(S_MUL((a).i,(b).r) , S_MUL((a).r,(b).i)); }while(0)

#   define C_MUL4(m,a,b) \
      do{ (m).r = SHR32(SUB32(S_MUL((a).r,(b).r) , S_MUL((a).i,(b).i)),2); \
          (m).i = SHR32(ADD32(S_MUL((a).r,(b).i) , S_MUL((a).i,(b).r)),2); }while(0)

#   define C_MULBYSCALAR( c, s ) \
      do{ (c).r =  S_MUL( (c).r , s ) ;\
          (c).i =  S_MUL( (c).i , s ) ; }while(0)

#   define DIVSCALAR(x,k) \
        (x) = S_MUL(  x, (TWID_MAX-((k)>>1))/(k)+1 )

#   define C_FIXDIV(c,div) \
        do {    DIVSCALAR( (c).r , div);  \
                DIVSCALAR( (c).i  , div); }while (0)

#define  C_ADD( res, a,b)\
    do {(res).r=ADD32((a).r,(b).r);  (res).i=ADD32((a).i,(b).i); \
    }while(0)
#define  C_SUB( res, a,b)\
    do {(res).r=SUB32((a).r,(b).r);  (res).i=SUB32((a).i,(b).i); \
    }while(0)
#define C_ADDTO( res , a)\
    do {(res).r = ADD32((res).r, (a).r);  (res).i = ADD32((res).i,(a).i);\
    }while(0)

#define C_SUBFROM( res , a)\
    do {(res).r = ADD32((res).r,(a).r);  (res).i = SUB32((res).i,(a).i); \
    }while(0)

#if defined(ARMv4_ASM)

#undef C_MUL
#define C_MUL(m,a,b) \
    do{ \
       int br__; \
       int bi__; \
       int tt__; \
        __asm__ __volatile__( \
            "#C_MUL\n\t" \
            "ldm %[ap], {r0,r1}\n\t" \
            "ldrsh %[br], [%[bp], #0]\n\t" \
            "ldrsh %[bi], [%[bp], #2]\n\t" \
            "smull %[tt], %[mi], r1, %[br]\n\t" \
            "smlal %[tt], %[mi], r0, %[bi]\n\t" \
            "rsb %[bi], %[bi], #0\n\t" \
            "smull r0, %[mr], %[br], r0\n\t" \
            "mov %[tt], %[tt], lsr #15\n\t" \
            "smlal r0, %[mr], r1, %[bi]\n\t" \
            "orr %[mi], %[tt], %[mi], lsl #17\n\t" \
            "mov r0, r0, lsr #15\n\t" \
            "orr %[mr], r0, %[mr], lsl #17\n\t" \
            : [mr]"=r"((m).r), [mi]"=r"((m).i), \
              [br]"=&r"(br__), [bi]"=r"(bi__), [tt]"=r"(tt__) \
            : [ap]"r"(&(a)), [bp]"r"(&(b)) \
            : "r0", "r1" \
        ); \
    } \
    while(0)

#undef C_MUL4
#define C_MUL4(m,a,b) \
    do{ \
       int br__; \
       int bi__; \
       int tt__; \
        __asm__ __volatile__( \
            "#C_MUL4\n\t" \
            "ldm %[ap], {r0,r1}\n\t" \
            "ldrsh %[br], [%[bp], #0]\n\t" \
            "ldrsh %[bi], [%[bp], #2]\n\t" \
            "smull %[tt], %[mi], r1, %[br]\n\t" \
            "smlal %[tt], %[mi], r0, %[bi]\n\t" \
            "rsb %[bi], %[bi], #0\n\t" \
            "smull r0, %[mr], %[br], r0\n\t" \
            "mov %[tt], %[tt], lsr #17\n\t" \
            "smlal r0, %[mr], r1, %[bi]\n\t" \
            "orr %[mi], %[tt], %[mi], lsl #15\n\t" \
            "mov r0, r0, lsr #17\n\t" \
            "orr %[mr], r0, %[mr], lsl #15\n\t" \
            : [mr]"=r"((m).r), [mi]"=r"((m).i), \
              [br]"=&r"(br__), [bi]"=r"(bi__), [tt]"=r"(tt__) \
            : [ap]"r"(&(a)), [bp]"r"(&(b)) \
            : "r0", "r1" \
        ); \
    } \
    while(0)

#undef C_MULC
#define C_MULC(m,a,b) \
    do{ \
       int br__; \
       int bi__; \
       int tt__; \
        __asm__ __volatile__( \
            "#C_MULC\n\t" \
            "ldm %[ap], {r0,r1}\n\t" \
            "ldrsh %[br], [%[bp], #0]\n\t" \
            "ldrsh %[bi], [%[bp], #2]\n\t" \
            "smull %[tt], %[mr], r0, %[br]\n\t" \
            "smlal %[tt], %[mr], r1, %[bi]\n\t" \
            "rsb %[bi], %[bi], #0\n\t" \
            "smull r1, %[mi], %[br], r1\n\t" \
            "mov %[tt], %[tt], lsr #15\n\t" \
            "smlal r1, %[mi], r0, %[bi]\n\t" \
            "orr %[mr], %[tt], %[mr], lsl #17\n\t" \
            "mov r1, r1, lsr #15\n\t" \
            "orr %[mi], r1, %[mi], lsl #17\n\t" \
            : [mr]"=r"((m).r), [mi]"=r"((m).i), \
              [br]"=&r"(br__), [bi]"=r"(bi__), [tt]"=r"(tt__) \
            : [ap]"r"(&(a)), [bp]"r"(&(b)) \
            : "r0", "r1" \
        ); \
    } \
    while(0)

#endif /* ARMv4_ASM */

#if defined(ARMv5E_ASM)

#if defined(__thumb__)||defined(__thumb2__)
#define LDRD_CONS "Q"
#else
#define LDRD_CONS "Uq"
#endif

#undef C_MUL
#define C_MUL(m,a,b) \
    do{ \
        int mr1__; \
        int mr2__; \
        int mi__; \
        long long aval__; \
        int bval__; \
        __asm__( \
            "#C_MUL\n\t" \
            "ldrd %[aval], %H[aval], %[ap]\n\t" \
            "ldr %[bval], %[bp]\n\t" \
            "smulwb %[mi], %H[aval], %[bval]\n\t" \
            "smulwb %[mr1], %[aval], %[bval]\n\t" \
            "smulwt %[mr2], %H[aval], %[bval]\n\t" \
            "smlawt %[mi], %[aval], %[bval], %[mi]\n\t" \
            : [mr1]"=r"(mr1__), [mr2]"=r"(mr2__), [mi]"=r"(mi__), \
              [aval]"=&r"(aval__), [bval]"=r"(bval__) \
            : [ap]LDRD_CONS(a), [bp]"m"(b) \
        ); \
        (m).r = SHL32(SUB32(mr1__, mr2__), 1); \
        (m).i = SHL32(mi__, 1); \
    } \
    while(0)

#undef C_MUL4
#define C_MUL4(m,a,b) \
    do{ \
        int mr1__; \
        int mr2__; \
        int mi__; \
        long long aval__; \
        int bval__; \
        __asm__( \
            "#C_MUL4\n\t" \
            "ldrd %[aval], %H[aval], %[ap]\n\t" \
            "ldr %[bval], %[bp]\n\t" \
            "smulwb %[mi], %H[aval], %[bval]\n\t" \
            "smulwb %[mr1], %[aval], %[bval]\n\t" \
            "smulwt %[mr2], %H[aval], %[bval]\n\t" \
            "smlawt %[mi], %[aval], %[bval], %[mi]\n\t" \
            : [mr1]"=r"(mr1__), [mr2]"=r"(mr2__), [mi]"=r"(mi__), \
              [aval]"=&r"(aval__), [bval]"=r"(bval__) \
            : [ap]LDRD_CONS(a), [bp]"m"(b) \
        ); \
        (m).r = SHR32(SUB32(mr1__, mr2__), 1); \
        (m).i = SHR32(mi__, 1); \
    } \
    while(0)

#undef C_MULC
#define C_MULC(m,a,b) \
    do{ \
        int mr__; \
        int mi1__; \
        int mi2__; \
        long long aval__; \
        int bval__; \
        __asm__( \
            "#C_MULC\n\t" \
            "ldrd %[aval], %H[aval], %[ap]\n\t" \
            "ldr %[bval], %[bp]\n\t" \
            "smulwb %[mr], %[aval], %[bval]\n\t" \
            "smulwb %[mi1], %H[aval], %[bval]\n\t" \
            "smulwt %[mi2], %[aval], %[bval]\n\t" \
            "smlawt %[mr], %H[aval], %[bval], %[mr]\n\t" \
            : [mr]"=r"(mr__), [mi1]"=r"(mi1__), [mi2]"=r"(mi2__), \
              [aval]"=&r"(aval__), [bval]"=r"(bval__) \
            : [ap]LDRD_CONS(a), [bp]"m"(b) \
        ); \
        (m).r = SHL32(mr__, 1); \
        (m).i = SHL32(SUB32(mi1__, mi2__), 1); \
    } \
    while(0)

#endif /* ARMv5E_ASM */

#else  /* not FIXED_POINT*/

#   define S_MUL(a,b) ( (a)*(b) )
#define C_MUL(m,a,b) \
    do{ (m).r = (a).r*(b).r - (a).i*(b).i;\
        (m).i = (a).r*(b).i + (a).i*(b).r; }while(0)
#define C_MULC(m,a,b) \
    do{ (m).r = (a).r*(b).r + (a).i*(b).i;\
        (m).i = (a).i*(b).r - (a).r*(b).i; }while(0)

#define C_MUL4(m,a,b) C_MUL(m,a,b)

#   define C_FIXDIV(c,div) /* NOOP */
#   define C_MULBYSCALAR( c, s ) \
    do{ (c).r *= (s);\
        (c).i *= (s); }while(0)
#endif

#ifndef CHECK_OVERFLOW_OP
#  define CHECK_OVERFLOW_OP(a,op,b) /* noop */
#endif

#ifndef C_ADD
#define  C_ADD( res, a,b)\
    do { \
            CHECK_OVERFLOW_OP((a).r,+,(b).r)\
            CHECK_OVERFLOW_OP((a).i,+,(b).i)\
            (res).r=(a).r+(b).r;  (res).i=(a).i+(b).i; \
    }while(0)
#define  C_SUB( res, a,b)\
    do { \
            CHECK_OVERFLOW_OP((a).r,-,(b).r)\
            CHECK_OVERFLOW_OP((a).i,-,(b).i)\
            (res).r=(a).r-(b).r;  (res).i=(a).i-(b).i; \
    }while(0)
#define C_ADDTO( res , a)\
    do { \
            CHECK_OVERFLOW_OP((res).r,+,(a).r)\
            CHECK_OVERFLOW_OP((res).i,+,(a).i)\
            (res).r += (a).r;  (res).i += (a).i;\
    }while(0)

#define C_SUBFROM( res , a)\
    do {\
            CHECK_OVERFLOW_OP((res).r,-,(a).r)\
            CHECK_OVERFLOW_OP((res).i,-,(a).i)\
            (res).r -= (a).r;  (res).i -= (a).i; \
    }while(0)
#endif /* C_ADD defined */

#ifdef FIXED_POINT
/*#  define KISS_FFT_COS(phase)  TRIG_UPSCALE*floor(MIN(32767,MAX(-32767,.5+32768 * cos (phase))))
#  define KISS_FFT_SIN(phase)  TRIG_UPSCALE*floor(MIN(32767,MAX(-32767,.5+32768 * sin (phase))))*/
#  define KISS_FFT_COS(phase)  floor(.5+TWID_MAX*cos (phase))
#  define KISS_FFT_SIN(phase)  floor(.5+TWID_MAX*sin (phase))
#  define HALF_OF(x) ((x)>>1)
#elif defined(USE_SIMD)
#  define KISS_FFT_COS(phase) _mm_set1_ps( cos(phase) )
#  define KISS_FFT_SIN(phase) _mm_set1_ps( sin(phase) )
#  define HALF_OF(x) ((x)*_mm_set1_ps(.5f))
#else
#  define KISS_FFT_COS(phase) (kiss_fft_scalar) cos(phase)
#  define KISS_FFT_SIN(phase) (kiss_fft_scalar) sin(phase)
#  define HALF_OF(x) ((x)*.5f)
#endif

#define  kf_cexp(x,phase) \
        do{ \
                (x)->r = KISS_FFT_COS(phase);\
                (x)->i = KISS_FFT_SIN(phase);\
        }while(0)

#define  kf_cexp2(x,phase) \
   do{ \
      (x)->r = TRIG_UPSCALE*celt_cos_norm((phase));\
      (x)->i = TRIG_UPSCALE*celt_cos_norm((phase)-32768);\
}while(0)

#endif /* KISS_FFT_GUTS_H */
