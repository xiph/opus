/* Copyright (c) 2008-2011 Xiph.Org Foundation, Mozilla Corporation,
                           Gregory Maxwell
   Copyright (c) 2024 Arm Limited
   Written by Jean-Marc Valin, Gregory Maxwell, Timothy B. Terriberry,
   and Yunho Huh */
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

#ifndef CUSTOM_MODES
#define CUSTOM_MODES
#endif

#include <stdio.h>
#include <math.h>
#include "bands.h"
#include "cpu_support.h"
#include "float_cast.h"
#include "mathops.h"

#ifdef FIXED_POINT
#define WORD "%d"
#define FIX_INT_TO_DOUBLE(x,q) ((double)(x) / (double)(1L << q))
#define DOUBLE_TO_FIX_INT(x,q) (((double)x * (double)(1L << q)))
#else
#define WORD "%f"
#endif

int ret = 0;

void testdiv(void)
{
   opus_int32 i;
   for (i=1;i<=327670;i++)
   {
      double prod;
      opus_val32 val;
      val = celt_rcp(i);
#ifdef FIXED_POINT
      prod = (1./32768./65526.)*val*i;
#else
      prod = val*i;
#endif
      if (fabs(prod-1) > .00025)
      {
         fprintf (stderr, "div failed: 1/%d="WORD" (product = %f)\n", i, val, prod);
         ret = 1;
      }
   }
}

void testsqrt(void)
{
   opus_int32 i;
   for (i=1;i<=1000000000;i++)
   {
      double ratio;
      opus_val16 val;
      val = celt_sqrt(i);
      ratio = val/sqrt(i);
      if (fabs(ratio - 1) > .0005 && fabs(val-sqrt(i)) > 2)
      {
         fprintf (stderr, "sqrt failed: sqrt(%d)="WORD" (ratio = %f)\n", i, val, ratio);
         ret = 1;
      }
      i+= i>>10;
   }
}

void testbitexactcos(void)
{
   int i;
   opus_int32 min_d,max_d,last,chk;
   chk=max_d=0;
   last=min_d=32767;
   for(i=64;i<=16320;i++)
   {
      opus_int32 d;
      opus_int32 q=bitexact_cos(i);
      chk ^= q*i;
      d = last - q;
      if (d>max_d)max_d=d;
      if (d<min_d)min_d=d;
      last = q;
   }
   if ((chk!=89408644)||(max_d!=5)||(min_d!=0)||(bitexact_cos(64)!=32767)||
       (bitexact_cos(16320)!=200)||(bitexact_cos(8192)!=23171))
   {
      fprintf (stderr, "bitexact_cos failed\n");
      ret = 1;
   }
}

void testbitexactlog2tan(void)
{
   int i,fail;
   opus_int32 min_d,max_d,last,chk;
   fail=chk=max_d=0;
   last=min_d=15059;
   for(i=64;i<8193;i++)
   {
      opus_int32 d;
      opus_int32 mid=bitexact_cos(i);
      opus_int32 side=bitexact_cos(16384-i);
      opus_int32 q=bitexact_log2tan(mid,side);
      chk ^= q*i;
      d = last - q;
      if (q!=-1*bitexact_log2tan(side,mid))
        fail = 1;
      if (d>max_d)max_d=d;
      if (d<min_d)min_d=d;
      last = q;
   }
   if ((chk!=15821257)||(max_d!=61)||(min_d!=-2)||fail||
       (bitexact_log2tan(32767,200)!=15059)||(bitexact_log2tan(30274,12540)!=2611)||
       (bitexact_log2tan(23171,23171)!=0))
   {
      fprintf (stderr, "bitexact_log2tan failed\n");
      ret = 1;
   }
}

#ifndef FIXED_POINT
void testlog2(void)
{
   float x;
   float error_threshold = 2.2e-06;
   float max_error = 0;
   for (x=0.001f;x<1677700.0;x+=(x/8.0))
   {
      float error = fabs((1.442695040888963387*log(x))-celt_log2(x));
      if (max_error < error)
      {
         max_error = error;
      }

      if (error > error_threshold)
      {
         fprintf (stderr,
                  "celt_log2 failed: "
                  "fabs((1.442695040888963387*log(x))-celt_log2(x))>%15.25e "
                  "(x = %f, error = %15.25e)\n", error_threshold, x, error);
         ret = 1;
      }
   }
   fprintf (stdout, "celt_log2 max_error: %15.25e\n", max_error);
}

void testexp2(void)
{
   float x;
   float error_threshold = 2.3e-07;
   float max_error = 0;
   for (x=-11.0;x<24.0;x+=0.0007f)
   {
      float error = fabs(x-(1.442695040888963387*log(celt_exp2(x))));
      if (max_error < error)
      {
         max_error = error;
      }

      if (error > error_threshold)
      {
         fprintf (stderr,
                  "celt_exp2 failed: "
                  "fabs(x-(1.442695040888963387*log(celt_exp2(x))))>%15.25e "
                  "(x = %f, error = %15.25e)\n", error_threshold, x, error);
         ret = 1;
      }
   }
   fprintf (stdout, "celt_exp2 max_error: %15.25e\n", max_error);
}

void testexp2log2(void)
{
   float x;
   float error_threshold = 2.0e-06;
   float max_error = 0;
   for (x=-11.0;x<24.0;x+=0.0007f)
   {
      float error = fabs(x-(celt_log2(celt_exp2(x))));
      if (max_error < error)
      {
         max_error = error;
      }

      if (error > error_threshold)
      {
         fprintf (stderr,
                  "celt_log2/celt_exp2 failed: "
                  "fabs(x-(celt_log2(celt_exp2(x))))>%15.25e "
                  "(x = %f, error = %15.25e)\n", error_threshold, x, error);
         ret = 1;
      }
   }
   fprintf (stdout, "celt_exp2, celt_log2 max_error: %15.25e\n", max_error);
}
#else

void testlog2_db(void)
{
#if defined(ENABLE_QEXT)
   /* celt_log2_db test */
   float error = -1;
   float max_error = -2;
   float error_threshold = 2.e-07;
   opus_int32 x = 0;
   int q_input = 14;
   for (x = 8; x < 1073741824; x += (x >> 3))
   {
      error = fabs((1.442695040888963387*log(FIX_INT_TO_DOUBLE(x, q_input))) -
                   FIX_INT_TO_DOUBLE(celt_log2_db(x), DB_SHIFT));
      if (error > max_error)
      {
         max_error = error;
      }
      if (error > error_threshold)
      {
         fprintf(stderr, "celt_log2_db failed: error: [%.5e > %.5e] (x = %f)\n",
                 error, error_threshold, FIX_INT_TO_DOUBLE(x, DB_SHIFT));
         ret = 1;
      }
   }
   fprintf(stdout, "celt_log2_db max_error: %.7e\n", max_error);
#endif  /* defined(ENABLE_QEXT) */
}

void testlog2(void)
{
   opus_val32 x;
   for (x=8;x<1073741824;x+=(x>>3))
   {
      float error = fabs((1.442695040888963387*log(x/16384.0))-celt_log2(x)/1024.0);
      if (error>0.003)
      {
         fprintf (stderr, "celt_log2 failed: x = %ld, error = %f\n", (long)x,error);
         ret = 1;
      }
   }
}

void testexp2(void)
{
   opus_val16 x;
   for (x=-32768;x<15360;x++)
   {
      float error1 = fabs(x/1024.0-(1.442695040888963387*log(celt_exp2(x)/65536.0)));
      float error2 = fabs(exp(0.6931471805599453094*x/1024.0)-celt_exp2(x)/65536.0);
      if (error1>0.0002&&error2>0.00004)
      {
         fprintf (stderr, "celt_exp2 failed: x = "WORD", error1 = %f, error2 = %f\n", x,error1,error2);
         ret = 1;
      }
   }
}

void testexp2_db(void)
{
#if defined(ENABLE_QEXT)
   float absolute_error = -1;
   float absolute_error_threshold = FIX_INT_TO_DOUBLE(2, 16);
   float relative_error_threshold = -2;
   float fx;
   float quantized_fx;
   opus_val32 x_32;

   for (fx = -32.0; fx < 15.0; fx += 0.0007)
   {
      double ground_truth;
      x_32 = DOUBLE_TO_FIX_INT(fx, DB_SHIFT);
      quantized_fx = FIX_INT_TO_DOUBLE(x_32, DB_SHIFT);

      ground_truth = (exp(0.6931471805599453094 * quantized_fx));
      absolute_error = fabs(ground_truth -
                            FIX_INT_TO_DOUBLE(celt_exp2_db(x_32), 16));

      relative_error_threshold = 1.24e-7 * ground_truth;
      if (absolute_error > absolute_error_threshold &&
          absolute_error > relative_error_threshold)
      {
         fprintf(stderr,
                 "celt_exp2_db failed: "
                 "absolute_error: [%.5e > %.5e] "
                 "relative_error: [%.5e > %.5e] (x = %f)\n",
                 absolute_error, absolute_error_threshold,
                 absolute_error, relative_error_threshold, quantized_fx);
         ret = 1;
      }
   }
#endif  /* defined(ENABLE_QEXT) */
}

void testexp2log2(void)
{
   opus_val32 x;
   for (x=8;x<65536;x+=(x>>3))
   {
      float error = fabs(x-0.25*celt_exp2(celt_log2(x)))/16384;
      if (error>0.004)
      {
         fprintf (stderr, "celt_log2/celt_exp2 failed: fabs(x-(celt_exp2(celt_log2(x))))>0.001 (x = %ld, error = %f)\n", (long)x,error);
         ret = 1;
      }
   }
}

void testilog2(void)
{
   opus_val32 x;
   for (x=1;x<=268435455;x+=127)
   {
      opus_val32 lg;
      opus_val32 y;

      lg = celt_ilog2(x);
      if (lg<0 || lg>=31)
      {
         printf("celt_ilog2 failed: 0<=celt_ilog2(x)<31 (x = %d, celt_ilog2(x) = %d)\n",x,lg);
         ret = 1;
      }
      y = 1<<lg;

      if (x<y || (x>>1)>=y)
      {
         printf("celt_ilog2 failed: 2**celt_ilog2(x)<=x<2**(celt_ilog2(x)+1) (x = %d, 2**celt_ilog2(x) = %d)\n",x,y);
         ret = 1;
      }
   }
}
#endif


#ifndef DISABLE_FLOAT_API

void testcelt_float2int16(int use_ref_impl, int buffer_size)
{

#define MAX_BUFFER_SIZE 2080
   int i, cnt;
   float floatsToConvert[MAX_BUFFER_SIZE];
   short results[MAX_BUFFER_SIZE] = { 0 };
   float scaleInt16RangeTo01;

   celt_assert(buffer_size <= MAX_BUFFER_SIZE);

   scaleInt16RangeTo01 = 1.f / 32768.f;
   cnt = 0;

   while (cnt + 15 < buffer_size && cnt < buffer_size / 2)
   {
      floatsToConvert[cnt++] = 77777.0f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = 33000.0f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = 32768.0f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = 32767.4f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = 32766.6f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = .501 * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = .499f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = .0f;
      floatsToConvert[cnt++] = -.499f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = -.501f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = -32767.6f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = -32768.4f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = -32769.0f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = -33000.0f * scaleInt16RangeTo01;
      floatsToConvert[cnt++] = -77777.0f * scaleInt16RangeTo01;

      celt_assert(cnt < buffer_size);
   }

   while (cnt < buffer_size)
   {
      float inInt16Range = cnt * 7 + .5;
      inInt16Range += (cnt & 0x01) ? .1 : -.1;
      inInt16Range *= (cnt & 0x02) ? 1 : -1;
      floatsToConvert[cnt++] = inInt16Range * scaleInt16RangeTo01;
   }

   for (i = 0; i < MAX_BUFFER_SIZE; ++i)
   {
      results[i] = 42;
   }

   if (use_ref_impl)
   {
      celt_float2int16_c(floatsToConvert, results, cnt);
   } else {
      celt_float2int16(floatsToConvert, results, cnt, opus_select_arch());
   }

   for (i = 0; i < cnt; ++i)
   {
      const float expected = FLOAT2INT16(floatsToConvert[i]);
      if (results[i] != expected)
      {
         fprintf (stderr, "testcelt_float2int16 failed: celt_float2int16 converted %f (index: %d) to %d (x*32768=%f, expected: %d, cnt: %d, ref: %d)\n",
               floatsToConvert[i], i, (int)results[i], floatsToConvert[i] * 32768.0f, (int)expected, buffer_size, use_ref_impl);
         ret = 1;
      }
   }

   for (i = cnt; i < MAX_BUFFER_SIZE; ++i)
   {
      if (results[i] != 42)
      {
         fprintf (stderr, "testcelt_float2int16 failed: buffer overflow (cnt: %d, ref: %d)\n", buffer_size, use_ref_impl);
         ret = 1;
         break;
      }
   }
#undef MAX_BUFFER_SIZE
}

void testopus_limit2_checkwithin1(int use_ref_impl)
{
#define BUFFER_SIZE 37 /* strange float count to trigger residue loop of SIMD implementation */
#define BYTE_COUNT (BUFFER_SIZE * sizeof(float))
   int i, within1;
   const int arch = opus_select_arch();

   float pattern[BUFFER_SIZE], buffer[BUFFER_SIZE];

   for (i = 0; i < BUFFER_SIZE; ++i)
   {
      pattern[i] = i % 2 ? -1.f : 1.f;
   }

   /* All values within -1..1:
   Nothing changed. Return value is implementation-dependent (not expected to recognise nothing exceeds -1..1) */
   memcpy(buffer, pattern, BYTE_COUNT);
   within1 = use_ref_impl ? opus_limit2_checkwithin1_c(buffer, BUFFER_SIZE) : opus_limit2_checkwithin1(buffer, BUFFER_SIZE, arch);
   if (memcmp(buffer, pattern, BYTE_COUNT) != 0)
   {
      fprintf (stderr, "opus_limit2_checkwithin1() modified values not exceeding -1..1 (ref=%d)\n", use_ref_impl);
      ret = 1;
   }

   /* One value exceeds -1..1, within -2..2:
   Values unchanged. Return value says not all values are within -1..1 */
   for (i = 0; i < BUFFER_SIZE; ++i)
   {
      const float replace_value = pattern[i] * 1.001f;

      memcpy(buffer, pattern, BYTE_COUNT);
      buffer[i] = replace_value;
      within1 = use_ref_impl ? opus_limit2_checkwithin1_c(buffer, BUFFER_SIZE) : opus_limit2_checkwithin1(buffer, BUFFER_SIZE, arch);
      if (within1 || buffer[i] != replace_value)
      {
         fprintf (stderr, "opus_limit2_checkwithin1() handled value exceeding -1..1 erroneously (ref=%d, i=%d)\n", use_ref_impl, i);
         ret = 1;
      }
      buffer[i] = pattern[i];
      if (memcmp(buffer, pattern, BYTE_COUNT) != 0)
      {
         fprintf (stderr, "opus_limit2_checkwithin1() modified value within -2..2  (ref=%d, i=%d)\n", use_ref_impl, i);
         ret = 1;
      }
   }

   /* One value exceeds -2..2:
   One value is hardclipped, others are unchanged. Return value says not all values are within -1..1 */
   for (i = 0; i < BUFFER_SIZE; ++i)
   {
      const float replace_value = pattern[i] * 2.1;

      memcpy(buffer, pattern, BYTE_COUNT);
      buffer[i] = replace_value;
      within1 = use_ref_impl ? opus_limit2_checkwithin1_c(buffer, BUFFER_SIZE) : opus_limit2_checkwithin1(buffer, BUFFER_SIZE, arch);
      if (within1 || buffer[i] != (replace_value > 0.f ? 2.f : -2.f))
      {
         fprintf (stderr, "opus_limit2_checkwithin1() handled value exceeding -2..2 erroneously (ref=%d, i=%d)\n", use_ref_impl, i);
         ret = 1;
      }
      buffer[i] = pattern[i];
      if (memcmp(buffer, pattern, BYTE_COUNT) != 0)
      {
         fprintf (stderr, "opus_limit2_checkwithin1() modified value within -2..2  (ref=%d, i=%d)\n", use_ref_impl, i);
         ret = 1;
      }
   }
#undef BUFFER_SIZE
#undef BYTE_COUNT
}

#endif

int main(void)
{
   int i;
   int use_ref_impl[2] = { 0, 1 };

   testbitexactcos();
   testbitexactlog2tan();
   testdiv();
   testsqrt();
   testlog2();
   testexp2();
   testexp2log2();
#ifdef FIXED_POINT
   testilog2();
   testlog2_db();
   testexp2_db();
#endif
#ifndef DISABLE_FLOAT_API
   for (i = 0; i <= 1; ++i)
   {
      testcelt_float2int16(use_ref_impl[i], 1);
      testcelt_float2int16(use_ref_impl[i], 32);
      testcelt_float2int16(use_ref_impl[i], 127);
      testcelt_float2int16(use_ref_impl[i], 1031);
      testopus_limit2_checkwithin1(use_ref_impl[i]);
   }
#endif
   return ret;
}
