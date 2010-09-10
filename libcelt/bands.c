/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2008-2009 Gregory Maxwell 
   Written by Jean-Marc Valin and Gregory Maxwell */
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
#include "bands.h"
#include "modes.h"
#include "vq.h"
#include "cwrs.h"
#include "stack_alloc.h"
#include "os_support.h"
#include "mathops.h"
#include "rate.h"

/* This is a cos() approximation designed to be bit-exact on any platform. Bit exactness
   with this approximation is important because it has an impact on the bit allocation */
static celt_int16 bitexact_cos(celt_int16 x)
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


#ifdef FIXED_POINT
/* Compute the amplitude (sqrt energy) in each of the bands */
void compute_band_energies(const CELTMode *m, const celt_sig *X, celt_ener *bank, int end, int _C, int M)
{
   int i, c, N;
   const celt_int16 *eBands = m->eBands;
   const int C = CHANNELS(_C);
   N = M*m->shortMdctSize;
   for (c=0;c<C;c++)
   {
      for (i=0;i<end;i++)
      {
         int j;
         celt_word32 maxval=0;
         celt_word32 sum = 0;
         
         j=M*eBands[i]; do {
            maxval = MAX32(maxval, X[j+c*N]);
            maxval = MAX32(maxval, -X[j+c*N]);
         } while (++j<M*eBands[i+1]);
         
         if (maxval > 0)
         {
            int shift = celt_ilog2(maxval)-10;
            j=M*eBands[i]; do {
               sum = MAC16_16(sum, EXTRACT16(VSHR32(X[j+c*N],shift)),
                                   EXTRACT16(VSHR32(X[j+c*N],shift)));
            } while (++j<M*eBands[i+1]);
            /* We're adding one here to make damn sure we never end up with a pitch vector that's
               larger than unity norm */
            bank[i+c*m->nbEBands] = EPSILON+VSHR32(EXTEND32(celt_sqrt(sum)),-shift);
         } else {
            bank[i+c*m->nbEBands] = EPSILON;
         }
         /*printf ("%f ", bank[i+c*m->nbEBands]);*/
      }
   }
   /*printf ("\n");*/
}

/* Normalise each band such that the energy is one. */
void normalise_bands(const CELTMode *m, const celt_sig * restrict freq, celt_norm * restrict X, const celt_ener *bank, int end, int _C, int M)
{
   int i, c, N;
   const celt_int16 *eBands = m->eBands;
   const int C = CHANNELS(_C);
   N = M*m->shortMdctSize;
   for (c=0;c<C;c++)
   {
      i=0; do {
         celt_word16 g;
         int j,shift;
         celt_word16 E;
         shift = celt_zlog2(bank[i+c*m->nbEBands])-13;
         E = VSHR32(bank[i+c*m->nbEBands], shift);
         g = EXTRACT16(celt_rcp(SHL32(E,3)));
         j=M*eBands[i]; do {
            X[j+c*N] = MULT16_16_Q15(VSHR32(freq[j+c*N],shift-1),g);
         } while (++j<M*eBands[i+1]);
      } while (++i<end);
   }
}

#else /* FIXED_POINT */
/* Compute the amplitude (sqrt energy) in each of the bands */
void compute_band_energies(const CELTMode *m, const celt_sig *X, celt_ener *bank, int end, int _C, int M)
{
   int i, c, N;
   const celt_int16 *eBands = m->eBands;
   const int C = CHANNELS(_C);
   N = M*m->shortMdctSize;
   for (c=0;c<C;c++)
   {
      for (i=0;i<end;i++)
      {
         int j;
         celt_word32 sum = 1e-10f;
         for (j=M*eBands[i];j<M*eBands[i+1];j++)
            sum += X[j+c*N]*X[j+c*N];
         bank[i+c*m->nbEBands] = celt_sqrt(sum);
         /*printf ("%f ", bank[i+c*m->nbEBands]);*/
      }
   }
   /*printf ("\n");*/
}

/* Normalise each band such that the energy is one. */
void normalise_bands(const CELTMode *m, const celt_sig * restrict freq, celt_norm * restrict X, const celt_ener *bank, int end, int _C, int M)
{
   int i, c, N;
   const celt_int16 *eBands = m->eBands;
   const int C = CHANNELS(_C);
   N = M*m->shortMdctSize;
   for (c=0;c<C;c++)
   {
      for (i=0;i<end;i++)
      {
         int j;
         celt_word16 g = 1.f/(1e-10f+bank[i+c*m->nbEBands]);
         for (j=M*eBands[i];j<M*eBands[i+1];j++)
            X[j+c*N] = freq[j+c*N]*g;
      }
   }
}

#endif /* FIXED_POINT */

void renormalise_bands(const CELTMode *m, celt_norm * restrict X, int end, int _C, int M)
{
   int i, c;
   const celt_int16 *eBands = m->eBands;
   const int C = CHANNELS(_C);
   for (c=0;c<C;c++)
   {
      i=0; do {
         renormalise_vector(X+M*eBands[i]+c*M*m->shortMdctSize, Q15ONE, M*eBands[i+1]-M*eBands[i], 1);
      } while (++i<end);
   }
}

/* De-normalise the energy to produce the synthesis from the unit-energy bands */
void denormalise_bands(const CELTMode *m, const celt_norm * restrict X, celt_sig * restrict freq, const celt_ener *bank, int end, int _C, int M)
{
   int i, c, N;
   const celt_int16 *eBands = m->eBands;
   const int C = CHANNELS(_C);
   N = M*m->shortMdctSize;
   celt_assert2(C<=2, "denormalise_bands() not implemented for >2 channels");
   for (c=0;c<C;c++)
   {
      celt_sig * restrict f;
      const celt_norm * restrict x;
      f = freq+c*N;
      x = X+c*N;
      for (i=0;i<end;i++)
      {
         int j, band_end;
         celt_word32 g = SHR32(bank[i+c*m->nbEBands],1);
         j=M*eBands[i];
         band_end = M*eBands[i+1];
         do {
            *f++ = SHL32(MULT16_32_Q15(*x, g),2);
            x++;
         } while (++j<band_end);
      }
      for (i=M*eBands[m->nbEBands];i<N;i++)
         *f++ = 0;
   }
}

static void stereo_band_mix(const CELTMode *m, celt_norm *X, celt_norm *Y, const celt_ener *bank, int stereo_mode, int bandID, int dir, int N)
{
   int i = bandID;
   int j;
   celt_word16 a1, a2;
   if (stereo_mode==0)
   {
      /* Do mid-side when not doing intensity stereo */
      a1 = QCONST16(.70711f,14);
      a2 = dir*QCONST16(.70711f,14);
   } else {
      celt_word16 left, right;
      celt_word16 norm;
#ifdef FIXED_POINT
      int shift = celt_zlog2(MAX32(bank[i], bank[i+m->nbEBands]))-13;
#endif
      left = VSHR32(bank[i],shift);
      right = VSHR32(bank[i+m->nbEBands],shift);
      norm = EPSILON + celt_sqrt(EPSILON+MULT16_16(left,left)+MULT16_16(right,right));
      a1 = DIV32_16(SHL32(EXTEND32(left),14),norm);
      a2 = dir*DIV32_16(SHL32(EXTEND32(right),14),norm);
   }
   for (j=0;j<N;j++)
   {
      celt_norm r, l;
      l = X[j];
      r = Y[j];
      X[j] = MULT16_16_Q14(a1,l) + MULT16_16_Q14(a2,r);
      Y[j] = MULT16_16_Q14(a1,r) - MULT16_16_Q14(a2,l);
   }
}

/* Decide whether we should spread the pulses in the current frame */
int folding_decision(const CELTMode *m, celt_norm *X, int *average, int *last_decision, int end, int _C, int M)
{
   int i, c, N0;
   int sum = 0, nbBands=0;
   const int C = CHANNELS(_C);
   const celt_int16 * restrict eBands = m->eBands;
   int decision;
   
   N0 = M*m->shortMdctSize;

   if (M*(eBands[end]-eBands[end-1]) <= 8)
      return 0;
   for (c=0;c<C;c++)
   {
      for (i=0;i<end;i++)
      {
         int j, N, tmp=0;
         int tcount[3] = {0};
         celt_norm * restrict x = X+M*eBands[i]+c*N0;
         N = M*(eBands[i+1]-eBands[i]);
         if (N<=8)
            continue;
         /* Compute rough CDF of |x[j]| */
         for (j=0;j<N;j++)
         {
            celt_word32 x2N; /* Q13 */

            x2N = MULT16_16(MULT16_16_Q15(x[j], x[j]), N);
            if (x2N < QCONST16(0.25f,13))
               tcount[0]++;
            if (x2N < QCONST16(0.0625f,13))
               tcount[1]++;
            if (x2N < QCONST16(0.015625f,13))
               tcount[2]++;
         }

         tmp = (2*tcount[2] >= N) + (2*tcount[1] >= N) + (2*tcount[0] >= N);
         sum += tmp*256;
         nbBands++;
      }
   }
   sum /= nbBands;
   /* Recursive averaging */
   sum = (sum+*average)>>1;
   *average = sum;
   /* Hysteresis */
   sum = (3*sum + ((*last_decision<<7) + 64) + 2)>>2;
   /* decision and last_decision do not use the same encoding */
   if (sum < 128)
   {
      decision = 2;
      *last_decision = 0;
   } else if (sum < 256)
   {
      decision = 1;
      *last_decision = 1;
   } else if (sum < 384)
   {
      decision = 3;
      *last_decision = 2;
   } else {
      decision = 0;
      *last_decision = 3;
   }
   return decision;
}

#ifdef MEASURE_NORM_MSE

float MSE[30] = {0};
int nbMSEBands = 0;
int MSECount[30] = {0};

void dump_norm_mse(void)
{
   int i;
   for (i=0;i<nbMSEBands;i++)
   {
      printf ("%g ", MSE[i]/MSECount[i]);
   }
   printf ("\n");
}

void measure_norm_mse(const CELTMode *m, float *X, float *X0, float *bandE, float *bandE0, int M, int N, int C)
{
   static int init = 0;
   int i;
   if (!init)
   {
      atexit(dump_norm_mse);
      init = 1;
   }
   for (i=0;i<m->nbEBands;i++)
   {
      int j;
      int c;
      float g;
      if (bandE0[i]<10 || (C==2 && bandE0[i+m->nbEBands]<1))
         continue;
      for (c=0;c<C;c++)
      {
         g = bandE[i+c*m->nbEBands]/(1e-15+bandE0[i+c*m->nbEBands]);
         for (j=M*m->eBands[i];j<M*m->eBands[i+1];j++)
            MSE[i] += (g*X[j+c*N]-X0[j+c*N])*(g*X[j+c*N]-X0[j+c*N]);
      }
      MSECount[i]+=C;
   }
   nbMSEBands = m->nbEBands;
}

#endif

static void interleave_vector(celt_norm *X, int N0, int stride)
{
   int i,j;
   VARDECL(celt_norm, tmp);
   int N;
   SAVE_STACK;
   N = N0*stride;
   ALLOC(tmp, N, celt_norm);
   for (i=0;i<stride;i++)
      for (j=0;j<N0;j++)
         tmp[j*stride+i] = X[i*N0+j];
   for (j=0;j<N;j++)
      X[j] = tmp[j];
   RESTORE_STACK;
}

static void deinterleave_vector(celt_norm *X, int N0, int stride)
{
   int i,j;
   VARDECL(celt_norm, tmp);
   int N;
   SAVE_STACK;
   N = N0*stride;
   ALLOC(tmp, N, celt_norm);
   for (i=0;i<stride;i++)
      for (j=0;j<N0;j++)
         tmp[i*N0+j] = X[j*stride+i];
   for (j=0;j<N;j++)
      X[j] = tmp[j];
   RESTORE_STACK;
}

static void haar1(celt_norm *X, int N0, int stride)
{
   int i, j;
   N0 >>= 1;
   for (i=0;i<stride;i++)
      for (j=0;j<N0;j++)
      {
         celt_norm tmp = X[stride*2*j+i];
         X[stride*2*j+i] = MULT16_16_Q15(QCONST16(.7070678f,15), X[stride*2*j+i] + X[stride*(2*j+1)+i]);
         X[stride*(2*j+1)+i] = MULT16_16_Q15(QCONST16(.7070678f,15), tmp - X[stride*(2*j+1)+i]);
      }
}

static int compute_qn(int N, int b, int offset, int stereo)
{
   static const celt_int16 exp2_table8[8] =
      {16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048};
   int qn, qb;
   int N2 = 2*N-1;
   if (stereo && N==2)
      N2--;
   qb = (b+N2*offset)/N2;
   if (qb > (b>>1)-(1<<BITRES))
      qb = (b>>1)-(1<<BITRES);

   if (qb<0)
       qb = 0;
   if (qb>14<<BITRES)
     qb = 14<<BITRES;

   if (qb<(1<<BITRES>>1)) {
      qn = 1;
   } else {
      qn = exp2_table8[qb&0x7]>>(14-(qb>>BITRES));
      qn = (qn+1)>>1<<1;
      if (qn>1024)
         qn = 1024;
   }
   return qn;
}


/* This function is responsible for encoding and decoding a band for both
   the mono and stereo case. Even in the mono case, it can split the band
   in two and transmit the energy difference with the two half-bands. It
   can be called recursively so bands can end up being split in 8 parts. */
static void quant_band(int encode, const CELTMode *m, int i, celt_norm *X, celt_norm *Y,
      int N, int b, int spread, int B, int tf_change, celt_norm *lowband, int resynth, void *ec,
      celt_int32 *remaining_bits, int LM, celt_norm *lowband_out, const celt_ener *bandE, int level, celt_int32 *seed)
{
   int q;
   int curr_bits;
   int stereo, split;
   int imid=0, iside=0;
   int N0=N;
   int N_B=N;
   int N_B0;
   int B0=B;
   int time_divide=0;
   int recombine=0;

   N_B /= B;
   N_B0 = N_B;

   split = stereo = Y != NULL;

   /* Special case for one sample */
   if (N==1)
   {
      int c;
      celt_norm *x = X;
      for (c=0;c<1+stereo;c++)
      {
         int sign=0;
         if (b>=1<<BITRES && *remaining_bits>=1<<BITRES)
         {
            if (encode)
            {
               sign = x[0]<0;
               ec_enc_bits((ec_enc*)ec, sign, 1);
            } else {
               sign = ec_dec_bits((ec_dec*)ec, 1);
            }
            *remaining_bits -= 1<<BITRES;
            b-=1<<BITRES;
         }
         if (resynth)
            x[0] = sign ? -NORM_SCALING : NORM_SCALING;
         x = Y;
      }
      if (lowband_out)
         lowband_out[0] = X[0];
      return;
   }

   /* Band recombining to increase frequency resolution */
   if (!stereo && B > 1 && level == 0 && tf_change>0)
   {
      while (B>1 && tf_change>0)
      {
         B>>=1;
         N_B<<=1;
         if (encode)
            haar1(X, N_B, B);
         if (lowband)
            haar1(lowband, N_B, B);
         recombine++;
         tf_change--;
      }
      B0=B;
      N_B0 = N_B;
   }

   /* Increasing the time resolution */
   if (!stereo && level==0)
   {
      while ((N_B&1) == 0 && tf_change<0 && B <= (1<<LM))
      {
         if (encode)
            haar1(X, N_B, B);
         if (lowband)
            haar1(lowband, N_B, B);
         B <<= 1;
         N_B >>= 1;
         time_divide++;
         tf_change++;
      }
      B0 = B;
      N_B0 = N_B;
   }

   /* Reorganize the samples in time order instead of frequency order */
   if (!stereo && B0>1 && level==0)
   {
      if (encode)
         deinterleave_vector(X, N_B, B0);
      if (lowband)
         deinterleave_vector(lowband, N_B, B0);
   }

   /* If we need more than 32 bits, try splitting the band in two. */
   if (!stereo && LM != -1 && b > 32<<BITRES && N>2)
   {
      if (LM>0 || (N&1)==0)
      {
         N >>= 1;
         Y = X+N;
         split = 1;
         LM -= 1;
         B = (B+1)>>1;
      }
   }

   if (split)
   {
      int qn;
      int itheta=0;
      int mbits, sbits, delta;
      int qalloc;
      celt_word16 mid, side;
      int offset;

      /* Decide on the resolution to give to the split parameter theta */
      offset = ((m->logN[i]+(LM<<BITRES))>>1) - (stereo ? QTHETA_OFFSET_STEREO : QTHETA_OFFSET);
      qn = compute_qn(N, b, offset, stereo);

      qalloc = 0;
      if (qn!=1)
      {
         if (encode)
         {
            if (stereo)
               stereo_band_mix(m, X, Y, bandE, 0, i, 1, N);

            mid = renormalise_vector(X, Q15ONE, N, 1);
            side = renormalise_vector(Y, Q15ONE, N, 1);

            /* theta is the atan() of the ratio between the (normalized)
               side and mid. With just that parameter, we can re-scale both
               mid and side because we know that 1) they have unit norm and
               2) they are orthogonal. */
   #ifdef FIXED_POINT
            /* 0.63662 = 2/pi */
            itheta = MULT16_16_Q15(QCONST16(0.63662f,15),celt_atan2p(side, mid));
   #else
            itheta = (int)floor(.5f+16384*0.63662f*atan2(side,mid));
   #endif

            itheta = (itheta*qn+8192)>>14;
         }

         /* Entropy coding of the angle. We use a uniform pdf for the
            first stereo split but a triangular one for the rest. */
         if (stereo || qn>256 || B>1)
         {
            if (encode)
               ec_enc_uint((ec_enc*)ec, itheta, qn+1);
            else
               itheta = ec_dec_uint((ec_dec*)ec, qn+1);
            qalloc = log2_frac(qn+1,BITRES);
         } else {
            int fs=1, ft;
            ft = ((qn>>1)+1)*((qn>>1)+1);
            if (encode)
            {
               int fl;

               fs = itheta <= (qn>>1) ? itheta + 1 : qn + 1 - itheta;
               fl = itheta <= (qn>>1) ? itheta*(itheta + 1)>>1 :
                ft - ((qn + 1 - itheta)*(qn + 2 - itheta)>>1);

               ec_encode((ec_enc*)ec, fl, fl+fs, ft);
            } else {
               int fl=0;
               int fm;
               fm = ec_decode((ec_dec*)ec, ft);

               if (fm < ((qn>>1)*((qn>>1) + 1)>>1))
               {
                  itheta = (isqrt32(8*(celt_uint32)fm + 1) - 1)>>1;
                  fs = itheta + 1;
                  fl = itheta*(itheta + 1)>>1;
               }
               else
               {
                  itheta = (2*(qn + 1)
                   - isqrt32(8*(celt_uint32)(ft - fm - 1) + 1))>>1;
                  fs = qn + 1 - itheta;
                  fl = ft - ((qn + 1 - itheta)*(qn + 2 - itheta)>>1);
               }

               ec_dec_update((ec_dec*)ec, fl, fl+fs, ft);
            }
            qalloc = log2_frac(ft,BITRES) - log2_frac(fs,BITRES) + 1;
         }
         itheta = itheta*16384/qn;
      } else {
         if (stereo && encode)
            stereo_band_mix(m, X, Y, bandE, 1, i, 1, N);
      }

      if (itheta == 0)
      {
         imid = 32767;
         iside = 0;
         delta = -10000;
      } else if (itheta == 16384)
      {
         imid = 0;
         iside = 32767;
         delta = 10000;
      } else {
         imid = bitexact_cos(itheta);
         iside = bitexact_cos(16384-itheta);
         /* This is the mid vs side allocation that minimizes squared error
            in that band. */
         delta = (N-1)*(log2_frac(iside,BITRES+2)-log2_frac(imid,BITRES+2))>>2;
      }

      /* This is a special case for N=2 that only works for stereo and takes
         advantage of the fact that mid and side are orthogonal to encode
         the side with just one bit. */
      if (N==2 && stereo)
      {
         int c;
         int sign=1;
         celt_norm *x2, *y2;
         mbits = b-qalloc;
         sbits = 0;
         /* Only need one bit for the side */
         if (itheta != 0 && itheta != 16384)
            sbits = 1<<BITRES;
         mbits -= sbits;
         c = itheta > 8192;
         *remaining_bits -= qalloc+sbits;

         x2 = c ? Y : X;
         y2 = c ? X : Y;
         if (sbits)
         {
            if (encode)
            {
               /* Here we only need to encode a sign for the side */
               sign = x2[0]*y2[1] - x2[1]*y2[0] > 0;
               ec_enc_bits((ec_enc*)ec, sign, 1);
            } else {
               sign = ec_dec_bits((ec_dec*)ec, 1);
            }
         }
         sign = 2*sign - 1;
         quant_band(encode, m, i, x2, NULL, N, mbits, spread, B, tf_change, lowband, resynth, ec, remaining_bits, LM, lowband_out, NULL, level+1, seed);
         y2[0] = -sign*x2[1];
         y2[1] = sign*x2[0];
      } else {
         /* "Normal" split code */
         celt_norm *next_lowband2=NULL;
         celt_norm *next_lowband_out1=NULL;
         int next_level=0;

         /* Give more bits to low-energy MDCTs than they would otherwise deserve */
         if (B>1 && !stereo)
            delta >>= 1;

         mbits = (b-qalloc-delta)/2;
         if (mbits > b-qalloc)
            mbits = b-qalloc;
         if (mbits<0)
            mbits=0;
         sbits = b-qalloc-mbits;
         *remaining_bits -= qalloc;

         if (lowband && !stereo)
            next_lowband2 = lowband+N; /* >32-bit split case */

         /* Only stereo needs to pass on lowband_out. Otherwise, it's handled at the end */
         if (stereo)
            next_lowband_out1 = lowband_out;
         else
            next_level = level+1;

         quant_band(encode, m, i, X, NULL, N, mbits, spread, B, tf_change, lowband, resynth, ec, remaining_bits, LM, next_lowband_out1, NULL, next_level, seed);
         quant_band(encode, m, i, Y, NULL, N, sbits, spread, B, tf_change, next_lowband2, resynth, ec, remaining_bits, LM, NULL, NULL, level, seed);
      }

   } else {
      /* This is the basic no-split case */
      q = bits2pulses(m, i, LM, b);
      curr_bits = pulses2bits(m, i, LM, q);
      *remaining_bits -= curr_bits;

      /* Ensures we can never bust the budget */
      while (*remaining_bits < 0 && q > 0)
      {
         *remaining_bits += curr_bits;
         q--;
         curr_bits = pulses2bits(m, i, LM, q);
         *remaining_bits -= curr_bits;
      }

      /* Finally do the actual quantization */
      if (encode)
         alg_quant(X, N, q, spread, B, lowband, resynth, (ec_enc*)ec, seed);
      else
         alg_unquant(X, N, q, spread, B, lowband, (ec_dec*)ec, seed);
   }

   /* This code is used by the decoder and by the resynthesis-enabled encoder */
   if (resynth)
   {
      int k;

      if (split)
      {
         int j;
         celt_word16 mid, side;
#ifdef FIXED_POINT
         mid = imid;
         side = iside;
#else
         mid = (1.f/32768)*imid;
         side = (1.f/32768)*iside;
#endif
         for (j=0;j<N;j++)
            X[j] = MULT16_16_Q15(X[j], mid);
         for (j=0;j<N;j++)
            Y[j] = MULT16_16_Q15(Y[j], side);
      }

      /* Undo the sample reorganization going from time order to frequency order */
      if (!stereo && B0>1 && level==0)
      {
         interleave_vector(X, N_B, B0);
         if (lowband)
            interleave_vector(lowband, N_B, B0);
      }

      /* Undo time-freq changes that we did earlier */
      N_B = N_B0;
      B = B0;
      for (k=0;k<time_divide;k++)
      {
         B >>= 1;
         N_B <<= 1;
         haar1(X, N_B, B);
         if (lowband)
            haar1(lowband, N_B, B);
      }

      for (k=0;k<recombine;k++)
      {
         haar1(X, N_B, B);
         if (lowband)
            haar1(lowband, N_B, B);
         N_B>>=1;
         B <<= 1;
      }

      /* Scale output for later folding */
      if (lowband_out && !stereo)
      {
         int j;
         celt_word16 n;
         n = celt_sqrt(SHL32(EXTEND32(N0),22));
         for (j=0;j<N0;j++)
            lowband_out[j] = MULT16_16_Q15(n,X[j]);
      }

      if (stereo)
      {
         stereo_band_mix(m, X, Y, bandE, 0, i, -1, N);
         /* We only need to renormalize because quantization may not
            have preserved orthogonality of mid and side */
         renormalise_vector(X, Q15ONE, N, 1);
         renormalise_vector(Y, Q15ONE, N, 1);
      }
   }
}

void quant_all_bands(int encode, const CELTMode *m, int start, int end, celt_norm *_X, celt_norm *_Y, const celt_ener *bandE, int *pulses, int shortBlocks, int fold, int *tf_res, int resynth, int total_bits, void *ec, int LM)
{
   int i, balance;
   celt_int32 remaining_bits;
   const celt_int16 * restrict eBands = m->eBands;
   celt_norm * restrict norm;
   VARDECL(celt_norm, _norm);
   int B;
   int M;
   celt_int32 seed;
   celt_norm *lowband;
   int update_lowband = 1;
   int C = _Y != NULL ? 2 : 1;
   SAVE_STACK;

   M = 1<<LM;
   B = shortBlocks ? M : 1;
   ALLOC(_norm, M*eBands[m->nbEBands], celt_norm);
   norm = _norm;

   if (encode)
      seed = ((ec_enc*)ec)->rng;
   else
      seed = ((ec_dec*)ec)->rng;
   balance = 0;
   lowband = NULL;
   for (i=start;i<end;i++)
   {
      int tell;
      int b;
      int N;
      int curr_balance;
      celt_norm * restrict X, * restrict Y;
      int tf_change=0;
      celt_norm *effective_lowband;
      
      X = _X+M*eBands[i];
      if (_Y!=NULL)
         Y = _Y+M*eBands[i];
      else
         Y = NULL;
      N = M*eBands[i+1]-M*eBands[i];
      if (encode)
         tell = ec_enc_tell((ec_enc*)ec, BITRES);
      else
         tell = ec_dec_tell((ec_dec*)ec, BITRES);

      /* Compute how many bits we want to allocate to this band */
      if (i != start)
         balance -= tell;
      remaining_bits = (total_bits<<BITRES)-tell-1;
      curr_balance = (end-i);
      if (curr_balance > 3)
         curr_balance = 3;
      curr_balance = balance / curr_balance;
      b = IMIN(remaining_bits+1,pulses[i]+curr_balance);
      if (b<0)
         b = 0;
      /* Prevents ridiculous bit depths */
      if (b > C*16*N<<BITRES)
         b = C*16*N<<BITRES;

      if (M*eBands[i]-N >= M*eBands[start] && (update_lowband || lowband==NULL))
            lowband = norm+M*eBands[i]-N;

      tf_change = tf_res[i];
      if (i>=m->effEBands)
      {
         X=norm;
         if (_Y!=NULL)
            Y = norm;
      }

      if (tf_change==0 && !shortBlocks && fold)
         effective_lowband = NULL;
      else
         effective_lowband = lowband;
      quant_band(encode, m, i, X, Y, N, b, fold, B, tf_change, effective_lowband, resynth, ec, &remaining_bits, LM, norm+M*eBands[i], bandE, 0, &seed);

      balance += pulses[i] + tell;

      /* Update the folding position only as long as we have 2 bit/sample depth */
      update_lowband = (b>>BITRES)>2*N;
   }
   RESTORE_STACK;
}

