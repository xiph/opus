/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2008 Gregory Maxwell 
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

#include "celt.h"
#include "modes.h"
#include "rate.h"
#include "os_support.h"
#include "stack_alloc.h"
#include "quant_bands.h"

#ifdef STATIC_MODES
#include "static_modes.c"
#endif

#define MODEVALID   0xa110ca7e
#define MODEPARTIAL 0x7eca10a1
#define MODEFREED   0xb10cf8ee

#ifndef M_PI
#define M_PI 3.141592653
#endif


int celt_mode_info(const CELTMode *mode, int request, celt_int32 *value)
{
   if (check_mode(mode) != CELT_OK)
      return CELT_INVALID_MODE;
   switch (request)
   {
      case CELT_GET_LOOKAHEAD:
         *value = mode->overlap;
         break;
      case CELT_GET_BITSTREAM_VERSION:
         *value = CELT_BITSTREAM_VERSION;
         break;
      case CELT_GET_SAMPLE_RATE:
         *value = mode->Fs;
         break;
      default:
         return CELT_UNIMPLEMENTED;
   }
   return CELT_OK;
}

#ifndef STATIC_MODES

/* Defining 25 critical bands for the full 0-20 kHz audio bandwidth
   Taken from http://ccrma.stanford.edu/~jos/bbt/Bark_Frequency_Scale.html */
#define BARK_BANDS 25
static const celt_int16 bark_freq[BARK_BANDS+1] = {
      0,   100,   200,   300,   400,
    510,   630,   770,   920,  1080,
   1270,  1480,  1720,  2000,  2320,
   2700,  3150,  3700,  4400,  5300,
   6400,  7700,  9500, 12000, 15500,
  20000};

/* This allocation table is per critical band. When creating a mode, the bits get added together 
   into the codec bands, which are sometimes larger than one critical band at low frequency */

#define BITALLOC_SIZE 12

static const celt_int16 eband5ms[] = {
       0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};

static const unsigned char band_allocation[] = {
    /* 0 200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 */
      10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      10,  3,  8,  2,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      10,  6,  8,  6,  5,  4,  3,  2,  7, 10, 11,  9,  7,  3,  1,  0,  0,  0,  0,  0,  0,
      10, 10, 14, 11, 10,  8,  6,  5, 10, 12, 13, 11,  8,  4,  2,  1,  0,  0,  0,  0,  0,
      13, 10, 17, 16, 14, 12, 10,  8, 12, 14, 14, 12,  9,  5,  3,  2,  2,  1,  0,  0,  0,
      17, 21, 23, 26, 24, 20, 17, 16, 17, 18, 16, 14, 11,  6,  3,  2,  2,  1,  1,  0,  0,
      21, 21, 36, 32, 28, 24, 23, 23, 22, 18, 18, 14, 11,  7,  5,  5,  5,  3,  3,  0,  0,
      31, 35, 40, 32, 30, 28, 26, 26, 25, 24, 19, 15, 15, 13,  9,  9,  8,  7,  5,  2,  0,
      42, 46, 46, 37, 35, 34, 33, 32, 34, 35, 32, 31, 27, 24, 23, 23, 18, 14, 11,  7,  0,
      46, 49, 46, 46, 42, 43, 44, 47, 50, 52, 51, 48, 39, 32, 27, 24, 22, 19, 17, 11,  5,
      53, 53, 49, 48, 55, 66, 71, 71, 71, 65, 64, 64, 56, 47, 41, 37, 31, 24, 20, 16, 10,
      60, 64, 74, 74, 87,103,106,102,101,100,101, 95, 80, 69, 63, 55, 47, 36, 26, 21, 15,
};

static celt_int16 *compute_ebands(celt_int32 Fs, int frame_size, int res, int *nbEBands)
{
   celt_int16 *eBands;
   int i, lin, low, high, nBark, offset=0;

   if (Fs == 400*(celt_int32)frame_size && Fs >= 40000)
   {
      *nbEBands = sizeof(eband5ms)/sizeof(eband5ms[0])-1;
      eBands = celt_alloc(sizeof(celt_int16)*(*nbEBands+2));
      for (i=0;i<*nbEBands+2;i++)
         eBands[i] = eband5ms[i];
      eBands[*nbEBands+1] = frame_size;
      return eBands;
   }
   /* Find the number of critical bands supported by our sampling rate */
   for (nBark=1;nBark<BARK_BANDS;nBark++)
    if (bark_freq[nBark+1]*2 >= Fs)
       break;

   /* Find where the linear part ends (i.e. where the spacing is more than min_width */
   for (lin=0;lin<nBark;lin++)
      if (bark_freq[lin+1]-bark_freq[lin] >= res)
         break;

   low = (bark_freq[lin]+res/2)/res;
   high = nBark-lin;
   *nbEBands = low+high;
   eBands = celt_alloc(sizeof(celt_int16)*(*nbEBands+2));
   
   if (eBands==NULL)
      return NULL;
   
   /* Linear spacing (min_width) */
   for (i=0;i<low;i++)
      eBands[i] = i;
   if (low>0)
      offset = eBands[low-1]*res - bark_freq[lin-1];
   /* Spacing follows critical bands */
   for (i=0;i<high;i++)
   {
      int target = bark_freq[lin+i];
      eBands[i+low] = (target+(offset+res)/2)/res;
      offset = eBands[i+low]*res - target;
   }
   /* Enforce the minimum spacing at the boundary */
   for (i=0;i<*nbEBands;i++)
      if (eBands[i] < i)
         eBands[i] = i;
   eBands[*nbEBands] = (bark_freq[nBark]+res/2)/res;
   eBands[*nbEBands+1] = frame_size;
   if (eBands[*nbEBands] > eBands[*nbEBands+1])
      eBands[*nbEBands] = eBands[*nbEBands+1];
   for (i=1;i<*nbEBands-1;i++)
   {
      if (eBands[i+1]-eBands[i] < eBands[i]-eBands[i-1])
      {
         eBands[i] -= (2*eBands[i]-eBands[i-1]-eBands[i+1])/2;
      }
   }
   /*for (i=0;i<=*nbEBands+1;i++)
      printf ("%d ", eBands[i]);
   printf ("\n");
   exit(1);*/
   /* FIXME: Remove last band if too small */
   return eBands;
}

static void compute_allocation_table(CELTMode *mode, int res)
{
   int i, j;
   unsigned char *allocVectors;
   int maxBands = sizeof(eband5ms)/sizeof(eband5ms[0])-1;

   mode->nbAllocVectors = BITALLOC_SIZE;
   allocVectors = celt_alloc(sizeof(unsigned char)*(BITALLOC_SIZE*mode->nbEBands));
   if (allocVectors==NULL)
      return;

   /* Check for standard mode */
   if (mode->Fs == 400*(celt_int32)mode->shortMdctSize && mode->Fs >= 40000)
   {
      for (i=0;i<BITALLOC_SIZE*mode->nbEBands;i++)
         allocVectors[i] = band_allocation[i];
      mode->allocVectors = allocVectors;
      return;
   }
   /* If not the standard mode, interpolate */

   /* Compute per-codec-band allocation from per-critical-band matrix */
   for (i=0;i<BITALLOC_SIZE;i++)
   {
      celt_int32 current = 0;
      int eband = 0;
      /* We may be looping over too many bands, but eband will stop being
         incremented once we reach the last band */
      for (j=0;j<maxBands;j++)
      {
         int edge, low, high;
         celt_int32 alloc;
         alloc = band_allocation[i*maxBands + j]*(mode->eBands[eband+1]-mode->eBands[eband])<<4;
         low = eband5ms[j]*200;
         high = eband5ms[j+1]*200;
         edge = mode->eBands[eband+1]*res;
         while (edge <= high && eband < mode->nbEBands)
         {
            celt_int32 num;
            int den, bits;
            int N = (mode->eBands[eband+1]-mode->eBands[eband]);
            num = alloc * (edge-low);
            den = high-low;
            /* Divide with rounding */
            bits = (2*num+den)/(2*den);
            allocVectors[i*mode->nbEBands+eband] = (2*(current+bits)+(N<<4))/(2*N<<4);
            /* Remove the part of the band we just allocated */
            low = edge;
            alloc -= bits;

            /* Move to next eband */
            current = 0;
            eband++;
            edge = mode->eBands[eband+1]*res;
         }
         current += alloc;
      }
      if (eband < mode->nbEBands)
      {
         int N = (mode->eBands[eband+1]-mode->eBands[eband]);
         allocVectors[i*mode->nbEBands+eband] = (2*current+(N<<4))/(2*N<<4);
      }
   }
   /*printf ("\n");
   for (i=0;i<BITALLOC_SIZE;i++)
   {
      for (j=0;j<mode->nbEBands;j++)
         printf ("%d ", allocVectors[i*mode->nbEBands+j]);
      printf ("\n");
   }
   exit(0);*/

   mode->allocVectors = allocVectors;
}

#endif /* STATIC_MODES */

CELTMode *celt_mode_create(celt_int32 Fs, int frame_size, int *error)
{
   int i;
   int LM;
#ifdef STDIN_TUNING
   scanf("%d ", &MIN_BINS);
   scanf("%d ", &BITALLOC_SIZE);
   band_allocation = celt_alloc(sizeof(int)*BARK_BANDS*BITALLOC_SIZE);
   for (i=0;i<BARK_BANDS*BITALLOC_SIZE;i++)
   {
      scanf("%d ", band_allocation+i);
   }
#endif
#ifdef STATIC_MODES
   const CELTMode *m = NULL;
   CELTMode *mode=NULL;
   ALLOC_STACK;
#if !defined(VAR_ARRAYS) && !defined(USE_ALLOCA)
   if (global_stack==NULL)
      goto failure;
#endif 
   for (i=0;i<TOTAL_MODES;i++)
   {
      if (Fs == static_mode_list[i]->Fs &&
          frame_size == static_mode_list[i]->shortMdctSize*static_mode_list[i]->nbShortMdcts)
      {
         m = static_mode_list[i];
         break;
      }
   }
   if (m == NULL)
   {
      celt_warning("Mode not included as part of the static modes");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
   mode = (CELTMode*)celt_alloc(sizeof(CELTMode));
   if (mode==NULL)
      goto failure;
   CELT_COPY(mode, m, 1);
   mode->bits = mode->_bits+1;
   mode->marker_start = MODEPARTIAL;
#else
   int res;
   CELTMode *mode=NULL;
   celt_word16 *window;
   celt_int16 *logN;
   ALLOC_STACK;
#if !defined(VAR_ARRAYS) && !defined(USE_ALLOCA)
   if (global_stack==NULL)
      goto failure;
#endif 

   /* The good thing here is that permutation of the arguments will automatically be invalid */
   
   if (Fs < 32000 || Fs > 96000)
   {
      celt_warning("Sampling rate must be between 32 kHz and 96 kHz");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
   if (frame_size < 64 || frame_size > 1024 || frame_size%2!=0)
   {
      celt_warning("Only even frame sizes from 64 to 1024 are supported");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
   
   mode = celt_alloc(sizeof(CELTMode));
   if (mode==NULL)
      goto failure;
   mode->marker_start = MODEPARTIAL;
   mode->Fs = Fs;
   mode->ePredCoef = QCONST16(.8f,15);

   if (frame_size >= 640 && (frame_size%16)==0)
   {
     LM = 3;
   } else if (frame_size >= 320 && (frame_size%8)==0)
   {
     LM = 2;
   } else if (frame_size >= 160 && (frame_size%4)==0)
   {
     LM = 1;
   } else
   {
     LM = 0;
   }

   mode->maxLM = LM;
   mode->nbShortMdcts = 1<<LM;
   mode->shortMdctSize = frame_size/mode->nbShortMdcts;
   res = (mode->Fs+mode->shortMdctSize)/(2*mode->shortMdctSize);

   mode->eBands = compute_ebands(Fs, mode->shortMdctSize, res, &mode->nbEBands);
   if (mode->eBands==NULL)
      goto failure;

   mode->pitchEnd = 4000*(celt_int32)mode->shortMdctSize/Fs;
   
   /* Overlap must be divisible by 4 */
   if (mode->nbShortMdcts > 1)
      mode->overlap = (mode->shortMdctSize>>2)<<2;
   else
      mode->overlap = (frame_size>>3)<<2;


   compute_allocation_table(mode, res);
   if (mode->allocVectors==NULL)
      goto failure;
   
   window = (celt_word16*)celt_alloc(mode->overlap*sizeof(celt_word16));
   if (window==NULL)
      goto failure;

#ifndef FIXED_POINT
   for (i=0;i<mode->overlap;i++)
      window[i] = Q15ONE*sin(.5*M_PI* sin(.5*M_PI*(i+.5)/mode->overlap) * sin(.5*M_PI*(i+.5)/mode->overlap));
#else
   for (i=0;i<mode->overlap;i++)
      window[i] = MIN32(32767,floor(.5+32768.*sin(.5*M_PI* sin(.5*M_PI*(i+.5)/mode->overlap) * sin(.5*M_PI*(i+.5)/mode->overlap))));
#endif
   mode->window = window;

   mode->bits = mode->_bits+1;
   for (i=0;(1<<i)<=mode->nbShortMdcts;i++)
   {
      mode->bits[i] = (const celt_int16 **)compute_alloc_cache(mode, 1<<i);
      if (mode->bits[i]==NULL)
         goto failure;
   }
   mode->bits[-1] = (const celt_int16 **)compute_alloc_cache(mode, 0);
   if (mode->bits[-1]==NULL)
      goto failure;

   logN = (celt_int16*)celt_alloc(mode->nbEBands*sizeof(celt_int16));
   if (logN==NULL)
      goto failure;

   for (i=0;i<mode->nbEBands;i++)
      logN[i] = log2_frac(mode->eBands[i+1]-mode->eBands[i], BITRES);
   mode->logN = logN;
#endif /* !STATIC_MODES */

   clt_mdct_init(&mode->mdct, 2*mode->shortMdctSize*mode->nbShortMdcts, LM);
   if ((mode->mdct.trig==NULL)
#ifndef ENABLE_TI_DSPLIB55
         || (mode->mdct.kfft==NULL)
#endif
   )
      goto failure;

   mode->prob = quant_prob_alloc(mode);
   if (mode->prob==NULL)
     goto failure;

   mode->marker_start = MODEVALID;
   mode->marker_end   = MODEVALID;
   if (error)
      *error = CELT_OK;
   return mode;
failure: 
   if (error)
      *error = CELT_INVALID_MODE;
   if (mode!=NULL)
      celt_mode_destroy(mode);
   return NULL;
}

void celt_mode_destroy(CELTMode *mode)
{
   int i, m;
   const celt_int16 *prevPtr = NULL;
   if (mode == NULL)
   {
      celt_warning("NULL passed to celt_mode_destroy");
      return;
   }

   if (mode->marker_start == MODEFREED || mode->marker_end == MODEFREED)
   {
      celt_warning("Freeing a mode which has already been freed"); 
      return;
   }

   if (mode->marker_start != MODEVALID && mode->marker_start != MODEPARTIAL)
   {
      celt_warning("This is not a valid CELT mode structure");
      return;  
   }
   mode->marker_start = MODEFREED;
#ifndef STATIC_MODES
   for (m=0;(1<<m)<=mode->nbShortMdcts;m++)
   {
      if (mode->bits[m]!=NULL)
      {
         for (i=0;i<mode->nbEBands;i++)
         {
            if (mode->bits[m][i] != prevPtr)
            {
               prevPtr = mode->bits[m][i];
               celt_free((int*)mode->bits[m][i]);
            }
         }
      }
      celt_free((celt_int16**)mode->bits[m]);
   }
   if (mode->bits[-1]!=NULL)
   {
      for (i=0;i<mode->nbEBands;i++)
      {
         if (mode->bits[-1][i] != prevPtr)
         {
            prevPtr = mode->bits[-1][i];
            celt_free((int*)mode->bits[-1][i]);
         }
      }
   }
   celt_free((celt_int16**)mode->bits[-1]);

   celt_free((celt_int16*)mode->eBands);
   celt_free((celt_int16*)mode->allocVectors);
   
   celt_free((celt_word16*)mode->window);
   celt_free((celt_int16*)mode->logN);

#endif
   clt_mdct_clear(&mode->mdct);

   quant_prob_free(mode->prob);
   mode->marker_end = MODEFREED;
   celt_free((CELTMode *)mode);
}

int check_mode(const CELTMode *mode)
{
   if (mode==NULL)
      return CELT_INVALID_MODE;
   if (mode->marker_start == MODEVALID && mode->marker_end == MODEVALID)
      return CELT_OK;
   if (mode->marker_start == MODEFREED || mode->marker_end == MODEFREED)
      celt_warning("Using a mode that has already been freed");
   else
      celt_warning("This is not a valid CELT mode");
   return CELT_INVALID_MODE;
}
