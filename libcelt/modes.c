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
      case CELT_GET_FRAME_SIZE:
         *value = mode->mdctSize;
         break;
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

#ifdef STDIN_TUNING
int BITALLOC_SIZE;
int *band_allocation;
#else
#define BITALLOC_SIZE 12
static const int band_allocation[BARK_BANDS*BITALLOC_SIZE] = 
   /* 0 100 200 300 400 510 630 770 920 1k  1.2 1.5 1.7 2k  2.3 2.7 3.1 3.7 4.4 5.3 6.4 7.7 9.5 12k 15k  */
   {  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, /*0*/
      2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, /*1*/
      2,  2,  2,  1,  2,  2,  2,  2,  2,  2,  2,  2,  4,  5,  7,  7,  7,  5,  4,  0,  0,  0,  0,  0,  0, /*2*/
      2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  5,  6,  8,  8,  8,  6,  5,  4,  0,  0,  0,  0,  0, /*3*/
      3,  2,  2,  2,  3,  4,  4,  4,  4,  4,  4,  4,  6,  7,  9,  9,  9,  7,  6,  5,  5,  5,  0,  0,  0, /*4*/
      3,  3,  3,  4,  4,  5,  6,  6,  6,  6,  6,  7,  7,  9, 10, 10, 10,  9,  6,  5,  5,  5,  5,  1,  0, /*5*/
      4,  3,  3,  4,  6,  7,  7,  7,  7,  7,  8,  9,  9,  9, 11, 10, 10,  9,  9,  8, 11, 10, 10,  1,  0, /*6*/
      5,  5,  5,  6,  7,  7,  7,  7,  8,  8,  9, 10, 10, 12, 12, 11, 11, 17, 12, 15, 15, 20, 18, 10,  1, /*7*/
      6,  7,  7,  7,  8,  8,  8,  8,  9, 10, 11, 12, 14, 17, 18, 21, 22, 27, 29, 39, 37, 38, 40, 35,  1, /*8*/
      7,  7,  7,  8,  8,  8, 10, 10, 10, 13, 14, 18, 20, 24, 28, 32, 32, 35, 38, 38, 42, 50, 59, 54, 31, /*9*/
      8,  8,  8,  8,  8,  9, 10, 12, 14, 20, 22, 25, 28, 30, 35, 42, 46, 50, 55, 60, 62, 62, 72, 82, 62, /*10*/
      9,  9,  9, 10, 12, 13, 15, 18, 22, 30, 32, 35, 40, 45, 55, 62, 66, 70, 85, 90, 92, 92, 92,102, 92, /*11*/
   };
#endif

static celt_int16 *compute_ebands(celt_int32 Fs, int frame_size, int nbShortMdcts, int *nbEBands)
{
   int min_bins = 3;
   celt_int16 *eBands;
   int i, res, min_width, lin, low, high, nBark, offset=0;

   /*if (min_bins < nbShortMdcts)
      min_bins = nbShortMdcts;*/
   res = (Fs+frame_size)/(2*frame_size);
   min_width = min_bins*res;

   /* Find the number of critical bands supported by our sampling rate */
   for (nBark=1;nBark<BARK_BANDS;nBark++)
    if (bark_freq[nBark+1]*2 >= Fs)
       break;

   /* Find where the linear part ends (i.e. where the spacing is more than min_width */
   for (lin=0;lin<nBark;lin++)
      if (bark_freq[lin+1]-bark_freq[lin] >= min_width)
         break;
   
   low = ((bark_freq[lin]/res)+(min_bins-1))/min_bins;
   high = nBark-lin;
   *nbEBands = low+high;
   eBands = celt_alloc(sizeof(celt_int16)*(*nbEBands+2));
   
   if (eBands==NULL)
      return NULL;
   
   /* Linear spacing (min_width) */
   for (i=0;i<low;i++)
      eBands[i] = min_bins*i;
   /* Spacing follows critical bands */
   for (i=0;i<high;i++)
   {
      int target = bark_freq[lin+i];
      eBands[i+low] = (2*target+offset+res)/(2*res);
      offset = eBands[i+low]*res - target;
   }
   /* Enforce the minimum spacing at the boundary */
   for (i=0;i<*nbEBands;i++)
      if (eBands[i] < min_bins*i)
         eBands[i] = min_bins*i;
   eBands[*nbEBands] = (bark_freq[nBark]+res/2)/res;
   eBands[*nbEBands+1] = frame_size;
   if (eBands[*nbEBands] > eBands[*nbEBands+1])
      eBands[*nbEBands] = eBands[*nbEBands+1];
   for (i=1;i<*nbEBands-1;i++)
   {
      if (eBands[i+1]-eBands[i] < eBands[i]-eBands[i-1])
      {
         eBands[i] -= (2*eBands[i]-eBands[i-1]-eBands[i+1]+1)/2;
      }
   }
   /*for (i=0;i<*nbEBands+1;i++)
      printf ("%d ", eBands[i]);
   printf ("\n");
   exit(1);*/
   /* FIXME: Remove last band if too small */
   return eBands;
}

static void compute_allocation_table(CELTMode *mode, int res)
{
   int i, j, nBark;
   celt_int16 *allocVectors;

   /* Find the number of critical bands supported by our sampling rate */
   for (nBark=1;nBark<BARK_BANDS;nBark++)
    if (bark_freq[nBark+1]*2 >= mode->Fs)
       break;

   mode->nbAllocVectors = BITALLOC_SIZE;
   allocVectors = celt_alloc(sizeof(celt_int16)*(BITALLOC_SIZE*mode->nbEBands));
   if (allocVectors==NULL)
      return;
   /* Compute per-codec-band allocation from per-critical-band matrix */
   for (i=0;i<BITALLOC_SIZE;i++)
   {
      celt_int32 current = 0;
      int eband = 0;
      for (j=0;j<nBark;j++)
      {
         int edge, low;
         celt_int32 alloc;
         edge = mode->eBands[eband+1]*res;
         alloc = mode->mdctSize*band_allocation[i*BARK_BANDS+j];
         if (edge < bark_freq[j+1])
         {
            int num, den;
            num = alloc * (edge-bark_freq[j]);
            den = bark_freq[j+1]-bark_freq[j];
            low = (num+den/2)/den;
            allocVectors[i*mode->nbEBands+eband] = (current+low+128)/256;
            current=0;
            eband++;
            current += alloc-low;
         } else {
            current += alloc;
         }   
      }
      allocVectors[i*mode->nbEBands+eband] = (current+128)/256;
   }
   mode->allocVectors = allocVectors;
}

#endif /* STATIC_MODES */

CELTMode *celt_mode_create(celt_int32 Fs, int frame_size, int *error)
{
   int i;
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
   {
      celt_free(global_stack);
      goto failure;
   }
#endif 
   for (i=0;i<TOTAL_MODES;i++)
   {
      if (Fs == static_mode_list[i]->Fs &&
          frame_size == static_mode_list[i]->mdctSize)
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
   mode->marker_start = MODEPARTIAL;
#else
   int res;
   CELTMode *mode=NULL;
   celt_word16 *window;
   ALLOC_STACK;
#if !defined(VAR_ARRAYS) && !defined(USE_ALLOCA)
   if (global_stack==NULL)
   {
      celt_free(global_stack);
      goto failure;
   }
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
   res = (Fs+frame_size)/(2*frame_size);
   
   mode = celt_alloc(sizeof(CELTMode));
   if (mode==NULL)
      goto failure;
   mode->marker_start = MODEPARTIAL;
   mode->Fs = Fs;
   mode->mdctSize = frame_size;
   mode->ePredCoef = QCONST16(.8f,15);

   if (frame_size > 640 && (frame_size%16)==0)
   {
     mode->nbShortMdcts = 8;
   } else if (frame_size > 384 && (frame_size%8)==0)
   {
     mode->nbShortMdcts = 4;
   } else if (frame_size > 384 && (frame_size%10)==0)
   {
     mode->nbShortMdcts = 5;
   } else if (frame_size > 256 && (frame_size%6)==0)
   {
     mode->nbShortMdcts = 3;
   } else if (frame_size > 256 && (frame_size%8)==0)
   {
     mode->nbShortMdcts = 4;
   } else if (frame_size > 64 && (frame_size%4)==0)
   {
     mode->nbShortMdcts = 2;
   } else if (frame_size > 128 && (frame_size%6)==0)
   {
     mode->nbShortMdcts = 3;
   } else
   {
     mode->nbShortMdcts = 1;
   }

   mode->eBands = compute_ebands(Fs, frame_size, mode->nbShortMdcts, &mode->nbEBands);
   if (mode->eBands==NULL)
      goto failure;

   mode->pitchEnd = 4000*(celt_int32)frame_size/Fs;
   
   /* Overlap must be divisible by 4 */
   if (mode->nbShortMdcts > 1)
      mode->overlap = ((frame_size/mode->nbShortMdcts)>>2)<<2; 
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
      window[i] = MIN32(32767,32768.*sin(.5*M_PI* sin(.5*M_PI*(i+.5)/mode->overlap) * sin(.5*M_PI*(i+.5)/mode->overlap)));
#endif
   mode->window = window;

   mode->bits = (const celt_int16 **)compute_alloc_cache(mode, 1);
   if (mode->bits==NULL)
      goto failure;

#endif /* !STATIC_MODES */

   clt_mdct_init(&mode->mdct, 2*mode->mdctSize);

   mode->shortMdctSize = mode->mdctSize/mode->nbShortMdcts;
   clt_mdct_init(&mode->shortMdct, 2*mode->shortMdctSize);
   mode->shortWindow = mode->window;
   mode->prob = quant_prob_alloc(mode);
   if ((mode->mdct.trig==NULL) || (mode->shortMdct.trig==NULL)
#ifndef ENABLE_TI_DSPLIB55
        || (mode->mdct.kfft==NULL) || (mode->shortMdct.kfft==NULL)
#endif
        || (mode->prob==NULL))
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
   int i;
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
   if (mode->bits!=NULL)
   {
      for (i=0;i<mode->nbEBands;i++)
      {
         if (mode->bits[i] != prevPtr)
         {
            prevPtr = mode->bits[i];
            celt_free((int*)mode->bits[i]);
          }
      }
   }   
   celt_free((int**)mode->bits);
   celt_free((int*)mode->eBands);
   celt_free((int*)mode->allocVectors);
   
   celt_free((celt_word16*)mode->window);

#endif
   clt_mdct_clear(&mode->mdct);
   clt_mdct_clear(&mode->shortMdct);
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
