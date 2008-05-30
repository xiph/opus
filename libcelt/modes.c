/* (C) 2007-2008 Jean-Marc Valin, CSIRO
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

#define MODEVALID 0xa110ca7e
#define MODEFREED 0xb10cf8ee

#ifndef M_PI
#define M_PI 3.141592653
#endif


int EXPORT celt_mode_info(const CELTMode *mode, int request, celt_int32_t *value)
{
   switch (request)
   {
      case CELT_GET_FRAME_SIZE:
         *value = mode->mdctSize;
         break;
      case CELT_GET_LOOKAHEAD:
         *value = mode->overlap;
         break;
      case CELT_GET_NB_CHANNELS:
         *value = mode->nbChannels;
         break;
      case CELT_GET_BITSTREAM_VERSION:
         *value = CELT_BITSTREAM_VERSION;
         break;
      default:
         return CELT_BAD_ARG;
   }
   return CELT_OK;
}

#ifndef STATIC_MODES

#define PBANDS 8

#ifdef STDIN_TUNING
int MIN_BINS;
#else
#define MIN_BINS 3
#endif

/* Defining 25 critical bands for the full 0-20 kHz audio bandwidth
   Taken from http://ccrma.stanford.edu/~jos/bbt/Bark_Frequency_Scale.html */
#define BARK_BANDS 25
static const celt_int16_t bark_freq[BARK_BANDS+1] = {
      0,   100,   200,   300,   400,
    510,   630,   770,   920,  1080,
   1270,  1480,  1720,  2000,  2320,
   2700,  3150,  3700,  4400,  5300,
   6400,  7700,  9500, 12000, 15500,
  20000};

static const celt_int16_t pitch_freq[PBANDS+1] ={0, 345, 689, 1034, 1378, 2067, 3273, 5340, 6374};

/* This allocation table is per critical band. When creating a mode, the bits get added together 
   into the codec bands, which are sometimes larger than one critical band at low frequency */

#ifdef STDIN_TUNING
int BITALLOC_SIZE;
int *band_allocation;
#else
#define BITALLOC_SIZE 10
static const int band_allocation[BARK_BANDS*BITALLOC_SIZE] = 
   {  2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      2,  2,  2,  1,  2,  2,  2,  2,  2,  1,  2,  2,  4,  5,  7,  7,  7,  5,  4,  0,  0,  0,  0,  0,  0,
      2,  2,  2,  2,  3,  2,  2,  2,  2,  2,  3,  3,  5,  6,  8,  8,  8,  6,  5,  4,  0,  0,  0,  0,  0,
      3,  2,  2,  2,  3,  3,  2,  3,  2,  3,  4,  4,  6,  7,  9,  9,  9,  7,  6,  5,  5,  5,  0,  0,  0,
      3,  3,  2,  2,  3,  3,  3,  3,  3,  4,  4,  5,  7,  9, 10, 10, 10,  9,  6,  5,  5,  5,  5,  1,  0,
      4,  3,  3,  3,  3,  3,  3,  3,  4,  4,  6,  7,  7,  9, 11, 10, 10,  9,  9,  8, 11, 10, 10,  1,  1,
      5,  5,  4,  4,  5,  5,  5,  5,  6,  6,  8,  8, 10, 12, 15, 15, 13, 12, 12, 12, 18, 18, 16, 10,  1,
      6,  6,  6,  6,  6,  6,  7,  7,  9,  9, 11, 12, 13, 18, 22, 23, 24, 25, 28, 30, 35, 35, 35, 35, 15,
      7,  7,  7,  7,  7,  7, 10, 10, 10, 13, 14, 18, 20, 24, 28, 32, 32, 35, 38, 38, 42, 50, 59, 54, 31,
      8,  8,  8,  8,  8,  9, 10, 12, 14, 20, 22, 25, 28, 30, 35, 42, 46, 50, 55, 60, 62, 62, 62, 62, 62,
};
#endif

static celt_int16_t *compute_ebands(celt_int32_t Fs, int frame_size, int *nbEBands)
{
   celt_int16_t *eBands;
   int i, res, min_width, lin, low, high;
   res = (Fs+frame_size)/(2*frame_size);
   min_width = MIN_BINS*res;
   /*printf ("min_width = %d\n", min_width);*/

   /* Find where the linear part ends (i.e. where the spacing is more than min_width */
   for (lin=0;lin<BARK_BANDS;lin++)
      if (bark_freq[lin+1]-bark_freq[lin] >= min_width)
         break;
   
   /*printf ("lin = %d (%d Hz)\n", lin, bark_freq[lin]);*/
   low = ((bark_freq[lin]/res)+(MIN_BINS-1))/MIN_BINS;
   high = BARK_BANDS-lin;
   *nbEBands = low+high;
   eBands = celt_alloc(sizeof(celt_int16_t)*(*nbEBands+2));
   
   /* Linear spacing (min_width) */
   for (i=0;i<low;i++)
      eBands[i] = MIN_BINS*i;
   /* Spacing follows critical bands */
   for (i=0;i<high;i++)
      eBands[i+low] = (bark_freq[lin+i]+res/2)/res;
   /* Enforce the minimum spacing at the boundary */
   for (i=0;i<*nbEBands;i++)
      if (eBands[i] < MIN_BINS*i)
         eBands[i] = MIN_BINS*i;
   eBands[*nbEBands] = (bark_freq[BARK_BANDS]+res/2)/res;
   eBands[*nbEBands+1] = frame_size;
   if (eBands[*nbEBands] > eBands[*nbEBands+1])
      eBands[*nbEBands] = eBands[*nbEBands+1];
   
   /* FIXME: Remove last band if too small */
   /*for (i=0;i<*nbEBands+2;i++)
      printf("%d ", eBands[i]);
   printf ("\n");*/
   return eBands;
}

static void compute_pbands(CELTMode *mode, int res)
{
   int i;
   celt_int16_t *pBands;
   pBands=celt_alloc(sizeof(celt_int16_t)*(PBANDS+2));
   mode->nbPBands = PBANDS;
   for (i=0;i<PBANDS+1;i++)
   {
      pBands[i] = (pitch_freq[i]+res/2)/res;
      if (pBands[i] < mode->eBands[i])
         pBands[i] = mode->eBands[i];
   }
   pBands[PBANDS+1] = mode->eBands[mode->nbEBands+1];
   for (i=1;i<mode->nbPBands+1;i++)
   {
      int j;
      for (j=0;j<mode->nbEBands;j++)
         if (mode->eBands[j] <= pBands[i] && mode->eBands[j+1] > pBands[i])
            break;
      /*printf ("%d %d\n", i, j);*/
      if (mode->eBands[j] != pBands[i])
      {
         if (pBands[i]-mode->eBands[j] < mode->eBands[j+1]-pBands[i] && 
             mode->eBands[j] != pBands[i-1])
            pBands[i] = mode->eBands[j];
         else
            pBands[i] = mode->eBands[j+1];
      }
   }
   /*for (i=0;i<mode->nbPBands+2;i++)
      printf("%d ", pBands[i]);
   printf ("\n");*/
   mode->pBands = pBands;
   mode->pitchEnd = pBands[PBANDS];
}

static void compute_allocation_table(CELTMode *mode, int res)
{
   int i, j, eband;
   celt_int16_t *allocVectors;
   
   mode->nbAllocVectors = BITALLOC_SIZE;
   allocVectors = celt_alloc(sizeof(celt_int16_t)*(BITALLOC_SIZE*mode->nbEBands));
   for (i=0;i<BITALLOC_SIZE;i++)
   {
      eband = 0;
      for (j=0;j<BARK_BANDS;j++)
      {
         int edge, low, alloc;
         edge = mode->eBands[eband+1]*res;
         alloc = band_allocation[i*BARK_BANDS+j];
         if (mode->nbChannels == 2)
            alloc += alloc/2;
         if (edge < bark_freq[j+1])
         {
            int num, den;
            num = alloc * (edge-bark_freq[j]);
            den = bark_freq[j+1]-bark_freq[j];
            low = (num+den/2)/den;
            allocVectors[i*mode->nbEBands+eband] += low;
            eband++;
            allocVectors[i*mode->nbEBands+eband] += alloc-low;
         } else {
            allocVectors[i*mode->nbEBands+eband] += alloc;
         }
      }
   }
   /*for (i=0;i<BITALLOC_SIZE;i++)
   {
      for (j=0;j<mode->nbEBands;j++)
         printf ("%2d ", allocVectors[i*mode->nbEBands+j]);
      printf ("\n");
   }*/
   mode->allocVectors = allocVectors;
}

#endif /* STATIC_MODES */

static void compute_energy_allocation_table(CELTMode *mode)
{
   int i, j;
   celt_int16_t *alloc;
   
   alloc = celt_alloc(sizeof(celt_int16_t)*(mode->nbAllocVectors*(mode->nbEBands+1)));
   for (i=0;i<mode->nbAllocVectors;i++)
   {
      int sum = 0;
      int min_bits = 1;
      if (mode->allocVectors[i*mode->nbEBands]>12)
         min_bits = 2;
      if (mode->allocVectors[i*mode->nbEBands]>24)
         min_bits = 3;
      for (j=0;j<mode->nbEBands;j++)
      {
         alloc[i*(mode->nbEBands+1)+j] = mode->allocVectors[i*mode->nbEBands+j]
                                         / (mode->eBands[j+1]-mode->eBands[j]-1);
         if (alloc[i*(mode->nbEBands+1)+j]<min_bits)
            alloc[i*(mode->nbEBands+1)+j] = min_bits;
         if (alloc[i*(mode->nbEBands+1)+j]>7)
            alloc[i*(mode->nbEBands+1)+j] = 7;
         sum += alloc[i*(mode->nbEBands+1)+j];
         /*printf ("%d ", alloc[i*(mode->nbEBands+1)+j]);*/
         /*printf ("%f ", mode->allocVectors[i*mode->nbEBands+j]*1.f/(mode->eBands[j+1]-mode->eBands[j]-1));*/
      }
      alloc[i*(mode->nbEBands+1)+mode->nbEBands] = sum;
      /*printf ("\n");*/
   }
   mode->energy_alloc = alloc;
}

CELTMode EXPORT *celt_mode_create(celt_int32_t Fs, int channels, int frame_size, int lookahead, int *error)
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
   for (i=0;i<TOTAL_MODES;i++)
   {
      if (Fs == static_mode_list[i]->Fs &&
          channels == static_mode_list[i]->nbChannels &&
          frame_size == static_mode_list[i]->mdctSize &&
          lookahead == static_mode_list[i]->overlap)
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
   CELT_COPY(mode, m, 1);
#else
   int res;
   CELTMode *mode;
   celt_word16_t *window;
   ALLOC_STACK;

   /* The good thing here is that permutation of the arguments will automatically be invalid */
   
   if (Fs < 32000 || Fs > 64000)
   {
      celt_warning("Sampling rate must be between 32 kHz and 64 kHz");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
   if (channels < 0 || channels > 2)
   {
      celt_warning("Only mono and stereo supported");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
   if (frame_size < 64 || frame_size > 256 || frame_size%2!=0)
   {
      celt_warning("Only even frame sizes between 64 and 256 are supported");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
   if (lookahead < 32 || lookahead > frame_size)
   {
      celt_warning("The overlap must be between 32 and the frame size");
      if (error)
         *error = CELT_BAD_ARG;
      return NULL;
   }
   res = (Fs+frame_size)/(2*frame_size);
   
   mode = celt_alloc(sizeof(CELTMode));
   mode->Fs = Fs;
   mode->overlap = lookahead;
   mode->mdctSize = frame_size;
   mode->nbChannels = channels;
   mode->eBands = compute_ebands(Fs, frame_size, &mode->nbEBands);
   compute_pbands(mode, res);
   mode->ePredCoef = QCONST16(.8f,15);
   
   compute_allocation_table(mode, res);
   /*printf ("%d bands\n", mode->nbEBands);*/
   
   window = (celt_word16_t*)celt_alloc(mode->overlap*sizeof(celt_word16_t));

#ifndef FIXED_POINT
   for (i=0;i<mode->overlap;i++)
      window[i] = Q15ONE*sin(.5*M_PI* sin(.5*M_PI*(i+.5)/mode->overlap) * sin(.5*M_PI*(i+.5)/mode->overlap));
#else
   for (i=0;i<mode->overlap;i++)
      window[i] = MIN32(32767,32768.*sin(.5*M_PI* sin(.5*M_PI*(i+.5)/mode->overlap) * sin(.5*M_PI*(i+.5)/mode->overlap)));
#endif
   mode->window = window;

   mode->bits = (const celt_int16_t **)compute_alloc_cache(mode, 1);

   mode->bits_stereo = NULL;
#ifndef SHORTCUTS
   psydecay_init(&mode->psy, MAX_PERIOD/2, mode->Fs);
#endif
   
   mode->marker_start = MODEVALID;
   mode->marker_end = MODEVALID;
#endif /* !STATIC_MODES */
   mdct_init(&mode->mdct, 2*mode->mdctSize);
   mode->fft = pitch_state_alloc(MAX_PERIOD);

   mode->prob = quant_prob_alloc(mode);
   compute_energy_allocation_table(mode);
   
   if (mode->nbChannels>=2)
      mode->bits_stereo = (const celt_int16_t **)compute_alloc_cache(mode, mode->nbChannels);

   if (error)
      *error = CELT_OK;
   return mode;
}

void EXPORT celt_mode_destroy(CELTMode *mode)
{
#ifndef STATIC_MODES
   int i;
   const celt_int16_t *prevPtr = NULL;
   for (i=0;i<mode->nbEBands;i++)
   {
      if (mode->bits[i] != prevPtr)
      {
         prevPtr = mode->bits[i];
         celt_free((int*)mode->bits[i]);
      }
   }
   celt_free((int**)mode->bits);
   if (check_mode(mode) != CELT_OK)
      return;
   celt_free((int*)mode->eBands);
   celt_free((int*)mode->pBands);
   celt_free((int*)mode->allocVectors);
   
   celt_free((celt_word16_t*)mode->window);

   mode->marker_start = MODEFREED;
   mode->marker_end = MODEFREED;
#ifndef SHORTCUTS
   psydecay_clear(&mode->psy);
#endif
#endif
   mdct_clear(&mode->mdct);
   pitch_state_free(mode->fft);
   quant_prob_free(mode->prob);
   celt_free((celt_int16_t *)mode->energy_alloc);
   celt_free((CELTMode *)mode);
}

int check_mode(const CELTMode *mode)
{
   if (mode->marker_start == MODEVALID && mode->marker_end == MODEVALID)
      return CELT_OK;
   if (mode->marker_start == MODEFREED || mode->marker_end == MODEFREED)
      celt_warning("Using a mode that has already been freed");
   else
      celt_warning("This is not a valid CELT mode");
   return CELT_INVALID_MODE;
}
