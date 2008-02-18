/* (C) 2007 Jean-Marc Valin, CSIRO
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

#include "celt.h"
#include "modes.h"
#include "os_support.h"

int celt_mode_info(const CELTMode *mode, int request, celt_int32_t *value)
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
      default:
         return CELT_BAD_ARG;
   }
   return CELT_OK;
}

#define PBANDS 8
#define MIN_BINS 4
#define BARK_BANDS 25
const celt_int16_t bark_freq[BARK_BANDS+1] = {
      0,   101,   200,   301,   405,
    516,   635,   766,   912,  1077,
   1263,  1476,  1720,  2003,  2333,
   2721,  3184,  3742,  4428,  5285,
   6376,  7791,  9662, 12181, 15624,
   20397};
   
const celt_int16_t pitch_freq[PBANDS+1] ={0, 345, 689, 1034, 1378, 2067, 3273, 5340, 6374};

#define BITALLOC_SIZE 10
int band_allocation[BARK_BANDS*BITALLOC_SIZE] = 
   {  2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      2,  2,  2,  1,  2,  2,  2,  2,  2,  1,  2,  2,  3,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,  0,  0,
      2,  2,  2,  2,  3,  2,  2,  2,  2,  2,  3,  2,  4,  4,  4,  4,  4,  4,  4,  4,  0,  0,  0,  0,  0,
      3,  2,  2,  2,  3,  3,  2,  3,  2,  2,  4,  3,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  0,  0,  0,
      3,  3,  2,  2,  3,  3,  3,  3,  3,  2,  4,  4,  7,  7,  5,  5,  5,  5,  5,  5,  5,  5,  5,  0,  0,
      3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  8,  8,  8,  8,  8,  8,  9, 10, 11, 10, 10,  5,  5,
      4,  4,  4,  4,  5,  5,  5,  5,  5,  4,  7,  7, 14, 13, 13, 13, 13, 13, 15, 16, 17, 18, 20, 18, 11,
      7,  7,  6,  6,  9,  8,  8,  8,  8,  8, 11, 11, 20, 18, 19, 19, 25, 22, 25, 30, 30, 35, 35, 35, 35,
      8,  8,  8,  8, 10, 10, 10, 10,  9,  9, 19, 18, 25, 24, 23, 21, 29, 27, 35, 40, 42, 50, 59, 54, 51,
     11, 11, 10, 10, 14, 13, 13, 13, 13, 12, 19, 18, 35, 34, 33, 31, 39, 37, 45, 50, 52, 60, 60, 60, 60,
   };


static int *compute_ebands(int Fs, int frame_size, int *nbEBands)
{
   int *eBands;
   int i, res, min_width, lin, low, high;
   res = (Fs+frame_size)/(2*frame_size);
   min_width = MIN_BINS*res;
   //printf ("min_width = %d\n", min_width);

   /* Find where the linear part ends (i.e. where the spacing is more than min_width */
   for (lin=0;lin<BARK_BANDS;lin++)
      if (bark_freq[lin+1]-bark_freq[lin] >= min_width)
         break;
   
   //printf ("lin = %d (%d Hz)\n", lin, bark_freq[lin]);
   low = ((bark_freq[lin]/res)+(MIN_BINS-1))/MIN_BINS;
   high = BARK_BANDS-lin;
   *nbEBands = low+high;
   eBands = celt_alloc(sizeof(int)*(*nbEBands+2));
   
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
   int *pBands;
   pBands=celt_alloc(sizeof(int)*(PBANDS+2));
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
      //printf ("%d %d\n", i, j);
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
   int *allocVectors;
   
   mode->nbAllocVectors = BITALLOC_SIZE;
   allocVectors = celt_alloc(sizeof(int)*(BITALLOC_SIZE*mode->nbEBands));
   for (i=0;i<BITALLOC_SIZE;i++)
   {
      eband = 0;
      for (j=0;j<BARK_BANDS;j++)
      {
         int edge, low;
         edge = mode->eBands[eband+1]*res;
         if (edge < bark_freq[j+1])
         {
            int num, den;
            num = band_allocation[i*BARK_BANDS+j] * (edge-bark_freq[j]);
            den = bark_freq[j+1]-bark_freq[j];
            //low = band_allocation[i*BARK_BANDS+j] * (edge-bark_freq[j])/(bark_freq[j+1]-bark_freq[j]);
            low = (num+den/2)/den;
            allocVectors[i*mode->nbEBands+eband] += low;
            eband++;
            allocVectors[i*mode->nbEBands+eband] += band_allocation[i*BARK_BANDS+j]-low;
         } else {
            allocVectors[i*mode->nbEBands+eband] += band_allocation[i*BARK_BANDS+j];
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

CELTMode *celt_mode_create(int Fs, int channels, int frame_size, int overlap)
{
   int res;
   CELTMode *mode;
   
   res = (Fs+frame_size)/(2*frame_size);
   
   mode = celt_alloc(sizeof(CELTMode));
   mode->overlap = overlap;
   mode->mdctSize = frame_size;
   mode->nbMdctBlocks = 1;
   mode->nbChannels = channels;
   mode->eBands = compute_ebands(Fs, frame_size, &mode->nbEBands);
   compute_pbands(mode, res);
   mode->ePredCoef = .8;
   
   compute_allocation_table(mode, res);
   
   //printf ("%d bands\n", mode->nbEBands);
   return mode;
}

void celt_mode_destroy(CELTMode *mode)
{
   celt_free((int*)mode->eBands);
   celt_free((int*)mode->pBands);
   celt_free((int*)mode->allocVectors);
}
