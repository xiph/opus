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

#include "modes.h"
#include "os_support.h"

#define NBANDS 18
#define PBANDS 8
#define PITCH_END 74

#define NBANDS128 15
#define PBANDS128 8
#define PITCH_END128 45

const int qbank0[NBANDS   +2] = {0,  4,  8, 12, 16, 20, 24, 28, 32, 38, 44, 52, 62, 74, 90,112,142,182, 232,256};
const int pbank0[PBANDS   +2] = {0,  4,  8, 12, 16,     24,         38,         62, PITCH_END, 256};

#define NALLOCS 7
int bitalloc0[NBANDS*NALLOCS] = 
   { 5,  4,  4,  4,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0,
     8,  7,  7,  6,  6,  6,  5,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
    10,  9,  9,  8,  8,  8,  8,  8,  8,  8,  9, 10, 11, 12, 17, 15,  6,  7,
    16, 15, 14, 14, 14, 13, 13, 13, 13, 13, 15, 16, 17, 18, 20, 18, 11, 12,
    26, 25, 24, 22, 20, 18, 19, 19, 25, 22, 25, 30, 30, 35, 35, 35, 35, 25,
    32, 30, 28, 27, 25, 24, 23, 21, 29, 27, 35, 40, 42, 50, 59, 54, 51, 36,
    42, 40, 38, 37, 35, 34, 33, 31, 39, 37, 45, 50, 52, 60, 60, 60, 60, 46,
};


#define NBANDS256 15
#define PBANDS256 8
#define PITCH_END256 88
const int qbank3[NBANDS256+2] = {0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 72, 88, 104, 136, 168, 232, 256};
//const int pbank3[PBANDS256+2] = {0, 8, 16, 24, 40, PITCH_END256, 256};
const int pbank3[PBANDS256+2] = {0, 4, 8, 12, 16, 24, 40, 56, PITCH_END256, 256};

static const CELTMode mono_mode = {
   128,         /**< overlap */
   256,         /**< mdctSize */
   1,           /**< nbMdctBlocks */
   1,           /**< channels */
   
   NBANDS,      /**< nbEBands */
   PBANDS,      /**< nbPBands */
   PITCH_END,   /**< pitchEnd */
   
   qbank0,      /**< eBands */
   pbank0,      /**< pBands*/
   
   0.8,         /**< ePredCoef */
   
   NALLOCS,     /**< nbAllocVectors */
   bitalloc0,   /**< allocVectors */
};


/* Stereo mode around 120 kbps */
static const CELTMode stereo_mode = {
   128,         /**< overlap */
   256,         /**< mdctSize */
   1,           /**< nbMdctBlocks */
   2,           /**< channels */
   
   NBANDS,      /**< nbEBands */
   PBANDS,      /**< nbPBands */
   PITCH_END,   /**< pitchEnd */
   
   qbank0,      /**< eBands */
   pbank0,      /**< pBands*/
   
   0.8,         /**< ePredCoef */
   
   NALLOCS,     /**< nbAllocVectors */
   bitalloc0,   /**< allocVectors */
};

const CELTMode const *celt_mono = &mono_mode;
const CELTMode const *celt_stereo = &stereo_mode;


#define NBANDS51 17
#define PBANDS51 8
#define PITCH_END51 64
const int qbank51[NBANDS51 +2] = {0,  4,  8, 12, 16, 20, 24, 28, 32, 38, 44, 52, 64, 78, 96,122,156,204, 256};
const int qbank51b[NBANDS +2] = {0,  3,  6, 9, 12, 16, 20, 24, 28, 32, 38, 44, 52, 64, 78, 96,122,156,204, 256};

const int pbank51[PBANDS51 +2] = {0,  4,  8, 12, 16,     24,     32,     44,     PITCH_END51, 256};
const int pbank51b[PBANDS +2] = {0,  3,  6, 9, 12,     20,     38,     52,     PITCH_END51, 256};
#define NALLOCS51 10
int bitalloc51[NBANDS51*NALLOCS51] = 
   { 6,   5,  3,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     7,   6,  5,  4,  3,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,  0,  0,
     8,   7,  6,  5,  4,  4,  4,  4,  4,  4,  4,  4,  0,  0,  0,  0,  0,
     9,   8,  7,  7,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  0,  0,  0,
     10,  9,  8,  8,  7,  7,  5,  5,  5,  5,  5,  5,  5,  5,  5,  0,  0,
     10,  9,  9,  8,  8,  8,  8,  8,  8,  8,  9, 10, 11, 10, 10,  5,  5,
     16, 15, 14, 14, 14, 13, 13, 13, 13, 13, 15, 16, 17, 18, 20, 18, 11,
     26, 25, 24, 22, 20, 18, 19, 19, 25, 22, 25, 30, 30, 35, 35, 35, 35,
     32, 30, 28, 27, 25, 24, 23, 21, 29, 27, 35, 40, 42, 50, 59, 54, 51,
     42, 40, 38, 37, 35, 34, 33, 31, 39, 37, 45, 50, 52, 60, 60, 60, 60,
   };

static const CELTMode ld51 = {
   128,         /**< overlap */
   256,         /**< mdctSize */
   1,           /**< nbMdctBlocks */
   1,           /**< channels */
   
   NBANDS51,    /**< nbEBands */
   PBANDS51,    /**< nbPBands */
   PITCH_END51, /**< pitchEnd */
   
   qbank51,     /**< eBands */
   pbank51,     /**< pBands*/
   
   0.8,         /**< ePredCoef */
   
   NALLOCS51,   /**< nbAllocVectors */
   bitalloc51,  /**< allocVectors */
};
const CELTMode const *celt_ld51 = &ld51;

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
   for (i=0;i<*nbEBands+2;i++)
      printf("%d ", eBands[i]);
   printf ("\n");
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
      printf ("%d %d\n", i, j);
      if (mode->eBands[j] != pBands[i])
      {
         if (pBands[i]-mode->eBands[j] < mode->eBands[j+1]-pBands[i] && 
             mode->eBands[j] != pBands[i-1])
            pBands[i] = mode->eBands[j];
         else
            pBands[i] = mode->eBands[j+1];
      }
   }
   for (i=0;i<mode->nbPBands+2;i++)
      printf("%d ", pBands[i]);
   printf ("\n");
   mode->pBands = pBands;
   mode->pitchEnd = pBands[PBANDS];
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
   
   
   printf ("%d bands\n", mode->nbEBands);
   return mode;
}

/*int main()
{
   celt_mode_create(44100, 1, 256, 128);
   return 0;
}*/

