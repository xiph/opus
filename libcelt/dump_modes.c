/* (C) 2008 Jean-Marc Valin, CSIRO
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

#include <stdio.h>
#include "modes.h"
#include "celt.h"

#define INT16 "%d"
#define INT32 "%d"
#define FLOAT "%f"

#ifdef FIXED_POINT
#define WORD16 INT16
#define WORD32 INT32
#else
#define WORD16 FLOAT
#define WORD32 FLOAT
#endif


void dump_modes(FILE *file, CELTMode *modes, int nb_modes)
{
   int i, j;
   for (i=0;i<nb_modes;i++)
   {
      fprintf(file, "#include \"modes.h\"\n");

      fprintf(file, "\n");
      fprintf (file, "const int eBands%d_%d[%d] = {\n", modes->Fs, modes->mdctSize, modes->nbEBands+2);
      for (j=0;j<modes->nbEBands+2;j++)
         fprintf (file, "%d, ", modes->eBands[j]);
      fprintf (file, "};\n");
      fprintf(file, "\n");
      fprintf (file, "const int pBands%d_%d[%d] = {\n", modes->Fs, modes->mdctSize, modes->nbPBands+2);
      for (j=0;j<modes->nbPBands+2;j++)
         fprintf (file, "%d, ", modes->pBands[j]);
      printf ("};\n");
      fprintf(file, "\n");
      fprintf (file, "const celt_word16_t window%d[%d] = {\n", modes->overlap, modes->overlap);
      for (j=0;j<modes->overlap;j++)
         fprintf (file, WORD16 ", ", modes->window[j]);
      printf ("};\n");
      fprintf(file, "\n");
      fprintf (file, "const int allocVectors%d_%d[%d] = {\n", modes->Fs, modes->mdctSize, modes->nbEBands*modes->nbAllocVectors);
      for (j=0;j<modes->nbAllocVectors;j++)
      {
         int k;
         for (k=0;k<modes->nbEBands;k++)
            fprintf (file, "%2d, ", modes->allocVectors[j*modes->nbEBands+k]);
         fprintf (file, "\n");
      }
      fprintf (file, "};\n");
      fprintf(file, "\n");
      fprintf(file, "CELTMode mode%d_%d_%d_%d = {\n", modes->Fs, modes->nbChannels, modes->mdctSize, modes->overlap);
      fprintf(file, "0x%x,\t/* marker */\n", 0xa110ca7e);
      fprintf(file, INT32 ",\t/* Fs */\n", modes->Fs);
      fprintf(file, "%d,\t/* overlap */\n", modes->overlap);
      fprintf(file, "%d,\t/* mdctSize */\n", modes->mdctSize);
      fprintf(file, "%d,\t/* nbMdctBlocks */\n", modes->nbMdctBlocks);
      fprintf(file, "%d,\t/* nbChannels */\n", modes->nbChannels);
      fprintf(file, "%d,\t/* nbEBands */\n", modes->nbEBands);
      fprintf(file, "%d,\t/* nbPBands */\n", modes->nbPBands);
      fprintf(file, "%d,\t/* pitchEnd */\n", modes->pitchEnd);
      fprintf(file, "eBands%d_%d,\t/* eBands */\n", modes->Fs, modes->mdctSize);
      fprintf(file, "pBands%d_%d,\t/* pBands */\n", modes->Fs, modes->mdctSize);
      fprintf(file, WORD16 ",\t/* ePredCoef */\n", modes->ePredCoef);
      fprintf(file, "%d,\t/* nbAllocVectors */\n", modes->nbAllocVectors);
      fprintf(file, "allocVectors%d_%d,\t/* allocVectors */\n", modes->Fs, modes->mdctSize);
      fprintf(file, "0,\t/* bits */\n");
      fprintf(file, "{%d, 0, 0},\t/* mdct */\n", 2*modes->mdctSize);
      fprintf(file, "window%d,\t/* window */\n", modes->overlap);
      fprintf(file, "0x%x,\t/* marker */\n", 0xa110ca7e);
      fprintf(file, "};\n");
      modes++;
   }
}

#if 0
int main()
{
   CELTMode *m = celt_mode_create(44100, 1, 256, 128, NULL);
   dump_modes(stdout, m, 1);
   return 0;
}
#endif
