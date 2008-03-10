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
#include "rate.h"

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


void dump_modes(FILE *file, CELTMode **modes, int nb_modes)
{
   int i, j;
   fprintf(file, "#include \"modes.h\"\n");
   fprintf(file, "#include \"rate.h\"\n");

   fprintf(file, "\n");
   for (i=0;i<nb_modes;i++)
   {
      CELTMode *mode = modes[i];
      fprintf(file, "#ifndef DEF_EBANDS%d_%d\n", mode->Fs, mode->mdctSize);
      fprintf(file, "#define DEF_EBANDS%d_%d\n", mode->Fs, mode->mdctSize);
      fprintf (file, "const int eBands%d_%d[%d] = {\n", mode->Fs, mode->mdctSize, mode->nbEBands+2);
      for (j=0;j<mode->nbEBands+2;j++)
         fprintf (file, "%d, ", mode->eBands[j]);
      fprintf (file, "};\n");
      fprintf(file, "#endif\n");
      fprintf(file, "\n");
      
      
      fprintf(file, "#ifndef DEF_PBANDS%d_%d\n", mode->Fs, mode->mdctSize);
      fprintf(file, "#define DEF_PBANDS%d_%d\n", mode->Fs, mode->mdctSize);
      fprintf (file, "const int pBands%d_%d[%d] = {\n", mode->Fs, mode->mdctSize, mode->nbPBands+2);
      for (j=0;j<mode->nbPBands+2;j++)
         fprintf (file, "%d, ", mode->pBands[j]);
      printf ("};\n");
      fprintf(file, "#endif\n");
      fprintf(file, "\n");
      
      
      fprintf(file, "#ifndef DEF_WINDOW%d\n", mode->overlap);
      fprintf(file, "#define DEF_WINDOW%d\n", mode->overlap);
      fprintf (file, "const celt_word16_t window%d[%d] = {\n", mode->overlap, mode->overlap);
      for (j=0;j<mode->overlap;j++)
         fprintf (file, WORD16 ", ", mode->window[j]);
      printf ("};\n");
      fprintf(file, "#endif\n");
      fprintf(file, "\n");
      
      fprintf(file, "#ifndef DEF_PSY%d\n", mode->Fs);
      fprintf(file, "#define DEF_PSY%d\n", mode->Fs);
      fprintf (file, "const celt_word16_t psy_decayR_%d[%d] = {\n", mode->Fs, MAX_PERIOD/2);
      for (j=0;j<MAX_PERIOD/2;j++)
         fprintf (file, WORD16 ", ", mode->psy.decayR[j]);
      printf ("};\n");
      fprintf(file, "#endif\n");
      fprintf(file, "\n");

      
      fprintf(file, "#ifndef DEF_ALLOC_VECTORS%d_%d\n", mode->Fs, mode->mdctSize);
      fprintf(file, "#define DEF_ALLOC_VECTORS%d_%d\n", mode->Fs, mode->mdctSize);
      fprintf (file, "const int allocVectors%d_%d[%d] = {\n", mode->Fs, mode->mdctSize, mode->nbEBands*mode->nbAllocVectors);
      for (j=0;j<mode->nbAllocVectors;j++)
      {
         int k;
         for (k=0;k<mode->nbEBands;k++)
            fprintf (file, "%2d, ", mode->allocVectors[j*mode->nbEBands+k]);
         fprintf (file, "\n");
      }
      fprintf (file, "};\n");
      fprintf(file, "#endif\n");
      fprintf(file, "\n");
      
      fprintf(file, "#ifndef DEF_ALLOC_CACHE%d_%d_%d\n", mode->Fs, mode->mdctSize, mode->nbChannels);
      fprintf(file, "#define DEF_ALLOC_CACHE%d_%d_%d\n", mode->Fs, mode->mdctSize, mode->nbChannels);
      for (j=0;j<mode->nbEBands;j++)
      {
         int k;
         if (j==0 || (mode->bits[j] != mode->bits[j-1]))
         {
            fprintf (file, "const int allocCache_band%d_%d_%d_%d[MAX_PULSES] = {\n", j, mode->Fs, mode->mdctSize, mode->nbChannels);
            for (k=0;k<MAX_PULSES;k++)
               fprintf (file, "%2d, ", mode->bits[j][k]);
            fprintf (file, "};\n");
         } else {
            fprintf (file, "#define allocCache_band%d_%d_%d_%d allocCache_band%d_%d_%d_%d\n", j, mode->Fs, mode->mdctSize, mode->nbChannels, j-1, mode->Fs, mode->mdctSize, mode->nbChannels);
         }
      }
      fprintf (file, "const int *allocCache%d_%d_%d[%d] = {\n", mode->Fs, mode->mdctSize, mode->nbChannels, mode->nbEBands);
      for (j=0;j<mode->nbEBands;j++)
      {
         fprintf (file, "allocCache_band%d_%d_%d_%d, ", j, mode->Fs, mode->mdctSize, mode->nbChannels);
      }
      fprintf (file, "};\n");
      fprintf(file, "#endif\n");
      fprintf(file, "\n");

      
      fprintf(file, "CELTMode mode%d_%d_%d_%d = {\n", mode->Fs, mode->nbChannels, mode->mdctSize, mode->overlap);
      fprintf(file, "0x%x,\t/* marker */\n", 0xa110ca7e);
      fprintf(file, INT32 ",\t/* Fs */\n", mode->Fs);
      fprintf(file, "%d,\t/* overlap */\n", mode->overlap);
      fprintf(file, "%d,\t/* mdctSize */\n", mode->mdctSize);
      fprintf(file, "%d,\t/* nbMdctBlocks */\n", mode->nbMdctBlocks);
      fprintf(file, "%d,\t/* nbChannels */\n", mode->nbChannels);
      fprintf(file, "%d,\t/* nbEBands */\n", mode->nbEBands);
      fprintf(file, "%d,\t/* nbPBands */\n", mode->nbPBands);
      fprintf(file, "%d,\t/* pitchEnd */\n", mode->pitchEnd);
      fprintf(file, "eBands%d_%d,\t/* eBands */\n", mode->Fs, mode->mdctSize);
      fprintf(file, "pBands%d_%d,\t/* pBands */\n", mode->Fs, mode->mdctSize);
      fprintf(file, WORD16 ",\t/* ePredCoef */\n", mode->ePredCoef);
      fprintf(file, "%d,\t/* nbAllocVectors */\n", mode->nbAllocVectors);
      fprintf(file, "allocVectors%d_%d,\t/* allocVectors */\n", mode->Fs, mode->mdctSize);
      fprintf(file, "allocCache%d_%d_%d,\t/* bits */\n", mode->Fs, mode->mdctSize, mode->nbChannels);
      fprintf(file, "{%d, 0, 0},\t/* mdct */\n", 2*mode->mdctSize);
      fprintf(file, "window%d,\t/* window */\n", mode->overlap);
      fprintf(file, "{psy_decayR_%d},\t/* psy */\n", mode->Fs);
      fprintf(file, "0x%x,\t/* marker */\n", 0xa110ca7e);
      fprintf(file, "};\n");
   }
   fprintf(file, "\n");
   fprintf(file, "/* List of all the available modes */\n");
   fprintf(file, "#define TOTAL_MODES %d\n", nb_modes);
   fprintf(file, "const CELTMode *static_mode_list[TOTAL_MODES] = {\n");
   for (i=0;i<nb_modes;i++)
   {
      CELTMode *mode = modes[i];
      fprintf(file, "&mode%d_%d_%d_%d,\n", mode->Fs, mode->nbChannels, mode->mdctSize, mode->overlap);
   }
   fprintf(file, "};\n");
}

#if 0
int main()
{
   CELTMode *m[3];
   m[0] = celt_mode_create(44100, 1, 256, 128, NULL);
   m[1] = celt_mode_create(48000, 1, 256, 128, NULL);
   m[2] = celt_mode_create(51200, 1, 256, 128, NULL);
   dump_modes(stdout, m, 3);
   return 0;
}
#endif
