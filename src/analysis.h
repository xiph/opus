/* Copyright (c) 2011 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ANALYSIS_H
#define ANALYSIS_H

#define NB_FRAMES 8
#define NB_TBANDS 18
#define NB_TOT_BANDS 21

typedef struct {
   float angle[240];
   float d_angle[240];
   float d2_angle[240];
   float prev_band_tonality[NB_TBANDS];
   float prev_tonality;
   float E[NB_FRAMES][NB_TBANDS];
   float lowE[NB_TBANDS], highE[NB_TBANDS];
   float meanE[NB_TOT_BANDS];
   float mem[32];
   float cmean[8];
   float std[9];
   float music_prob;
   float Etracker;
   float lowECount;
   int E_count;
   int last_music;
   int last_transition;
   int count;
   int opus_bandwidth;
} TonalityAnalysisState;

void tonality_analysis(TonalityAnalysisState *tonal, AnalysisInfo *info,
     CELTEncoder *celt_enc, const opus_val16 *x, int C, int lsb_depth);

#endif
