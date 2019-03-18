
/* Copyright (c) 2017-2019 Mozilla */
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include "freq.h"
#include "lpcnet_private.h"


static void single_interp(float *x, const float *left, const float *right, int id)
{
    int i;
    float ref[NB_BANDS];
    float pred[3*NB_BANDS];
    RNN_COPY(ref, x, NB_BANDS);
    for (i=0;i<NB_BANDS;i++) pred[i] = .5*(left[i] + right[i]);
    for (i=0;i<NB_BANDS;i++) pred[NB_BANDS+i] = left[i];
    for (i=0;i<NB_BANDS;i++) pred[2*NB_BANDS+i] = right[i];
    for (i=0;i<NB_BANDS;i++) {
      x[i] = pred[id*NB_BANDS + i];
    }
    if (0) {
        float err = 0;
        for (i=0;i<NB_BANDS;i++) {
            err += (x[i]-ref[i])*(x[i]-ref[i]);
        }
        printf("%f\n", sqrt(err/NB_BANDS));
    }
}

void perform_double_interp(float features[4][NB_TOTAL_FEATURES], const float *mem, int best_id) {
    int id0, id1;
    best_id += (best_id >= FORBIDDEN_INTERP);
    id0 = best_id / 3;
    id1 = best_id % 3;
    single_interp(features[0], mem, features[1], id0);
    single_interp(features[2], features[1], features[3], id1);
}
