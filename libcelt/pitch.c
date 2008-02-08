/**
   @file pitch.c
   @brief Pitch analysis
*/

/* Copyright (C) 2005

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/


#include <stdio.h>
#include <math.h>
#include "pitch.h"
#include "psy.h"

void find_spectral_pitch(kiss_fftr_cfg fft, float *x, float *y, int lag, int len, int C, int *pitch)
{
   int c;
   int n2 = lag/2;
   float xx[lag*C];
   float yy[lag*C];
   float X[lag*C];
   float Y[lag*C];
   float curve[n2*C];
   int i;
   
   for (i=0;i<C*lag;i++)
      xx[i] = 0;
   for (c=0;c<C;c++)
   {
      for (i=0;i<len;i++)
         xx[c*lag+i] = x[C*i+c];
      for (i=0;i<lag;i++)
         yy[c*lag+i] = y[C*i+c];
      
   }
   
   kiss_fftr(fft, xx, X);
   kiss_fftr(fft, yy, Y);
   
   compute_masking(X, curve, lag*C, 44100);
   
   for (i=1;i<C*n2;i++)
   {
      float n;
      //n = 1.f/(1e1+sqrt(sqrt((X[2*i-1]*X[2*i-1] + X[2*i  ]*X[2*i  ])*(Y[2*i-1]*Y[2*i-1] + Y[2*i  ]*Y[2*i  ]))));
      //n = 1;
      n = 1.f/pow(1+curve[i],.5)/(i+60);
      //n = 1.f/(1+curve[i]);
      float tmp = X[2*i];
      X[2*i] = (X[2*i  ]*Y[2*i  ] + X[2*i+1]*Y[2*i+1])*n;
      X[2*i+1] = (- X[2*i+1]*Y[2*i  ] + tmp*Y[2*i+1])*n;
   }
   X[0] = X[1] = 0;
   kiss_fftri(fft, X, xx);
   
   float max_corr=-1e10;
   //int pitch;
   *pitch = 0;
   for (i=0;i<lag-len;i++)
   {
      //printf ("%f ", xx[i]);
      if (xx[i] > max_corr)
      {
         *pitch = i;
         max_corr = xx[i];
      }
   }
   //printf ("\n");
   //printf ("%d %f\n", *pitch, max_corr);
   //printf ("%d\n", *pitch);
}
