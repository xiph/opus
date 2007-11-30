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
#include "fftwrap.h"


void find_spectral_pitch(float *x, float *y, int lag, int len, int *pitch, float *curve)
{
   //FIXME: Yuck!!!
   static void *fft;
   
   if (!fft)
      fft = spx_fft_init(lag);
   
   float xx[lag];
   float X[lag];
   float Y[lag];
   int i;
   
   for (i=0;i<lag;i++)
      xx[i] = 0;
   for (i=0;i<len;i++)
      xx[i] = x[i];
   
   spx_fft(fft, xx, X);
   spx_fft(fft, y, Y);
   X[0] = X[0]*Y[0];
   for (i=1;i<lag/2;i++)
   {
      float n = 1.f/(1e1+sqrt((X[2*i-1]*X[2*i-1] + X[2*i  ]*X[2*i  ])*(Y[2*i-1]*Y[2*i-1] + Y[2*i  ]*Y[2*i  ])));
      //n = 1;
      //n = 1.f/(1+curve[i]);
      
      float tmp = X[2*i-1];
      X[2*i-1] = (X[2*i-1]*Y[2*i-1] + X[2*i  ]*Y[2*i  ])*n;
      X[2*i  ] = (- X[2*i  ]*Y[2*i-1] + tmp*Y[2*i  ])*n;
   }
   X[len-1] = 0;
   X[0] = X[len-1] = 0;
   spx_ifft(fft, X, xx);
   
   float max_corr=-1;
   //int pitch;
   *pitch = -1;
   for (i=0;i<lag-len;i++)
   {
      //printf ("%f ", xx[i]);
      if (xx[i] > max_corr)
      {
         *pitch = i;
         max_corr = xx[i];
      }
   }
   //printf ("%d\n", *pitch);
}
