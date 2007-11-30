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
#include "pitch.h"

void find_spectral_pitch(void *fft, float *x, float *y, int lag, int len, int *pitch)
{
   int n2 = lag/2;
   float xx[lag];
   float X[lag];
   float Y[lag];
   float curve[n2];
   int i;
   
   for (i=0;i<lag;i++)
      xx[i] = 0;
   for (i=0;i<len;i++)
      xx[i] = x[i];
   
   spx_fft(fft, xx, X);
   spx_fft(fft, y, Y);
   curve[0] = 1;
   for (i=1;i<n2;i++)
   {
      curve[i] = sqrt((X[2*i-1]*X[2*i-1] + X[2*i  ]*X[2*i  ])*(Y[2*i-1]*Y[2*i-1] + Y[2*i  ]*Y[2*i  ]));
      curve[i] = curve[i]+.7*curve[i];
   }
   for (i=n2-2;i>=0;i--)
      curve[i] = curve[i] + .7*curve[i+1];
   
   X[0] = 0;
   for (i=1;i<lag/2;i++)
   {
      float n;
      //n = 1.f/(1e1+sqrt(sqrt((X[2*i-1]*X[2*i-1] + X[2*i  ]*X[2*i  ])*(Y[2*i-1]*Y[2*i-1] + Y[2*i  ]*Y[2*i  ]))));
      //n = 1;
      n = 1.f/pow(1+curve[i],.8)/(i+60);
      //if (i>lag/6)
      //   n *= .5;
      float tmp = X[2*i-1];
      X[2*i-1] = (X[2*i-1]*Y[2*i-1] + X[2*i  ]*Y[2*i  ])*n;
      X[2*i  ] = (- X[2*i  ]*Y[2*i-1] + tmp*Y[2*i  ])*n;
   }
   X[lag-1] = 0;
   X[0] = X[lag-1] = 0;
   spx_ifft(fft, X, xx);
   
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
