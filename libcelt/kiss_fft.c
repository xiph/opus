/*
Copyright (c) 2003-2004, Mark Borgerding
Lots of modifications by JMV
Copyright (c) 2005-2007, Jean-Marc Valin
Copyright (c) 2008,      Jean-Marc Valin, CSIRO

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef SKIP_CONFIG_H
#  ifdef HAVE_CONFIG_H
#    include "config.h"
#  endif
#endif

#include "_kiss_fft_guts.h"
#include "arch.h"
#include "os_support.h"
#include "mathops.h"
#include "stack_alloc.h"

/* The guts header contains all the multiplication and addition macros that are defined for
   complex numbers.  It also delares the kf_ internal functions.
*/

static void kf_bfly2(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     int m,
                     int N,
                     int mm
                    )
{
   kiss_fft_cpx * Fout2;
   kiss_twiddle_cpx * tw1;
   int i,j;
   kiss_fft_cpx * Fout_beg = Fout;
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      Fout2 = Fout + m;
      tw1 = st->twiddles;
      for(j=0;j<m;j++)
      {
         kiss_fft_cpx t;
         Fout->r = SHR(Fout->r, 1);Fout->i = SHR(Fout->i, 1);
         Fout2->r = SHR(Fout2->r, 1);Fout2->i = SHR(Fout2->i, 1);
         C_MUL (t,  *Fout2 , *tw1);
         tw1 += fstride;
         C_SUB( *Fout2 ,  *Fout , t );
         C_ADDTO( *Fout ,  t );
         ++Fout2;
         ++Fout;
      }
   }
}

static void ki_bfly2(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     int m,
                     int N,
                     int mm
                    )
{
   kiss_fft_cpx * Fout2;
   kiss_twiddle_cpx * tw1;
   kiss_fft_cpx t;
   int i,j;
   kiss_fft_cpx * Fout_beg = Fout;
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      Fout2 = Fout + m;
      tw1 = st->twiddles;
      for(j=0;j<m;j++)
      {
         C_MULC (t,  *Fout2 , *tw1);
         tw1 += fstride;
         C_SUB( *Fout2 ,  *Fout , t );
         C_ADDTO( *Fout ,  t );
         ++Fout2;
         ++Fout;
      }
   }
}

static void kf_bfly4(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     int m,
                     int N,
                     int mm
                    )
{
   kiss_twiddle_cpx *tw1,*tw2,*tw3;
   kiss_fft_cpx scratch[6];
   const size_t m2=2*m;
   const size_t m3=3*m;
   int i, j;

   kiss_fft_cpx * Fout_beg = Fout;
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      tw3 = tw2 = tw1 = st->twiddles;
      for (j=0;j<m;j++)
      {
         C_MUL4(scratch[0],Fout[m] , *tw1 );
         C_MUL4(scratch[1],Fout[m2] , *tw2 );
         C_MUL4(scratch[2],Fout[m3] , *tw3 );
             
         Fout->r = PSHR(Fout->r, 2);
         Fout->i = PSHR(Fout->i, 2);
         C_SUB( scratch[5] , *Fout, scratch[1] );
         C_ADDTO(*Fout, scratch[1]);
         C_ADD( scratch[3] , scratch[0] , scratch[2] );
         C_SUB( scratch[4] , scratch[0] , scratch[2] );
         Fout[m2].r = PSHR(Fout[m2].r, 2);
         Fout[m2].i = PSHR(Fout[m2].i, 2);
         C_SUB( Fout[m2], *Fout, scratch[3] );
         tw1 += fstride;
         tw2 += fstride*2;
         tw3 += fstride*3;
         C_ADDTO( *Fout , scratch[3] );
             
         Fout[m].r = scratch[5].r + scratch[4].i;
         Fout[m].i = scratch[5].i - scratch[4].r;
         Fout[m3].r = scratch[5].r - scratch[4].i;
         Fout[m3].i = scratch[5].i + scratch[4].r;
         ++Fout;
      }
   }
}

static void ki_bfly4(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     int m,
                     int N,
                     int mm
                    )
{
   kiss_twiddle_cpx *tw1,*tw2,*tw3;
   kiss_fft_cpx scratch[6];
   const size_t m2=2*m;
   const size_t m3=3*m;
   int i, j;

   kiss_fft_cpx * Fout_beg = Fout;
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      tw3 = tw2 = tw1 = st->twiddles;
      for (j=0;j<m;j++)
      {
         C_MULC(scratch[0],Fout[m] , *tw1 );
         C_MULC(scratch[1],Fout[m2] , *tw2 );
         C_MULC(scratch[2],Fout[m3] , *tw3 );
             
         C_SUB( scratch[5] , *Fout, scratch[1] );
         C_ADDTO(*Fout, scratch[1]);
         C_ADD( scratch[3] , scratch[0] , scratch[2] );
         C_SUB( scratch[4] , scratch[0] , scratch[2] );
         C_SUB( Fout[m2], *Fout, scratch[3] );
         tw1 += fstride;
         tw2 += fstride*2;
         tw3 += fstride*3;
         C_ADDTO( *Fout , scratch[3] );
             
         Fout[m].r = scratch[5].r - scratch[4].i;
         Fout[m].i = scratch[5].i + scratch[4].r;
         Fout[m3].r = scratch[5].r + scratch[4].i;
         Fout[m3].i = scratch[5].i - scratch[4].r;
         ++Fout;
      }
   }
}

#ifndef RADIX_TWO_ONLY

static void kf_bfly3(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     size_t m
                    )
{
   size_t k=m;
   const size_t m2 = 2*m;
   kiss_twiddle_cpx *tw1,*tw2;
   kiss_fft_cpx scratch[5];
   kiss_twiddle_cpx epi3;
   epi3 = st->twiddles[fstride*m];

   tw1=tw2=st->twiddles;
   do{
      C_FIXDIV(*Fout,3); C_FIXDIV(Fout[m],3); C_FIXDIV(Fout[m2],3);

      C_MUL(scratch[1],Fout[m] , *tw1);
      C_MUL(scratch[2],Fout[m2] , *tw2);

      C_ADD(scratch[3],scratch[1],scratch[2]);
      C_SUB(scratch[0],scratch[1],scratch[2]);
      tw1 += fstride;
      tw2 += fstride*2;

      Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
      Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

      C_MULBYSCALAR( scratch[0] , epi3.i );

      C_ADDTO(*Fout,scratch[3]);

      Fout[m2].r = Fout[m].r + scratch[0].i;
      Fout[m2].i = Fout[m].i - scratch[0].r;

      Fout[m].r -= scratch[0].i;
      Fout[m].i += scratch[0].r;

      ++Fout;
   }while(--k);
}

static void ki_bfly3(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     size_t m
                    )
{
   size_t k=m;
   const size_t m2 = 2*m;
   kiss_twiddle_cpx *tw1,*tw2;
   kiss_fft_cpx scratch[5];
   kiss_twiddle_cpx epi3;
   epi3 = st->twiddles[fstride*m];

   tw1=tw2=st->twiddles;
   do{

      C_MULC(scratch[1],Fout[m] , *tw1);
      C_MULC(scratch[2],Fout[m2] , *tw2);

      C_ADD(scratch[3],scratch[1],scratch[2]);
      C_SUB(scratch[0],scratch[1],scratch[2]);
      tw1 += fstride;
      tw2 += fstride*2;

      Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
      Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

      C_MULBYSCALAR( scratch[0] , -epi3.i );

      C_ADDTO(*Fout,scratch[3]);

      Fout[m2].r = Fout[m].r + scratch[0].i;
      Fout[m2].i = Fout[m].i - scratch[0].r;

      Fout[m].r -= scratch[0].i;
      Fout[m].i += scratch[0].r;

      ++Fout;
   }while(--k);
}


static void kf_bfly5(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     int m
                    )
{
   kiss_fft_cpx *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
   int u;
   kiss_fft_cpx scratch[13];
   kiss_twiddle_cpx * twiddles = st->twiddles;
   kiss_twiddle_cpx *tw;
   kiss_twiddle_cpx ya,yb;
   ya = twiddles[fstride*m];
   yb = twiddles[fstride*2*m];

   Fout0=Fout;
   Fout1=Fout0+m;
   Fout2=Fout0+2*m;
   Fout3=Fout0+3*m;
   Fout4=Fout0+4*m;

   tw=st->twiddles;
   for ( u=0; u<m; ++u ) {
      C_FIXDIV( *Fout0,5); C_FIXDIV( *Fout1,5); C_FIXDIV( *Fout2,5); C_FIXDIV( *Fout3,5); C_FIXDIV( *Fout4,5);
      scratch[0] = *Fout0;

      C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
      C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
      C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
      C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);

      C_ADD( scratch[7],scratch[1],scratch[4]);
      C_SUB( scratch[10],scratch[1],scratch[4]);
      C_ADD( scratch[8],scratch[2],scratch[3]);
      C_SUB( scratch[9],scratch[2],scratch[3]);

      Fout0->r += scratch[7].r + scratch[8].r;
      Fout0->i += scratch[7].i + scratch[8].i;

      scratch[5].r = scratch[0].r + S_MUL(scratch[7].r,ya.r) + S_MUL(scratch[8].r,yb.r);
      scratch[5].i = scratch[0].i + S_MUL(scratch[7].i,ya.r) + S_MUL(scratch[8].i,yb.r);

      scratch[6].r =  S_MUL(scratch[10].i,ya.i) + S_MUL(scratch[9].i,yb.i);
      scratch[6].i = -S_MUL(scratch[10].r,ya.i) - S_MUL(scratch[9].r,yb.i);

      C_SUB(*Fout1,scratch[5],scratch[6]);
      C_ADD(*Fout4,scratch[5],scratch[6]);

      scratch[11].r = scratch[0].r + S_MUL(scratch[7].r,yb.r) + S_MUL(scratch[8].r,ya.r);
      scratch[11].i = scratch[0].i + S_MUL(scratch[7].i,yb.r) + S_MUL(scratch[8].i,ya.r);
      scratch[12].r = - S_MUL(scratch[10].i,yb.i) + S_MUL(scratch[9].i,ya.i);
      scratch[12].i = S_MUL(scratch[10].r,yb.i) - S_MUL(scratch[9].r,ya.i);

      C_ADD(*Fout2,scratch[11],scratch[12]);
      C_SUB(*Fout3,scratch[11],scratch[12]);

      ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
   }
}

static void ki_bfly5(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_cfg st,
                     int m
                    )
{
   kiss_fft_cpx *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
   int u;
   kiss_fft_cpx scratch[13];
   kiss_twiddle_cpx * twiddles = st->twiddles;
   kiss_twiddle_cpx *tw;
   kiss_twiddle_cpx ya,yb;
   ya = twiddles[fstride*m];
   yb = twiddles[fstride*2*m];

   Fout0=Fout;
   Fout1=Fout0+m;
   Fout2=Fout0+2*m;
   Fout3=Fout0+3*m;
   Fout4=Fout0+4*m;

   tw=st->twiddles;
   for ( u=0; u<m; ++u ) {
      scratch[0] = *Fout0;

      C_MULC(scratch[1] ,*Fout1, tw[u*fstride]);
      C_MULC(scratch[2] ,*Fout2, tw[2*u*fstride]);
      C_MULC(scratch[3] ,*Fout3, tw[3*u*fstride]);
      C_MULC(scratch[4] ,*Fout4, tw[4*u*fstride]);

      C_ADD( scratch[7],scratch[1],scratch[4]);
      C_SUB( scratch[10],scratch[1],scratch[4]);
      C_ADD( scratch[8],scratch[2],scratch[3]);
      C_SUB( scratch[9],scratch[2],scratch[3]);

      Fout0->r += scratch[7].r + scratch[8].r;
      Fout0->i += scratch[7].i + scratch[8].i;

      scratch[5].r = scratch[0].r + S_MUL(scratch[7].r,ya.r) + S_MUL(scratch[8].r,yb.r);
      scratch[5].i = scratch[0].i + S_MUL(scratch[7].i,ya.r) + S_MUL(scratch[8].i,yb.r);

      scratch[6].r = -S_MUL(scratch[10].i,ya.i) - S_MUL(scratch[9].i,yb.i);
      scratch[6].i =  S_MUL(scratch[10].r,ya.i) + S_MUL(scratch[9].r,yb.i);

      C_SUB(*Fout1,scratch[5],scratch[6]);
      C_ADD(*Fout4,scratch[5],scratch[6]);

      scratch[11].r = scratch[0].r + S_MUL(scratch[7].r,yb.r) + S_MUL(scratch[8].r,ya.r);
      scratch[11].i = scratch[0].i + S_MUL(scratch[7].i,yb.r) + S_MUL(scratch[8].i,ya.r);
      scratch[12].r =  S_MUL(scratch[10].i,yb.i) - S_MUL(scratch[9].i,ya.i);
      scratch[12].i = -S_MUL(scratch[10].r,yb.i) + S_MUL(scratch[9].r,ya.i);

      C_ADD(*Fout2,scratch[11],scratch[12]);
      C_SUB(*Fout3,scratch[11],scratch[12]);

      ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
   }
}

/* perform the butterfly for one stage of a mixed radix FFT */
static void kf_bfly_generic(
                            kiss_fft_cpx * Fout,
                            const size_t fstride,
                            const kiss_fft_cfg st,
                            int m,
                            int p
                           )
{
   int u,k,q1,q;
   kiss_twiddle_cpx * twiddles = st->twiddles;
   kiss_fft_cpx t;
   VARDECL(kiss_fft_cpx, scratchbuf);
   int Norig = st->nfft;
   ALLOC(scratchbuf, p, kiss_fft_cpx);

   for ( u=0; u<m; ++u ) {
      k=u;
      for ( q1=0 ; q1<p ; ++q1 ) {
         scratchbuf[q1] = Fout[ k  ];
         C_FIXDIV(scratchbuf[q1],p);
         k += m;
      }

      k=u;
      for ( q1=0 ; q1<p ; ++q1 ) {
         int twidx=0;
         Fout[ k ] = scratchbuf[0];
         for (q=1;q<p;++q ) {
            twidx += fstride * k;
            if (twidx>=Norig) twidx-=Norig;
            C_MUL(t,scratchbuf[q] , twiddles[twidx] );
            C_ADDTO( Fout[ k ] ,t);
         }
         k += m;
      }
   }
}

static void ki_bfly_generic(
                            kiss_fft_cpx * Fout,
                            const size_t fstride,
                            const kiss_fft_cfg st,
                            int m,
                            int p
                           )
{
   int u,k,q1,q;
   kiss_twiddle_cpx * twiddles = st->twiddles;
   kiss_fft_cpx t;
   VARDECL(kiss_fft_cpx, scratchbuf);
   int Norig = st->nfft;
   ALLOC(scratchbuf, p, kiss_fft_cpx);

   for ( u=0; u<m; ++u ) {
      k=u;
      for ( q1=0 ; q1<p ; ++q1 ) {
         scratchbuf[q1] = Fout[ k  ];
         k += m;
      }

      k=u;
      for ( q1=0 ; q1<p ; ++q1 ) {
         int twidx=0;
         Fout[ k ] = scratchbuf[0];
         for (q=1;q<p;++q ) {
            twidx += fstride * k;
            if (twidx>=Norig) twidx-=Norig;
            C_MULC(t,scratchbuf[q] , twiddles[twidx] );
            C_ADDTO( Fout[ k ] ,t);
         }
         k += m;
      }
   }
}
#endif

static
void compute_bitrev_table(
         int Fout,
         int *f,
         const size_t fstride,
         int in_stride,
         int * factors,
         const kiss_fft_cfg st
            )
{
   const int p=*factors++; /* the radix  */
   const int m=*factors++; /* stage's fft length/p */
   
    /*printf ("fft %d %d %d %d %d %d\n", p*m, m, p, s2, fstride*in_stride, N);*/
   if (m==1)
   {
      int j;
      for (j=0;j<p;j++)
      {
         *f = Fout+j;
         f += fstride*in_stride;
      }
   } else {
      int j;
      for (j=0;j<p;j++)
      {
         compute_bitrev_table( Fout , f, fstride*p, in_stride, factors,st);
         f += fstride*in_stride;
         Fout += m;
      }
   }
}


void kf_work(
        kiss_fft_cpx * Fout,
        const kiss_fft_cpx * f,
        const size_t fstride,
        int in_stride,
        int * factors,
        const kiss_fft_cfg st,
        int N,
        int s2,
        int m2
        )
{
#ifndef RADIX_TWO_ONLY
    int i;
    kiss_fft_cpx * Fout_beg=Fout;
#endif
    const int p=*factors++; /* the radix  */
    const int m=*factors++; /* stage's fft length/p */
    /*printf ("fft %d %d %d %d %d %d %d\n", p*m, m, p, s2, fstride*in_stride, N, m2);*/
    if (m!=1) 
        kf_work( Fout , f, fstride*p, in_stride, factors,st, N*p, fstride*in_stride, m);

    switch (p) {
        case 2: kf_bfly2(Fout,fstride,st,m, N, m2); break;
        case 4: kf_bfly4(Fout,fstride,st,m, N, m2); break;
#ifndef RADIX_TWO_ONLY
        case 3: for (i=0;i<N;i++){Fout=Fout_beg+i*m2; kf_bfly3(Fout,fstride,st,m);} break; 
        case 5: for (i=0;i<N;i++){Fout=Fout_beg+i*m2; kf_bfly5(Fout,fstride,st,m);} break; 
        default: for (i=0;i<N;i++){Fout=Fout_beg+i*m2; kf_bfly_generic(Fout,fstride,st,m,p);} break;
#else
       default: celt_fatal("kiss_fft: only powers of two enabled");
#endif
    }    
}


void ki_work(
             kiss_fft_cpx * Fout,
             const kiss_fft_cpx * f,
             const size_t fstride,
             int in_stride,
             int * factors,
             const kiss_fft_cfg st,
             int N,
             int s2,
             int m2
            )
{
#ifndef RADIX_TWO_ONLY
   int i;
   kiss_fft_cpx * Fout_beg=Fout;
#endif
   const int p=*factors++; /* the radix  */
   const int m=*factors++; /* stage's fft length/p */
   /*printf ("fft %d %d %d %d %d %d %d\n", p*m, m, p, s2, fstride*in_stride, N, m2);*/
   if (m!=1) 
      ki_work( Fout , f, fstride*p, in_stride, factors,st, N*p, fstride*in_stride, m);

   switch (p) {
      case 2: ki_bfly2(Fout,fstride,st,m, N, m2); break;
      case 4: ki_bfly4(Fout,fstride,st,m, N, m2); break;
#ifndef RADIX_TWO_ONLY
      case 3: for (i=0;i<N;i++){Fout=Fout_beg+i*m2; ki_bfly3(Fout,fstride,st,m);} break; 
      case 5: for (i=0;i<N;i++){Fout=Fout_beg+i*m2; ki_bfly5(Fout,fstride,st,m);} break; 
      default: for (i=0;i<N;i++){Fout=Fout_beg+i*m2; ki_bfly_generic(Fout,fstride,st,m,p);} break;
#else
      default: celt_fatal("kiss_fft: only powers of two enabled");
#endif
   }    
}

/*  facbuf is populated by p1,m1,p2,m2, ...
    where 
    p[i] * m[i] = m[i-1]
    m0 = n                  */
static 
void kf_factor(int n,int * facbuf)
{
    int p=4;

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
                case 4: p = 2; break;
                case 2: p = 3; break;
                default: p += 2; break;
            }
            if (p>32000 || (celt_int32_t)p*(celt_int32_t)p > n)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
    } while (n > 1);
}
/*
 *
 * User-callable function to allocate all necessary storage space for the fft.
 *
 * The return value is a contiguous block of memory, allocated with malloc.  As such,
 * It can be freed with free(), rather than a kiss_fft-specific function.
 * */
kiss_fft_cfg kiss_fft_alloc(int nfft,void * mem,size_t * lenmem )
{
    kiss_fft_cfg st=NULL;
    size_t memneeded = sizeof(struct kiss_fft_state)
          + sizeof(kiss_twiddle_cpx)*(nfft-1) + sizeof(int)*nfft; /* twiddle factors*/

    if ( lenmem==NULL ) {
        st = ( kiss_fft_cfg)KISS_FFT_MALLOC( memneeded );
    }else{
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_fft_cfg)mem;
        *lenmem = memneeded;
    }
    if (st) {
        int i;
        st->nfft=nfft;
#ifndef FIXED_POINT
        st->scale = 1./nfft;
#endif
#if defined(FIXED_POINT) && (!defined(DOUBLE_PRECISION) || defined(MIXED_PRECISION))
        for (i=0;i<nfft;++i) {
            celt_word32_t phase = -i;
            kf_cexp2(st->twiddles+i, DIV32(SHL32(phase,17),nfft));
        }
#else
        for (i=0;i<nfft;++i) {
           const double pi=3.14159265358979323846264338327;
           double phase = ( -2*pi /nfft ) * i;
           kf_cexp(st->twiddles+i, phase );
        }
#endif
        kf_factor(nfft,st->factors);
        
        /* bitrev */
        st->bitrev = (int*)((char*)st + memneeded - sizeof(int)*nfft);
        compute_bitrev_table(0, st->bitrev, 1,1, st->factors,st);
    }
    return st;
}



    
void kiss_fft_stride(kiss_fft_cfg st,const kiss_fft_cpx *fin,kiss_fft_cpx *fout,int in_stride)
{
    if (fin == fout) 
    {
       celt_fatal("In-place FFT not supported");
    } else {
       /* Bit-reverse the input */
       int i;
       for (i=0;i<st->nfft;i++)
       {
          fout[st->bitrev[i]] = fin[i];
#ifndef FIXED_POINT
          fout[st->bitrev[i]].r *= st->scale;
          fout[st->bitrev[i]].i *= st->scale;
#endif
       }
       kf_work( fout, fin, 1,in_stride, st->factors,st, 1, in_stride, 1);
    }
}

void kiss_fft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout)
{
    kiss_fft_stride(cfg,fin,fout,1);
}

void kiss_ifft_stride(kiss_fft_cfg st,const kiss_fft_cpx *fin,kiss_fft_cpx *fout,int in_stride)
{
   if (fin == fout) 
   {
      celt_fatal("In-place FFT not supported");
   } else {
      /* Bit-reverse the input */
      int i;
      for (i=0;i<st->nfft;i++)
         fout[st->bitrev[i]] = fin[i];
      ki_work( fout, fin, 1,in_stride, st->factors,st, 1, in_stride, 1);
   }
}

void kiss_ifft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout)
{
   kiss_ifft_stride(cfg,fin,fout,1);
}

