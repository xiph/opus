
#ifndef NEW_PLC
#define NEW_PLC
#endif

float _celt_lpc(
      float       *lpc, /* out: [0...p-1] LPC coefficients      */
const float *ac,  /* in:  [0...p] autocorrelation values  */
int          p
)
{
   int i, j;  
   float r;
   float error = ac[0];

   if (ac[0] == 0)
   {
      for (i = 0; i < p; i++)
         lpc[i] = 0;
      return 0;
   }
   
   for (i = 0; i < p; i++) {
      
      /* Sum up this iteration's reflection coefficient */
      float rr = -ac[i + 1];
      for (j = 0; j < i; j++) 
         rr = rr - lpc[j]*ac[i - j];
      r = rr/(error+1e-15);
      /*  Update LPC coefficients and total error */
      lpc[i] = r;
      for (j = 0; j < i>>1; j++) 
      {
         float tmp  = lpc[j];
         lpc[j]     = lpc[j    ] + r*lpc[i-1-j];
         lpc[i-1-j] = lpc[i-1-j] + r*tmp;
      }
      if (i & 1) 
         lpc[j] = lpc[j] + lpc[j]*r;
      
      error = error - r*r*error;
      if (error<.00001*ac[0])
         break;
   }
   return error;
}

void fir(const float *x,
         const float *num,
         float *y,
         int N,
         int ord,
         float *mem)
{
   int i,j;

   for (i=0;i<N;i++)
   {
      float sum = x[i];
      for (j=0;j<ord;j++)
      {
         sum += num[j]*mem[j];
      }
      for (j=ord-1;j>=1;j--)
      {
         mem[j]=mem[j-1];
      }
      mem[0] = x[i];
      y[i] = sum;
   }
}

void iir(const celt_word32 *x,
         const float *den,
         celt_word32 *y,
         int N,
         int ord,
         float *mem)
{
   int i,j;
   for (i=0;i<N;i++)
   {
      float sum = x[i];
      for (j=0;j<ord;j++)
      {
         sum -= den[j]*mem[j];
      }
      for (j=ord-1;j>=1;j--)
      {
         mem[j]=mem[j-1];
      }
      mem[0] = sum;
      y[i] = sum;
   }
}

void _celt_autocorr(
                   const float *x,   /*  in: [0...n-1] samples x   */
                   float       *ac,  /* out: [0...lag-1] ac values */
                   const celt_word16       *window,
                   int          overlap,
                   int          lag, 
                   int          n
                  )
{
   float d;
   int i;
   VARDECL(float, xx);
   SAVE_STACK;
   ALLOC(xx, n, float);
   for (i=0;i<n;i++)
      xx[i] = x[i];
   for (i=0;i<overlap;i++)
   {
      xx[i] *= (1./Q15ONE)*window[i];
      xx[n-i-1] *= (1./Q15ONE)*window[i];
   }
   while (lag>=0)
   {
      for (i = lag, d = 0; i < n; i++) 
         d += x[i] * x[i-lag];
      ac[lag] = d;
      lag--;
   }
   ac[0] += 10;
   RESTORE_STACK;
}
