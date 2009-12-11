



celt_word32 _celt_lpc(
celt_word16       *lpc, /* out: [0...p-1] LPC coefficients      */
const celt_word16 *ac,  /* in:  [0...p] autocorrelation values  */
int          p
)
{
   int i, j;  
   celt_word16 r;
   celt_word16 error = ac[0];

   if (ac[0] == 0)
   {
      for (i = 0; i < p; i++)
         lpc[i] = 0;
      return 0;
   }
   
   for (i = 0; i < p; i++) {
      
      /* Sum up this iteration's reflection coefficient */
      celt_word32 rr = NEG32(SHL32(EXTEND32(ac[i + 1]),13));
      for (j = 0; j < i; j++) 
         rr = SUB32(rr,MULT16_16(lpc[j],ac[i - j]));
#ifdef FIXED_POINT
      r = DIV32_16(rr+PSHR32(error,1),ADD16(error,8));
#else
      r = rr/(error+.003*ac[0]);
#endif
      /*  Update LPC coefficients and total error */
      lpc[i] = r;
      for (j = 0; j < i>>1; j++) 
      {
         celt_word16 tmp  = lpc[j];
         lpc[j]     = MAC16_16_P13(lpc[j],r,lpc[i-1-j]);
         lpc[i-1-j] = MAC16_16_P13(lpc[i-1-j],r,tmp);
      }
      if (i & 1) 
         lpc[j] = MAC16_16_P13(lpc[j],lpc[j],r);
      
      error = SUB16(error,MULT16_16_Q13(r,MULT16_16_Q13(error,r)));
   }
   return error;
}

void fir(const celt_word16 *x,
         const celt_word16 *num,
         celt_word16 *y,
         int N,
         int ord,
         celt_word32 *mem)
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

void iir(const celt_word16 *x,
         const celt_word16 *den,
         celt_word16 *y,
         int N,
         int ord,
         celt_word32 *mem)
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
                   const celt_word16 *x,   /*  in: [0...n-1] samples x   */
                   float       *ac,  /* out: [0...lag-1] ac values */
                   const float       *window,
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
      xx[i] *= window[i];
      xx[n-i-1] *= window[i];
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
