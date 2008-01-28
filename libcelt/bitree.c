#include "bitree.h"

void ec_bitree_to_counts(unsigned *_this,int _sz,int _split){
  int p;
  int q;
  int s;
  for(p=_split;p>1;p=q){
    q=p>>1;
    for(s=p-1;s<_sz;s+=p)_this[s]-=_this[s-q];
  }
}

void ec_bitree_from_counts(unsigned *_this,int _sz){
  int p;
  int q;
  int s;
  for(q=1,p=2;p<=_sz;q=p,p=q<<1){
    for(s=p-1;s<_sz;s+=p)_this[s]+=_this[s-q];
  }
}

unsigned ec_bitree_get_cumul(const unsigned *_this,int _sym){
  unsigned ret;
  ret=0;
  while(_sym>0){
    ret+=_this[_sym-1];
    _sym&=_sym-1;
  }
  return ret;
}

unsigned ec_bitree_get_freq(const unsigned *_this,int _sym){
  unsigned ret;
  int      p;
  ret=_this[_sym];
  p=_sym&_sym+1;
  while(_sym!=p){
    ret-=_this[_sym-1];
    _sym&=_sym-1;
  }
  return ret;
}

#if 0
/*Fenwick's approach to re-scaling the counts.
  This tests to be twice as slow or more than the one below, even with inline
   functions enabled, and without loop vectorization (which would make Moffat's
   approach even faster).*/
void ec_bitree_halve(unsigned *_this,int _sz,int _split){
  int i;
  for(i=_sz;i-->0;){
    ec_bitree_update(_this,_sz,i,-(int)(ec_bitree_get_freq(_this,i)>>1));
  }
}
#else
/*Fenwick mentions that this approach is also possible, and this is what
   Moffat uses.
  Simply convert the tree into a simple array of counts, perform the halving,
   and then convert it back.*/
void ec_bitree_halve(unsigned *_this,int _sz,int _split){
  int i;
  ec_bitree_to_counts(_this,_sz,_split);
  /*LOOP VECTORIZES.*/
  for(i=0;i<_sz;i++)_this[i]-=_this[i]>>1;
  ec_bitree_from_counts(_this,_sz);
}
#endif

#if 0
#include <stdio.h>
/*Simple regression test code.
  Compile with bitrenc.c and bitrdec.c as well.*/

static void ec_bitree_print(unsigned *_this,int _sz){
  int i;
  for(i=0;i<_sz;i++)printf("%3i%c",_this[i],i+1<_sz?' ':'\n');
}

int main(void){
  unsigned t[16]={0,8,4,9,2,10,5,11,1,12,6,13,3,14,7,15};
  int      fl;
  int      s;
  int      i;
  ec_bitree_print(t,16);
  ec_bitree_from_counts(t,16);
  ec_bitree_print(t,16);
  for(i=0;i<=16;i++)printf("%3i%c",ec_bitree_get_cumul(t,i),i<16?' ':'\n');
  for(i=0;i<t[15];i++){
    s=ec_bitree_find_and_update(t,16,16,i,&fl,0);
    printf("%3i: %i %3i\n",i,s,fl);
  }
  for(i=0;i<16;i++){
    s=ec_bitree_find_and_update(t,16,ec_bitree_get_cumul(t,i),&fl,100);
    ec_bitree_to_counts(t,16,16);
    ec_bitree_print(t,16);
    ec_bitree_from_counts(t,16);
    ec_bitree_update(t,16,s,-100);
  }
  return 0;
}
#endif
