
#include <valgrind/memcheck.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MIN(a,b) ((a)<(b)?(a):(b))
#define COEF 0.0f
#define MAX_ENTRIES 16384

#define MULTI 4
#define MULTI_MASK (MULTI-1)

void compute_weights(const float *x, float *w, int ndim)
{
  int i;
  w[0] = MIN(x[0], x[1]-x[0]);
  for (i=1;i<ndim-1;i++)
    w[i] = MIN(x[i]-x[i-1], x[i+1]-x[i]);
  w[ndim-1] = MIN(x[ndim-1]-x[ndim-2], M_PI-x[ndim-1]);
  
  for (i=0;i<ndim;i++)
    w[i] = 1./(.01+w[i]);
  w[0]*=3;
  w[1]*=2;
  /*
  for (i=0;i<ndim;i++)
    w[i] = 1;*/
}

int find_nearest(const float *codebook, int nb_entries, const float *x, int ndim, float *dist)
{
  int i, j;
  float min_dist = 1e15;
  int nearest = 0;
  
  for (i=0;i<nb_entries;i++)
  {
    float dist=0;
    for (j=0;j<ndim;j++)
      dist += (x[j]-codebook[i*ndim+j])*(x[j]-codebook[i*ndim+j]);
    if (dist<min_dist)
    {
      min_dist = dist;
      nearest = i;
    }
  }
  if (dist)
    *dist = min_dist;
  return nearest;
}

int find_nearest_multi(const float *codebook, int nb_entries, const float *x, int ndim, float *dist, int sign)
{
  int i, j;
  float min_dist = 1e15;
  int nearest = 0;

  for (i=0;i<nb_entries;i++)
  {
    int offset;
    float dist=0;
    offset = (i&MULTI_MASK)*ndim;
    for (j=0;j<ndim;j++)
      dist += (x[offset+j]-codebook[i*ndim+j])*(x[offset+j]-codebook[i*ndim+j]);
    if (dist<min_dist)
    {
      min_dist = dist;
      nearest = i;
    }
  }
  if (sign) {
    for (i=0;i<nb_entries;i++)
    {
      int offset;
      float dist=0;
      offset = (i&MULTI_MASK)*ndim;
      for (j=0;j<ndim;j++)
        dist += (x[offset+j]+codebook[i*ndim+j])*(x[offset+j]+codebook[i*ndim+j]);
      if (dist<min_dist)
      {
        min_dist = dist;
        nearest = i+nb_entries;
      }
    }
  }
  if (dist)
    *dist = min_dist;
  return nearest;
}


int find_nearest_weighted(const float *codebook, int nb_entries, float *x, const float *w, int ndim)
{
  int i, j;
  float min_dist = 1e15;
  int nearest = 0;
  
  for (i=0;i<nb_entries;i++)
  {
    float dist=0;
    for (j=0;j<ndim;j++)
      dist += w[j]*(x[j]-codebook[i*ndim+j])*(x[j]-codebook[i*ndim+j]);
    if (dist<min_dist)
    {
      min_dist = dist;
      nearest = i;
    }
  }
  return nearest;
}

int quantize_lsp(const float *x, const float *codebook1, const float *codebook2, 
		 const float *codebook3, int nb_entries, float *xq, int ndim)
{
  int i, n1, n2, n3;
  float err[ndim], err2[ndim], err3[ndim];
  float w[ndim], w2[ndim], w3[ndim];
  
  w[0] = MIN(x[0], x[1]-x[0]);
  for (i=1;i<ndim-1;i++)
    w[i] = MIN(x[i]-x[i-1], x[i+1]-x[i]);
  w[ndim-1] = MIN(x[ndim-1]-x[ndim-2], M_PI-x[ndim-1]);
  
  /*
  for (i=0;i<ndim;i++)
    w[i] = 1./(.003+w[i]);
  w[0]*=3;
  w[1]*=2;*/
  compute_weights(x, w, ndim);
  
  for (i=0;i<ndim;i++)
    err[i] = x[i]-COEF*xq[i];
  n1 = find_nearest(codebook1, nb_entries, err, ndim, NULL);
  
  for (i=0;i<ndim;i++)
  {
    xq[i] = COEF*xq[i] + codebook1[ndim*n1+i];
    err[i] -= codebook1[ndim*n1+i];
  }
  for (i=0;i<ndim/2;i++)
  {
    err2[i] = err[2*i];  
    err3[i] = err[2*i+1];
    w2[i] = w[2*i];  
    w3[i] = w[2*i+1];
  }
  n2 = find_nearest_weighted(codebook2, nb_entries, err2, w2, ndim/2);
  n3 = find_nearest_weighted(codebook3, nb_entries, err3, w3, ndim/2);
  
  for (i=0;i<ndim/2;i++)
  {
    xq[2*i] += codebook2[ndim*n2/2+i];
    xq[2*i+1] += codebook3[ndim*n3/2+i];
  }
  return 0;
}

void split(float *codebook, int nb_entries, int ndim)
{
  int i,j;
  for (i=0;i<nb_entries;i++)
  {
    for (j=0;j<ndim;j++)
    {
      float delta = .01*(rand()/(float)RAND_MAX-.5);
      codebook[i*ndim+j] += delta;
      codebook[(i+nb_entries)*ndim+j] = codebook[i*ndim+j] - delta;
    }
  }
}


void split1(float *codebook, int nb_entries, const float *data, int nb_vectors, int ndim)
{
  int i,j;
  int nearest[nb_vectors];
  float dist[nb_entries];
  int count[nb_entries];
  int worst;
  for (i=0;i<nb_entries;i++)
    dist[i] = 0;
  for (i=0;i<nb_entries;i++)
    count[i]=0;
  for (i=0;i<nb_vectors;i++)
  {
    float d;
    nearest[i] = find_nearest(codebook, nb_entries, data+i*ndim, ndim, &d);
    dist[nearest[i]] += d;
    count[nearest[i]]++;
  }

  worst=0;
  for (i=1;i<nb_entries;i++)
  {
    if (dist[i] > dist[worst])
      worst=i;
  }
  
  for (j=0;j<ndim;j++)
  {
    float delta = .001*(rand()/(float)RAND_MAX-.5);
    codebook[worst*ndim+j] += delta;
    codebook[nb_entries*ndim+j] = codebook[worst*ndim+j] - delta;
  }
}



void update(float *data, int nb_vectors, float *codebook, int nb_entries, int ndim)
{
  int i,j;
  int count[nb_entries];
  int nearest[nb_vectors];
  double err=0;

  for (i=0;i<nb_entries;i++)
    count[i] = 0;
  
  for (i=0;i<nb_vectors;i++)
  {
    float dist;
    nearest[i] = find_nearest(codebook, nb_entries, data+i*ndim, ndim, &dist);
    err += dist;
  }
  printf("RMS error = %f\n", sqrt(err/nb_vectors/ndim));
  for (i=0;i<nb_entries*ndim;i++)
    codebook[i] = 0;
  
  for (i=0;i<nb_vectors;i++)
  {
    int n = nearest[i];
    count[n]++;
    for (j=0;j<ndim;j++)
      codebook[n*ndim+j] += data[i*ndim+j];
  }

  float w2=0;
  int min_count = 1000000000;
  int small=0;
  for (i=0;i<nb_entries;i++)
  { 
    for (j=0;j<ndim;j++)
      codebook[i*ndim+j] *= (1./count[i]);
    w2 += (count[i]/(float)nb_vectors)*(count[i]/(float)nb_vectors);
    if (count[i] < min_count) min_count = count[i];
    small += (count[i] < 50);
  }
  fprintf(stderr, "%f / %d, min = %d, small=%d\n", 1./w2, nb_entries, min_count, small);
}

void update_multi(float *data, int nb_vectors, float *codebook, int nb_entries, int ndim, int sign)
{
  int i,j;
  int count[nb_entries];
  int idcount[8]={0};
  int nearest[nb_vectors];
  double err=0;

  for (i=0;i<nb_entries;i++)
    count[i] = 0;

  for (i=0;i<nb_vectors;i++)
  {
    float dist;
    nearest[i] = find_nearest_multi(codebook, nb_entries, data+MULTI*i*ndim, ndim, &dist, sign);
    err += dist;
  }
  printf("RMS error = %f\n", sqrt(err/nb_vectors/ndim));
  for (i=0;i<nb_entries*ndim;i++)
    codebook[i] = 0;

  for (i=0;i<nb_vectors;i++)
  {
    int n = nearest[i] % nb_entries;
    float sign = nearest[i] < nb_entries ? 1 : -1;
    count[n]++;
    idcount[(n&MULTI_MASK) + 4*(sign!=1)]++;
    for (j=0;j<ndim;j++)
      codebook[n*ndim+j] += sign*data[(MULTI*i + (n&MULTI_MASK))*ndim+j];
  }

  float w2=0;
  int min_count = 1000000000;
  int small=0;
  for (i=0;i<nb_entries;i++)
  {
    for (j=0;j<ndim;j++)
      codebook[i*ndim+j] *= (1./count[i]);
    w2 += (count[i]/(float)nb_vectors)*(count[i]/(float)nb_vectors);
    if (count[i] < min_count) min_count = count[i];
    small += (count[i] < 50);
  }
  fprintf(stderr, "%d %d %d %d %d %d %d %d ", idcount[0], idcount[1], idcount[2], idcount[3], idcount[4], idcount[5], idcount[6], idcount[7]);
  fprintf(stderr, "| %f / %d, min = %d, small=%d\n", 1./w2, nb_entries, min_count, small);
}


void update_weighted(float *data, float *weight, int nb_vectors, float *codebook, int nb_entries, int ndim)
{
  int i,j;
  float count[MAX_ENTRIES][ndim];
  int nearest[nb_vectors];
  
  for (i=0;i<nb_entries;i++)
    for (j=0;j<ndim;j++)
      count[i][j] = 0;
  
  for (i=0;i<nb_vectors;i++)
  {
    nearest[i] = find_nearest_weighted(codebook, nb_entries, data+i*ndim, weight+i*ndim, ndim);
  }
  for (i=0;i<nb_entries*ndim;i++)
    codebook[i] = 0;
  
  for (i=0;i<nb_vectors;i++)
  {
    int n = nearest[i];
    for (j=0;j<ndim;j++)
    {
      float w = sqrt(weight[i*ndim+j]);
      count[n][j]+=w;
      codebook[n*ndim+j] += w*data[i*ndim+j];
    }
  }

  //float w2=0;
  for (i=0;i<nb_entries;i++)
  { 
    for (j=0;j<ndim;j++)
      codebook[i*ndim+j] *= (1./count[i][j]);
    //w2 += (count[i]/(float)nb_vectors)*(count[i]/(float)nb_vectors);
  }
  //fprintf(stderr, "%f / %d\n", 1./w2, nb_entries);
}

void vq_train(float *data, int nb_vectors, float *codebook, int nb_entries, int ndim)
{
  int i, j, e;
  e = 1;
  for (j=0;j<ndim;j++)
    codebook[j] = 0;
  for (i=0;i<nb_vectors;i++)
    for (j=0;j<ndim;j++)
      codebook[j] += data[i*ndim+j];
  for (j=0;j<ndim;j++)
    codebook[j] *= (1./nb_vectors);
  
  
  while (e< nb_entries)
  {
#if 1
    split(codebook, e, ndim);
    e<<=1;
#else
    split1(codebook, e, data, nb_vectors, ndim);
    e++;
#endif
    fprintf(stderr, "%d\n", e);
    for (j=0;j<4;j++)
      update(data, nb_vectors, codebook, e, ndim);
  }
  for (j=0;j<20;j++)
    update(data, nb_vectors, codebook, e, ndim);
}

void vq_train_multi(float *data, int nb_vectors, float *codebook, int nb_entries, int ndim, int sign)
{
  int i, j, e;
#if 1
  for (e=0;e<MULTI;e++) {
    for (j=0;j<ndim;j++)
      codebook[e*ndim+j] = 0;
    for (i=0;i<nb_vectors;i++)
      for (j=0;j<ndim;j++)
        codebook[e*ndim+j] += data[(MULTI*i+e)*ndim+j];
    for (j=0;j<ndim;j++) {
      float delta = .01*(rand()/(float)RAND_MAX-.5);
      codebook[e*ndim+j] *= (1./nb_vectors);
      codebook[e*ndim+j] += delta;
    }
  }
#else
  for (i=0;i<MULTI*ndim;i++) codebook[i] = .01*(rand()/(float)RAND_MAX-.5);
#endif
  e = MULTI;
  for (j=0;j<10;j++)
    update_multi(data, nb_vectors, codebook, e, ndim, sign);

  while (e < nb_entries)
  {
    split(codebook, e, ndim);
    e<<=1;
    fprintf(stderr, "%d\n", e);
    for (j=0;j<4;j++)
      update_multi(data, nb_vectors, codebook, e, ndim, sign);
  }
  for (j=0;j<20;j++)
    update_multi(data, nb_vectors, codebook, e, ndim, sign);
}


void vq_train_weighted(float *data, float *weight, int nb_vectors, float *codebook, int nb_entries, int ndim)
{
  int i, j, e;
  e = 1;
  for (j=0;j<ndim;j++)
    codebook[j] = 0;
  for (i=0;i<nb_vectors;i++)
    for (j=0;j<ndim;j++)
      codebook[j] += data[i*ndim+j];
  for (j=0;j<ndim;j++)
    codebook[j] *= (1./nb_vectors);
  
  
  while (e< nb_entries)
  {
#if 0
    split(codebook, e, ndim);
    e<<=1;
#else
    split1(codebook, e, data, nb_vectors, ndim);
    e++;
#endif
    fprintf(stderr, "%d\n", e);
    for (j=0;j<ndim;j++)
      update_weighted(data, weight, nb_vectors, codebook, e, ndim);
  }
}


int main(int argc, char **argv)
{
  int i,j;
  int nb_vectors, nb_entries, nb_entries1, nb_entries2a, nb_entries2b, ndim, ndim0, total_dim;
  float *data, *pred, *multi_data, *multi_data2, *qdata;
  float *codebook, *codebook2, *codebook3, *codebook_diff2, *codebook_diff4;
  float *delta;
  double err;
  FILE *fout;
  
  ndim = atoi(argv[1]);
  ndim0 = ndim-1;
  total_dim = atoi(argv[2]);
  nb_vectors = atoi(argv[3]);
  nb_entries = 1<<atoi(argv[4]);
  nb_entries1 = 1024;
  nb_entries2a = 4096;
  nb_entries2b = 64;
  
  data = malloc((nb_vectors*ndim+total_dim)*sizeof(*data));
  qdata = malloc((nb_vectors*ndim+total_dim)*sizeof(*qdata));
  pred = malloc(nb_vectors*ndim0*sizeof(*pred));
  multi_data = malloc(MULTI*nb_vectors*ndim*sizeof(*multi_data));
  multi_data2 = malloc(MULTI*nb_vectors*ndim*sizeof(*multi_data));
  codebook = malloc(nb_entries*ndim0*sizeof(*codebook));
  codebook2 = malloc(nb_entries1*ndim0*sizeof(*codebook2));
  codebook3 = malloc(nb_entries1*ndim0*sizeof(*codebook3));
  codebook_diff4 = malloc(nb_entries2a*ndim*sizeof(*codebook_diff4));
  codebook_diff2 = malloc(nb_entries2b*ndim*sizeof(*codebook_diff2));
  
  for (i=0;i<nb_vectors;i++)
  {
    fread(&data[i*ndim], sizeof(float), total_dim, stdin);
    if (feof(stdin))
      break;
  }
  nb_vectors = i;
  VALGRIND_CHECK_MEM_IS_DEFINED(data, nb_entries*ndim);

  for (i=0;i<4;i++)
  {
    for (j=0;j<ndim0;j++)
      pred[i*ndim0+j] = 0;
  }
  for (i=4;i<nb_vectors;i++)
  {
    for (j=0;j<ndim0;j++)
      pred[i*ndim0+j] = data[i*ndim+j+1] - COEF*data[(i-4)*ndim+j+1];
  }
#if 1
  VALGRIND_CHECK_MEM_IS_DEFINED(pred, nb_entries*ndim0);
  vq_train(pred, nb_vectors, codebook, nb_entries, ndim0);
  
  delta = malloc(nb_vectors*ndim0*sizeof(*data));
  err = 0;
  for (i=0;i<nb_vectors;i++)
  {
    int nearest = find_nearest(codebook, nb_entries, &pred[i*ndim0], ndim0, NULL);
    qdata[i*ndim] = data[i*ndim];
    for (j=0;j<ndim0;j++)
    {
      qdata[i*ndim+j+1] = codebook[nearest*ndim0+j];
      delta[i*ndim0+j] = pred[i*ndim0+j] - codebook[nearest*ndim0+j];
      err += delta[i*ndim0+j]*delta[i*ndim0+j];
    }
    //printf("\n");
  }
  fprintf(stderr, "Cepstrum RMS error: %f\n", sqrt(err/nb_vectors/ndim));

  vq_train(delta, nb_vectors, codebook2, nb_entries1, ndim0);
  
  err=0;
  for (i=0;i<nb_vectors;i++)
  {
    int n1;
    n1 = find_nearest(codebook2, nb_entries1, &delta[i*ndim0], ndim0, NULL);
    for (j=0;j<ndim0;j++)
    {
      qdata[i*ndim+j+1] += codebook2[n1*ndim0+j];
      //delta[i*ndim0+j] = delta[i*ndim0+j] - codebook2[n1*ndim0+j];
      delta[i*ndim0+j] = data[i*ndim+j+1] - qdata[i*ndim+j+1];
      err += delta[i*ndim0+j]*delta[i*ndim0+j];
    }
  }
  fprintf(stderr, "Cepstrum RMS error after stage 2: %f)\n", sqrt(err/nb_vectors/ndim));

  vq_train(delta, nb_vectors, codebook3, nb_entries1, ndim0);
  err=0;
  for (i=0;i<nb_vectors;i++)
  {
    int n1;
    n1 = find_nearest(codebook3, nb_entries1, &delta[i*ndim0], ndim0, NULL);
    for (j=0;j<ndim0;j++)
    {
      qdata[i*ndim+j+1] += codebook3[n1*ndim0+j];
      //delta[i*ndim0+j] = delta[i*ndim0+j] - codebook2[n1*ndim0+j];
      delta[i*ndim0+j] = data[i*ndim+j+1] - qdata[i*ndim+j+1];
      err += delta[i*ndim0+j]*delta[i*ndim0+j];
    }
  }
  fprintf(stderr, "Cepstrum RMS error after stage 3: %f)\n", sqrt(err/nb_vectors/ndim));
#else
  qdata = data;
#endif
  for (i=0;i<nb_vectors-4;i++)
  {
    for (j=0;j<ndim;j++)
      multi_data[MULTI*i*ndim+j]     = data[(i+1)*ndim+j] - .5*(qdata[i*ndim+j]+qdata[(i+2)*ndim+j]);
    for (j=0;j<ndim;j++)
      multi_data[(MULTI*i+1)*ndim+j] = data[(i+1)*ndim+j] - .5*(qdata[i*ndim+j]+qdata[(i+2)*ndim+j]);
    for (j=0;j<ndim;j++)
      multi_data[(MULTI*i+2)*ndim+j] = data[(i+1)*ndim+j] - qdata[i*ndim+j];
    for (j=0;j<ndim;j++)
      multi_data[(MULTI*i+3)*ndim+j] = data[(i+1)*ndim+j] - qdata[(i+2)*ndim+j];
    //for (j=0;j<4*ndim;j++) printf("%f ", multi_data[MULTI*i*ndim + j]);
    //printf("\n");
  }

  for (i=0;i<nb_vectors-4;i++)
  {
    for (j=0;j<ndim;j++)
      multi_data2[MULTI*i*ndim+j]     = data[(i+2)*ndim+j] - .5*(qdata[i*ndim+j]+qdata[(i+4)*ndim+j]);
    for (j=0;j<ndim;j++)
      multi_data2[(MULTI*i+1)*ndim+j] = data[(i+2)*ndim+j] - .5*(qdata[i*ndim+j]+qdata[(i+4)*ndim+j]);
    for (j=0;j<ndim;j++)
      multi_data2[(MULTI*i+2)*ndim+j] = data[(i+2)*ndim+j] - qdata[i*ndim+j];
    for (j=0;j<ndim;j++)
      multi_data2[(MULTI*i+3)*ndim+j] = data[(i+2)*ndim+j] - qdata[(i+4)*ndim+j];
  }

  vq_train_multi(multi_data2, nb_vectors-4, codebook_diff4, nb_entries2a, ndim, 1);

  printf("done\n");
  vq_train_multi(multi_data, nb_vectors-4, codebook_diff2, nb_entries2b, ndim, 0);


  fout = fopen("ceps_codebooks.c", "w");
  fprintf(fout, "/* This file is automatically generated */\n\n");
  fprintf(fout, "float ceps_codebook1[%d*%d] = {\n",nb_entries, ndim0);
  
  for (i=0;i<nb_entries;i++)
  {
    for (j=0;j<ndim0;j++)
      fprintf(fout, "%f, ", codebook[i*ndim0+j]);
    fprintf(fout, "\n");
  }
  fprintf(fout, "};\n\n");

  fprintf(fout, "float ceps_codebook2[%d*%d] = {\n",nb_entries1, ndim0);
  for (i=0;i<nb_entries1;i++)
  {
    for (j=0;j<ndim0;j++)
      fprintf(fout, "%f, ", codebook2[i*ndim0+j]);
    fprintf(fout, "\n");
  }
  fprintf(fout, "};\n\n");

  fprintf(fout, "float ceps_codebook3[%d*%d] = {\n",nb_entries1, ndim0);
  for (i=0;i<nb_entries1;i++)
  {
    for (j=0;j<ndim0;j++)
      fprintf(fout, "%f, ", codebook3[i*ndim0+j]);
    fprintf(fout, "\n");
  }
  fprintf(fout, "};\n\n");

  fprintf(fout, "float ceps_codebook_diff4[%d*%d] = {\n",nb_entries2a, ndim);
  for (i=0;i<nb_entries2a;i++)
  {
    for (j=0;j<ndim;j++)
      fprintf(fout, "%f, ", codebook_diff4[i*ndim+j]);
    fprintf(fout, "\n");
  }
  fprintf(fout, "};\n\n");

  fprintf(fout, "float ceps_codebook_diff2[%d*%d] = {\n",nb_entries2b, ndim);
  for (i=0;i<nb_entries2b;i++)
  {
    for (j=0;j<ndim;j++)
      fprintf(fout, "%f, ", codebook_diff2[i*ndim+j]);
    fprintf(fout, "\n");
  }
  fprintf(fout, "};\n\n");
  
  fclose(fout);
  return 0;
}
