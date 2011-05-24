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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPSILON 1e-300
#define MIN_VARIANCE 1e-5

static void gmm_init(float *mean, float *var_1, float *weight,
        int nbGaussians, int N, const float *data, int nbVectors)
{
    double mean_acc[N],var_acc[N];
    int i, j, k;

    for (k=0;k<N;k++)
        mean_acc[k] = var_acc[k] = 0;

    for (i=0;i<nbVectors;i++)
    {
        const float *x = data+i*N;
        for (j=0;j<N;j++)
        {
            mean_acc[j] += x[j];
            var_acc[j] += x[j]*x[j];
        }
    }
    for (j=0;j<N;j++)
    {
        double w_tmp;
        double w_1 = 1./nbVectors;
        mean[j] = w_1*mean_acc[j];
        var_1[j] = w_1*var_acc[j] - mean[j]*mean[j];
        if (var_1[j]<MIN_VARIANCE)
            var_1[j] = MIN_VARIANCE;
        var_1[j] = 1/(var_1[j]+EPSILON);
        /* Folding scaling in the weight */
        w_tmp = sqrt(var_1[j]);
        if (w_tmp > 1e15)
            w_tmp = 1e15;
        weight[0] = w_tmp;
        printf("%f ", var_1[j]);
    }
    printf("\n");

    /*for (k=nbGaussians-1;k>=0;k--)
    {
        int k2 = rand()%nbVectors;
        weight[k] = weight[0]/nbGaussians;
        for (j=0;j<N;j++)
        {
            mean[k*N+j] = data[k2*N+j];
            var_1[k*N+j] = var_1[j];
        }
    }*/
}

static void gmm_split(float *mean, float *var_1, float *weight,
        int nbGaussians, int N)
{
    int j, k;
    for (k=0;k<nbGaussians;k++)
    {
        weight[k] *= .5;
        weight[k+nbGaussians] = weight[k];
        for (j=0;j<N;j++)
        {
            float d = rand()&1 ? 1 : -1;
            d *= .5/sqrt(var_1[j]);
            mean[j+nbGaussians] = mean[j]+d;
            mean[j] = mean[j]-d;
            var_1[j+nbGaussians] = var_1[j];
        }
    }
}

static void run_em_iteration(float *mean, float *var_1, float *weight,
        int nbGaussians, int N, const float *data, int nbVectors)
{
    double mean_acc[N*nbGaussians],var_acc[N*nbGaussians],weight_acc[nbGaussians];
    double Neff=0;
    double KL_cost = 0;
    int i, j, k;

    for (k=0;k<nbGaussians*N;k++)
        mean_acc[k] = var_acc[k] = 0;
    for (k=0;k<nbGaussians;k++)
        weight_acc[k] = 0;

    for (i=0;i<nbVectors;i++)
    {
        const float *x = data+i*N;
        double probs[nbGaussians];
        double prob_sum=0;
        /* Expectation */
        for (k=0;k<nbGaussians;k++)
        {
            float *u, *v_1;
            float dist=0;

            u = mean+k*N;
            v_1 = var_1+k*N;
            for (j=0;j<N;j++)
                dist += v_1[j]*(u[j]-x[j])*(u[j]-x[j]);
            //printf("%f ", dist);
            probs[k] = weight[k]*exp(-.5*dist);
            if (!isfinite(probs[k]))
            {
                printf("BOOM! %f %f\n", weight[k], dist);
                exit(1);
            }
            prob_sum += probs[k];
        }
        KL_cost -= log2(prob_sum);

        //printf("%f ", prob_sum);
        prob_sum = 1./(prob_sum+EPSILON);
        for (k=0;k<nbGaussians;k++)
        {
            probs[k] *= prob_sum;
            //printf("%f ", probs[k]);
        }
        //printf("\n");
        /* Maximization */
        for (k=0;k<nbGaussians;k++)
        {
            weight_acc[k] += probs[k];
            for (j=0;j<N;j++)
                mean_acc[k*N+j] += probs[k]*x[j];
            for (j=0;j<N;j++)
                var_acc[k*N+j] += probs[k]*x[j]*x[j];
        }
    }

    for (k=0;k<nbGaussians;k++)
    {
        double w_1;
        w_1 = 1/(weight_acc[k]+EPSILON);
        weight[k] = weight_acc[k]/nbVectors;
        Neff += weight[k]*weight[k];
        //printf ("%f ", weight[k]);
        for (j=0;j<N;j++)
            mean[k*N+j] = w_1*mean_acc[k*N+j];
        for (j=0;j<N;j++)
        {
            double w_tmp;
            var_1[k*N+j] = w_1*var_acc[k*N+j] - mean[k*N+j]*mean[k*N+j];
            if (var_1[k*N+j]<MIN_VARIANCE)
                var_1[k*N+j] = MIN_VARIANCE;
            var_1[k*N+j] = 1/(var_1[k*N+j]+EPSILON);
            /* Folding scaling in the weight */
            w_tmp = weight[k] * sqrt(var_1[k*N+j]);
            if (w_tmp > 1e15)
                w_tmp = 1e15;
            weight[k] = w_tmp;
        }
    }
    Neff = 1./Neff;
    KL_cost /= nbVectors;
    fprintf(stderr, "Neff = %f, KL cost = %f, nbGaussians = %d\n", Neff, KL_cost, nbGaussians);
}

static void gmm_prob(float *mean, float *var_1, float *weight,
        int nbGaussians, int N, const float *data, int nbVectors)
{
    double mean_acc[N*nbGaussians],var_acc[N*nbGaussians],weight_acc[nbGaussians];
    int i, j, k;

    for (k=0;k<nbGaussians*N;k++)
        mean_acc[k] = var_acc[k] = 0;
    for (k=0;k<nbGaussians;k++)
        weight_acc[k] = 0;

    for (i=0;i<nbVectors;i++)
    {
        const float *x = data+i*N;
        double probs[nbGaussians];
        double prob_sum=0;
        /* Expectation */
        for (k=0;k<nbGaussians;k++)
        {
            float *u, *v_1;
            float dist=0;

            u = mean+k*N;
            v_1 = var_1+k*N;
            for (j=0;j<N;j++)
                dist += v_1[j]*(u[j]-x[j])*(u[j]-x[j]);
            //printf("%f ", dist);
            probs[k] = weight[k]*exp(-.5*dist);
            if (!isfinite(probs[k]))
            {
                printf("%f %f\n", weight[k], dist);
                exit(1);
            }
            prob_sum += probs[k];
        }
        printf("%g\n", prob_sum);
    }
}


int main(int argc, char **argv)
{
    int ret;
    int i, j, k;
    float *data;
    int nbGaussians;
    float *mean, *var_1, *w;
    int N, nbVectors;
    unsigned t;

    t = time(NULL);
    srand(t);
    N = atoi(argv[1]);
    nbVectors = atoi(argv[2]);
    nbGaussians = atoi(argv[3]);

    data = malloc(sizeof(*data)*N*nbVectors);
    for (i=0;i<N*nbVectors;i++)
        ret = scanf("%f ", data+i);

    mean = malloc(sizeof(*mean)*N*nbGaussians);
    var_1 = malloc(sizeof(*var_1)*N*nbGaussians);
    w = malloc(sizeof(*w)*nbGaussians);

    for(i=0;i<nbGaussians;i++)
        w[i] = .5;

    for(k=0;k<nbGaussians;k++)
    {
        i = rand()%nbVectors;
        for (j=0;j<N;j++)
            mean[k*N+j] = data[i*N+j];
    }
    for(i=0;i<nbGaussians*N;i++)
        var_1[i] = .1;

#if 1
    //gmm_init(mean, var_1, w, nbGaussians, N, data, nbVectors);

    for (i=0;i<2000;i++)
        run_em_iteration(mean, var_1, w, nbGaussians, N, data, nbVectors);
#else
    //gmm_init(mean, var_1, w, nbGaussians, N, data, nbVectors);
    run_em_iteration(mean, var_1, w, 1, N, data, nbVectors);

    int g=1;
    for (i=0;i<2000;i++)
    {
        if (i%20==0 && 2*g<=nbGaussians)
        {
            gmm_split(mean, var_1, w, g, N);
            g *= 2;
        }
        run_em_iteration(mean, var_1, w, g, N, data, nbVectors);
    }
#endif
    //gmm_prob(mean, var_1, w, nbGaussians, N, data, nbVectors);
    //exit(1);

    for (i=0;i<N*nbGaussians;i++)
        printf ("%f ", mean[i]);
    printf("\n");
    for (i=0;i<N*nbGaussians;i++)
        printf ("%f ", var_1[i]);
    printf("\n");
    for (i=0;i<nbGaussians;i++)
        printf ("%f ", w[i]);
    printf("\n");

    return 0;
}
