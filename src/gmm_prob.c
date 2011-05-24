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
    int i;
    float *data;
    int nbGaussians;
    float *mean, *var_1, *w;
    int N, nbVectors;
    FILE *GMM;

    N = atoi(argv[1]);
    nbVectors = atoi(argv[2]);
    nbGaussians = atoi(argv[3]);

    GMM = fopen(argv[4], "r");

    data = malloc(sizeof(*data)*N*nbVectors);
    for (i=0;i<N*nbVectors;i++)
        ret = scanf("%f ", data+i);

    mean = malloc(sizeof(*mean)*N*nbGaussians);
    var_1 = malloc(sizeof(*var_1)*N*nbGaussians);
    w = malloc(sizeof(*w)*nbGaussians);

    for (i=0;i<N*nbGaussians;i++)
        ret = fscanf (GMM, "%f ", &mean[i]);
    for (i=0;i<N*nbGaussians;i++)
        ret = fscanf (GMM, "%f ", &var_1[i]);
    for (i=0;i<nbGaussians;i++)
        ret = fscanf (GMM, "%f ", &w[i]);

    gmm_prob(mean, var_1, w, nbGaussians, N, data, nbVectors);

    return 0;
}
