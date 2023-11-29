#include <math.h>

#include "osce.h"
#include "os_support.h"
#include "nndsp.h"

#include <stdio.h>

#define CLIP(a, min, max) (((a) < (min) ? (min) : (a)) > (max) ? (max) : (a))
#define MAX(a, b) ((a) < (b) ? (b) : (a))

static void compute_lace_numbits_embedding(float *emb, float numbits, int dim, float min_val, float max_val, int logscale)
{
    float x;
    (void) dim;

    numbits = logscale ? log(numbits) : numbits;
    x = CLIP(numbits, min_val, max_val) - (max_val + min_val) / 2;

    emb[0] = sin(x * LACE_NUMBITS_SCALE_0 - 0.5f);
    emb[1] = sin(x * LACE_NUMBITS_SCALE_1 - 0.5f);
    emb[2] = sin(x * LACE_NUMBITS_SCALE_2 - 0.5f);
    emb[3] = sin(x * LACE_NUMBITS_SCALE_3 - 0.5f);
    emb[4] = sin(x * LACE_NUMBITS_SCALE_4 - 0.5f);
    emb[5] = sin(x * LACE_NUMBITS_SCALE_5 - 0.5f);
    emb[6] = sin(x * LACE_NUMBITS_SCALE_6 - 0.5f);
    emb[7] = sin(x * LACE_NUMBITS_SCALE_7 - 0.5f);
}


void init_lace(LACE *hLACE)
{
    int i;
    OPUS_CLEAR(hLACE, 1);

    init_lacelayers(&hLACE->layers, lacelayers_arrays);

    init_adacomb_state(&hLACE->state.cf1_state);
    init_adacomb_state(&hLACE->state.cf2_state);
    init_adaconv_state(&hLACE->state.af1_state);

    for (i = 0; i < LACE_OVERLAP_SIZE; i ++)
    {
        hLACE->window[i] = 0.5 + 0.5 * cos(M_PI * (i + 0.5) / LACE_OVERLAP_SIZE);
    }
}

void lace_feature_net(
    LACE *hLACE,
    float *output,
    const float *features,
    const float *numbits,
    const int *periods
);

void lace_feature_net(
    LACE *hLACE,
    float *output,
    const float *features,
    const float *numbits,
    const int *periods
)
{
    float input_buffer[4 * MAX(LACE_COND_DIM, LACE_HIDDEN_FEATURE_DIM)];
    float output_buffer[4 * MAX(LACE_COND_DIM, LACE_HIDDEN_FEATURE_DIM)];
    float numbits_embedded[2 * LACE_NUMBITS_EMBEDDING_DIM];
    int i_subframe;

    compute_lace_numbits_embedding(numbits_embedded, numbits[0], LACE_NUMBITS_EMBEDDING_DIM,
        log(LACE_NUMBITS_RANGE_LOW), log(LACE_NUMBITS_RANGE_HIGH), 1);
    compute_lace_numbits_embedding(numbits_embedded + LACE_NUMBITS_EMBEDDING_DIM, numbits[1], LACE_NUMBITS_EMBEDDING_DIM,
        log(LACE_NUMBITS_RANGE_LOW), log(LACE_NUMBITS_RANGE_HIGH), 1);

    /* scaling and dimensionality reduction */
    for (i_subframe = 0; i_subframe < 4; i_subframe ++)
    {
        OPUS_COPY(input_buffer, features + i_subframe * LACE_NUM_FEATURES, LACE_NUM_FEATURES);
        OPUS_COPY(input_buffer + LACE_NUM_FEATURES, hLACE->layers.embed.float_weights + periods[i_subframe] * LACE_PITCH_EMBEDDING_DIM, LACE_PITCH_EMBEDDING_DIM);
        OPUS_COPY(input_buffer + LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM, numbits_embedded, 2 * LACE_NUMBITS_EMBEDDING_DIM);

        compute_generic_conv1d(
            &hLACE->layers.lace_feature_net_conv1,
            output_buffer + i_subframe * LACE_HIDDEN_FEATURE_DIM,
            NULL,
            input_buffer,
            LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM + 2 * LACE_NUMBITS_EMBEDDING_DIM,
            ACTIVATION_TANH);
    }

    /* subframe accumulation */
    OPUS_COPY(input_buffer, output_buffer, 4 * LACE_HIDDEN_FEATURE_DIM);
    compute_generic_conv1d(
        &hLACE->layers.lace_feature_net_conv2,
        output_buffer,
        hLACE->state.feature_net_conv2_state,
        input_buffer,
        4 * LACE_HIDDEN_FEATURE_DIM,
        ACTIVATION_TANH
    );

    /* tconv upsampling */
    OPUS_COPY(input_buffer, output_buffer, 4 * LACE_COND_DIM);
    compute_generic_dense(
        &hLACE->layers.lace_feature_net_tconv,
        output_buffer,
        input_buffer,
        ACTIVATION_LINEAR
    );

    /* GRU */
    OPUS_COPY(input_buffer, output_buffer, 4 * LACE_COND_DIM);
    for (i_subframe = 0; i_subframe < 4; i_subframe++)
    {
        compute_generic_gru(
            &hLACE->layers.lace_feature_net_gru_input,
            &hLACE->layers.lace_feature_net_gru_recurrent,
            hLACE->state.feature_net_gru_state,
            input_buffer + i_subframe * LACE_COND_DIM
        );
        OPUS_COPY(output + i_subframe * LACE_COND_DIM, hLACE->state.feature_net_gru_state, LACE_COND_DIM);
    }
}


//#define DEBUG_LACE
#ifdef DEBUG_LACE
#define FINIT(fid, name, mode) do{if (fid == NULL) {fid = fopen(name, mode);}} while(0)
#endif

void lace_process_20ms_frame(
    LACE* hLACE,
    float *x_out,
    const float *x_in,
    const float *features,
    const float *numbits,
    const int *periods
)
{
    float feature_buffer[4 * LACE_COND_DIM];
    float output_buffer[4 * LACE_FRAME_SIZE];
    int i_subframe, i_sample;

#ifdef DEBUG_LACE
    static FILE *f_features=NULL, *f_encfeatures=NULL, *f_xin=NULL, *f_xpreemph=NULL, *f_postcf1=NULL;
    static FILE *f_postcf2=NULL, *f_postaf1=NULL, *f_xdeemph, *f_numbits, *f_periods;


    FINIT(f_features, "debug/c_features.f32", "wb");
    FINIT(f_encfeatures, "debug/c_encoded_features.f32", "wb");
    FINIT(f_xin, "debug/c_x_in.f32", "wb");
    FINIT(f_xpreemph, "debug/c_xpreemph.f32", "wb");
    FINIT(f_xdeemph, "debug/c_xdeemph.f32", "wb");
    FINIT(f_postcf1, "debug/c_post_cf1.f32", "wb");
    FINIT(f_postcf2, "debug/c_post_cf2.f32", "wb");
    FINIT(f_postaf1, "debug/c_post_af1.f32", "wb");
    FINIT(f_numbits, "debug/c_numbits.f32", "wb");
    FINIT(f_periods, "debug/c_periods.s32", "wb");

    fwrite(x_in, sizeof(*x_in), 4 * LACE_FRAME_SIZE, f_xin);
    fwrite(numbits, sizeof(*numbits), 2, f_numbits);
    fwrite(periods, sizeof(*periods), 4, f_periods);
#endif

    /* pre-emphasis */
    for (i_sample = 0; i_sample < 4 * LACE_FRAME_SIZE; i_sample ++)
    {
        output_buffer[i_sample] = x_in[i_sample] - LACE_PREEMPH * hLACE->state.preemph_mem;
        hLACE->state.preemph_mem = x_in[i_sample];
    }

    /* run feature encoder */
    lace_feature_net(hLACE, feature_buffer, features, numbits, periods);
#ifdef DEBUG_LACE
    fwrite(features, sizeof(*features), 4 * LACE_NUM_FEATURES, f_features);
    fwrite(feature_buffer, sizeof(*feature_buffer), 4 * LACE_COND_DIM, f_encfeatures);
    fwrite(output_buffer, sizeof(float), 4 * LACE_FRAME_SIZE, f_xpreemph);
#endif

    /* 1st comb filtering stage */
    for (i_subframe = 0; i_subframe < 4; i_subframe++)
    {
        adacomb_process_frame(
            &hLACE->state.cf1_state,
            output_buffer + i_subframe * LACE_FRAME_SIZE,
            output_buffer + i_subframe * LACE_FRAME_SIZE,
            feature_buffer + i_subframe * LACE_COND_DIM,
            &hLACE->layers.lace_cf1_kernel,
            &hLACE->layers.lace_cf1_gain,
            &hLACE->layers.lace_cf1_global_gain,
            periods[i_subframe],
            LACE_COND_DIM,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            LACE_CF1_KERNEL_SIZE,
            LACE_CF1_LEFT_PADDING,
            LACE_CF1_FILTER_GAIN_A,
            LACE_CF1_FILTER_GAIN_B,
            LACE_CF1_LOG_GAIN_LIMIT,
            hLACE->window);
    }

#ifdef DEBUG_LACE
    fwrite(output_buffer, sizeof(float), 4 * LACE_FRAME_SIZE, f_postcf1);
#endif

    /* 2nd comb filtering stage */
    for (i_subframe = 0; i_subframe < 4; i_subframe++)
    {
        adacomb_process_frame(
            &hLACE->state.cf2_state,
            output_buffer + i_subframe * LACE_FRAME_SIZE,
            output_buffer + i_subframe * LACE_FRAME_SIZE,
            feature_buffer + i_subframe * LACE_COND_DIM,
            &hLACE->layers.lace_cf2_kernel,
            &hLACE->layers.lace_cf2_gain,
            &hLACE->layers.lace_cf2_global_gain,
            periods[i_subframe],
            LACE_COND_DIM,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            LACE_CF2_KERNEL_SIZE,
            LACE_CF2_LEFT_PADDING,
            LACE_CF2_FILTER_GAIN_A,
            LACE_CF2_FILTER_GAIN_B,
            LACE_CF2_LOG_GAIN_LIMIT,
            hLACE->window);
    }
#ifdef DEBUG_LACE
    fwrite(output_buffer, sizeof(float), 4 * LACE_FRAME_SIZE, f_postcf2);
#endif

    /* final adaptive filtering stage */
    for (i_subframe = 0; i_subframe < 4; i_subframe++)
    {
        adaconv_process_frame(
            &hLACE->state.af1_state,
            output_buffer + i_subframe * LACE_FRAME_SIZE,
            output_buffer + i_subframe * LACE_FRAME_SIZE,
            feature_buffer + i_subframe * LACE_COND_DIM,
            &hLACE->layers.lace_af1_kernel,
            &hLACE->layers.lace_af1_gain,
            LACE_COND_DIM,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            LACE_AF1_IN_CHANNELS,
            LACE_AF1_OUT_CHANNELS,
            LACE_AF1_KERNEL_SIZE,
            LACE_AF1_LEFT_PADDING,
            LACE_AF1_FILTER_GAIN_A,
            LACE_AF1_FILTER_GAIN_B,
            LACE_AF1_SHAPE_GAIN,
            hLACE->window);
    }
#ifdef DEBUG_LACE
    fwrite(output_buffer, sizeof(float), 4 * LACE_FRAME_SIZE, f_postaf1);
#endif

    /* de-emphasis */
    for (i_sample = 0; i_sample < 4 * LACE_FRAME_SIZE; i_sample ++)
    {
        x_out[i_sample] = output_buffer[i_sample] + LACE_PREEMPH * hLACE->state.deemph_mem;
        hLACE->state.deemph_mem = x_out[i_sample];
    }
#ifdef DEBUG_LACE
    fwrite(x_out, sizeof(float), 4 * LACE_FRAME_SIZE, f_xdeemph);
#endif
}

#if 0

void lace_feature_net_compare(
    const char * prefix,
    int num_frames,
    LACE* hLACE
)
{
    char in_feature_file[256];
    char out_feature_file[256];
    char numbits_file[256];
    char periods_file[256];
    char message[512];
    int i_frame, i_feature;
    float mse;
    float in_features[4 * LACE_NUM_FEATURES];
    float out_features[4 * LACE_COND_DIM];
    float out_features2[4 * LACE_COND_DIM];
    float numbits[2];
    int periods[4];

    init_lace(hLACE);

    FILE *f_in_features, *f_out_features, *f_numbits, *f_periods;

    strcpy(in_feature_file, prefix);
    strcat(in_feature_file, "_in_features.f32");
    f_in_features = fopen(in_feature_file, "rb");
    if (f_in_features == NULL)
    {
        sprintf(message, "could not open file %s", in_feature_file);
        perror(message);
        exit(1);
    }

    strcpy(out_feature_file, prefix);
    strcat(out_feature_file, "_out_features.f32");
    f_out_features = fopen(out_feature_file, "rb");
    if (f_out_features == NULL)
    {
        sprintf(message, "could not open file %s", out_feature_file);
        perror(message);
        exit(1);
    }

    strcpy(periods_file, prefix);
    strcat(periods_file, "_periods.s32");
    f_periods = fopen(periods_file, "rb");
    if (f_periods == NULL)
    {
        sprintf(message, "could not open file %s", periods_file);
        perror(message);
        exit(1);
    }

    strcpy(numbits_file, prefix);
    strcat(numbits_file, "_numbits.f32");
    f_numbits = fopen(numbits_file, "rb");
    if (f_numbits == NULL)
    {
        sprintf(message, "could not open file %s", numbits_file);
        perror(message);
        exit(1);
    }

    for (i_frame = 0; i_frame < num_frames; i_frame ++)
    {
        if(fread(in_features, sizeof(float), 4 * LACE_NUM_FEATURES, f_in_features) != 4 * LACE_NUM_FEATURES)
        {
            fprintf(stderr, "could not read frame %d from in_features\n", i_frame);
            exit(1);
        }
        if(fread(out_features, sizeof(float), 4 * LACE_COND_DIM, f_out_features) != 4 * LACE_COND_DIM)
        {
            fprintf(stderr, "could not read frame %d from out_features\n", i_frame);
            exit(1);
        }
        if(fread(periods, sizeof(int), 4, f_periods) != 4)
        {
            fprintf(stderr, "could not read frame %d from periods\n", i_frame);
            exit(1);
        }
        if(fread(numbits, sizeof(float), 2, f_numbits) != 2)
        {
            fprintf(stderr, "could not read frame %d from numbits\n", i_frame);
            exit(1);
        }


        lace_feature_net(hLACE, out_features2, in_features, numbits, periods);

        float mse = 0;
        for (int i = 0; i < 4 * LACE_COND_DIM; i ++)
        {
            mse += pow(out_features[i] - out_features2[i], 2);
        }
        mse /= (4 * LACE_COND_DIM);
        printf("rmse: %f\n", sqrt(mse));

    }

    fclose(f_in_features);
    fclose(f_out_features);
    fclose(f_numbits);
    fclose(f_periods);
}

void lace_demo(
    char *prefix,
    char *output
)
{
    char feature_file[256];
    char numbits_file[256];
    char periods_file[256];
    char x_in_file[256];
    char message[512];
    int i_frame;
    float mse;
    float features[4 * LACE_NUM_FEATURES];
    float numbits[2];
    int periods[4];
    float x_in[4 * LACE_FRAME_SIZE];
    int16_t x_out[4 * LACE_FRAME_SIZE];
    float buffer[4 * LACE_FRAME_SIZE];
    LACE hLACE;
    int frame_counter = 0;
    FILE *f_features, *f_numbits, *f_periods, *f_x_in, *f_x_out;

    init_lace(&hLACE);

    strcpy(feature_file, prefix);
    strcat(feature_file, "_features.f32");
    f_features = fopen(feature_file, "rb");
    if (f_features == NULL)
    {
        sprintf(message, "could not open file %s", feature_file);
        perror(message);
        exit(1);
    }

    strcpy(x_in_file, prefix);
    strcat(x_in_file, "_x_in.f32");
    f_x_in = fopen(x_in_file, "rb");
    if (f_x_in == NULL)
    {
        sprintf(message, "could not open file %s", x_in_file);
        perror(message);
        exit(1);
    }

    f_x_out = fopen(output, "wb");
    if (f_x_out == NULL)
    {
        sprintf(message, "could not open file %s", output);
        perror(message);
        exit(1);
    }

    strcpy(periods_file, prefix);
    strcat(periods_file, "_periods.s32");
    f_periods = fopen(periods_file, "rb");
    if (f_periods == NULL)
    {
        sprintf(message, "could not open file %s", periods_file);
        perror(message);
        exit(1);
    }

    strcpy(numbits_file, prefix);
    strcat(numbits_file, "_numbits.f32");
    f_numbits = fopen(numbits_file, "rb");
    if (f_numbits == NULL)
    {
        sprintf(message, "could not open file %s", numbits_file);
        perror(message);
        exit(1);
    }

    printf("processing %s\n", prefix);

    while (fread(x_in, sizeof(float), 4 * LACE_FRAME_SIZE, f_x_in) == 4 * LACE_FRAME_SIZE)
    {
        printf("\rframe: %d", frame_counter++);
        if(fread(features, sizeof(float), 4 * LACE_NUM_FEATURES, f_features) != 4 * LACE_NUM_FEATURES)
        {
            fprintf(stderr, "could not read frame %d from features\n", i_frame);
            exit(1);
        }
        if(fread(periods, sizeof(int), 4, f_periods) != 4)
        {
            fprintf(stderr, "could not read frame %d from periods\n", i_frame);
            exit(1);
        }
        if(fread(numbits, sizeof(float), 2, f_numbits) != 2)
        {
            fprintf(stderr, "could not read frame %d from numbits\n", i_frame);
            exit(1);
        }

        lace_process_20ms_frame(
            &hLACE,
            buffer,
            x_in,
            features,
            numbits,
            periods
        );

        for (int n=0; n < 4 * LACE_FRAME_SIZE; n ++)
        {
            float tmp = (1UL<<15) * buffer[n];
            tmp = CLIP(tmp, INT16_MIN, INT16_MAX);
            x_out[n] = (int16_t) round(tmp);
        }

        fwrite(x_out, sizeof(int16_t), 4 * LACE_FRAME_SIZE, f_x_out);
    }
    printf("\ndone!\n");

    fclose(f_features);
    fclose(f_numbits);
    fclose(f_periods);
    fclose(f_x_in);
    fclose(f_x_out);
}

int main()
{
    LACE hLACE;

    lace_feature_net_compare("testvec2/lace", 5, &hLACE);

    lace_demo("testdata/test9", "out_lace_c_9kbps.pcm");
    lace_demo("testdata/test6", "out_lace_c_6kbps.pcm");


}
#endif

//gcc  -I ../include -I . -I ../celt osce.c nndsp.c lace_data.c nolace_data.c nnet.c parse_lpcnet_weights.c -lm -o lacetest