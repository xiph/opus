/* Copyright (c) 2017 Google Inc.
   Written by Andrew Allen */
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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "float_cast.h"
#include "opus.h"
#include "test_opus_common.h"
#include "opus_projection.h"
#include "mathops.h"
#include "../src/mapping_matrix.c"
#include "mathops.c"

#ifdef ENABLE_EXPERIMENTAL_AMBISONICS

#define BUFFER_SIZE 960
#define MAX_DATA_BYTES 32768
#define MAX_FRAME_SAMPLES 5760

#define INT16_TO_FLOAT(x) ((1/32768.f)*(float)x)

void print_matrix_short(const opus_int16 *data, int rows, int cols)
{
  int i, j;
  for (i = 0; i < rows; i++)
  {
    for (j = 0; j < cols; j++)
    {
      fprintf(stderr, "%8.5f  ", (float)INT16_TO_FLOAT(data[j * rows + i]));
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

void print_matrix_float(const float *data, int rows, int cols)
{
  int i, j;
  for (i = 0; i < rows; i++)
  {
    for (j = 0; j < cols; j++)
    {
      fprintf(stderr, "%8.5f ", data[j * rows + i]);
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

void print_matrix(MappingMatrix *matrix)
{
  opus_int16 *data;

  fprintf(stderr, "%d x %d, gain: %d\n", matrix->rows, matrix->cols,
    matrix->gain);

  data = mapping_matrix_get_data(matrix);
  print_matrix_short(data, matrix->rows, matrix->cols);
}

int assert_transform_short(
  const opus_int16 *a, const opus_int16 *b, int size, opus_int16 tolerance)
{
  int i;
  for (i = 0; i < size; i++)
  {
    if (abs(a[i] - b[i]) > tolerance)
    {
      return 0;
    }
  }
  return 1;
}

int assert_transform_float(
  const float *a, const float *b, int size, float tolerance)
{
  int i;
  for (i = 0; i < size; i++)
  {
    if (fabsf(a[i] - b[i]) > tolerance)
    {
      return 0;
    }
  }
  return 1;
}

void test_matrix_transform(void)
{
  /* Create testing mixing matrix (4 x 3), gain 0dB:
  *   [ 0 1 0 ]
  *   [ 1 0 0 ]
  *   [ 0 0 0 ]
  *   [ 0 0 1 ]
  */
  opus_int32 matrix_size;
  MappingMatrix *testing_matrix;
  const opus_int16 testing_matrix_data[12] = {
    0, 32767, 0, 0, 32767, 0, 0, 0, 0, 0, 0, 32767 };

  const int frame_size = 10;
  const opus_int16 input[30] = {
    32767, 0, -32768, 29491, -3277, -29491, 26214, -6554, -26214, 22938, -9830,
    -22938, 19661, -13107, -19661, 16384, -16384, -16384, 13107, -19661, -13107,
    9830, -22938, -9830, 6554, -26214, -6554, 3277, -29491, -3277};
  const opus_int16 expected_output[40] = {
    0, 32767, 0, -32768, -3277, 29491, 0, -29491, -6554, 26214, 0, -26214,
    -9830, 22938, 0, -22938, -13107, 19661, 0, -19661, -16384, 16384, 0, -16384,
    -19661, 13107, 0, -13107, -22938, 9830, 0, -9830, -26214, 6554, 0, -6554,
    -29491, 3277, 0, -3277};
  opus_int16 output[40] = {0};

#ifndef DISABLE_FLOAT_API
  int i;
  /* Sample-accurate to -93.9794 dB */
  float flt_tolerance = 2e-5f;
  float input32[30] = {0};
  float output32[40] = {0};
  float expected_output32[40] = {0};

  /* Convert short to float representations. */
  for (i = 0; i < 30; i++)
  {
    input32[i] = INT16_TO_FLOAT(input[i]);
  }
  for (i = 0; i < 40; i++)
  {
    expected_output32[i] = INT16_TO_FLOAT(expected_output[i]);
  }
#endif /* DISABLE_FLOAT_API */

  /* Create the matrix. */
  matrix_size = mapping_matrix_get_size(4, 3);
  testing_matrix = (MappingMatrix *)opus_alloc(matrix_size);
  mapping_matrix_init(testing_matrix, 4, 3, 0, testing_matrix_data,
    12 * sizeof(opus_int16));

  mapping_matrix_multiply_short(testing_matrix, input, testing_matrix->cols,
    output, testing_matrix->rows, frame_size);
  if (!assert_transform_short(output, expected_output, 40, 1))
  {
    fprintf(stderr, "Matrix:\n");
    print_matrix(testing_matrix);

    fprintf(stderr, "Input (short):\n");
    print_matrix_short(input, testing_matrix->cols, frame_size);

    fprintf(stderr, "Expected Output (short):\n");
    print_matrix_short(expected_output, testing_matrix->rows, frame_size);

    fprintf(stderr, "Output (short):\n");
    print_matrix_short(output, testing_matrix->rows, frame_size);

    goto bad_cleanup;
  }

#ifndef DISABLE_FLOAT_API
  mapping_matrix_multiply_float(testing_matrix, input32, testing_matrix->cols,
    output32, testing_matrix->rows, frame_size);
  if (!assert_transform_float(output32, expected_output32, 40, flt_tolerance))
  {
    fprintf(stderr, "Matrix:\n");
    print_matrix(testing_matrix);

    fprintf(stderr, "Input (float):\n");
    print_matrix_float(input32, testing_matrix->cols, frame_size);

    fprintf(stderr, "Expected Output (float):\n");
    print_matrix_float(expected_output32, testing_matrix->rows, frame_size);

    fprintf(stderr, "Output (float):\n");
    print_matrix_float(output32, testing_matrix->rows, frame_size);

    goto bad_cleanup;
  }
#endif
  opus_free(testing_matrix);
  return;
bad_cleanup:
  opus_free(testing_matrix);
  test_failed();
}

void test_creation_arguments(const int channels, const int mapping_family)
{
  int streams;
  int coupled_streams;
  int enc_error;
  int dec_error;
  int ret;
  OpusProjectionEncoder *st_enc = NULL;
  OpusProjectionDecoder *st_dec = NULL;

  const opus_int32 Fs = 48000;
  const int application = OPUS_APPLICATION_AUDIO;

  int order_plus_one = (int)floor(sqrt((float)channels));
  int nondiegetic_channels = channels - order_plus_one * order_plus_one;

  int is_channels_valid = 0;
  int is_projection_valid = 0;

  st_enc = opus_projection_ambisonics_encoder_create(Fs, channels,
    mapping_family, &streams, &coupled_streams, application, &enc_error);
  if (st_enc != NULL)
  {
    opus_int32 matrix_size;
    unsigned char *matrix;

    ret = opus_projection_encoder_ctl(st_enc,
      OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST, &matrix_size);
    if (ret != OPUS_OK || !matrix_size)
      test_failed();

    matrix = (unsigned char *)opus_alloc(matrix_size);
    ret = opus_projection_encoder_ctl(st_enc,
      OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST, matrix, matrix_size);

    opus_projection_encoder_destroy(st_enc);

    st_dec = opus_projection_decoder_create(Fs, channels, streams,
      coupled_streams, matrix, matrix_size, &dec_error);
    if (st_dec != NULL)
    {
      opus_projection_decoder_destroy(st_dec);
    }
    opus_free(matrix);
  }

  is_channels_valid = (order_plus_one >= 2 && order_plus_one <= 4) &&
    (nondiegetic_channels == 0 || nondiegetic_channels == 2);
  is_projection_valid = (enc_error == OPUS_OK && dec_error == OPUS_OK);
  if (is_channels_valid ^ is_projection_valid)
  {
    fprintf(stderr, "Channels: %d, Family: %d\n", channels, mapping_family);
    fprintf(stderr, "Order+1: %d, Non-diegetic Channels: %d\n",
      order_plus_one, nondiegetic_channels);
    fprintf(stderr, "Streams: %d, Coupled Streams: %d\n",
      streams, coupled_streams);
    test_failed();
  }
}

void generate_music(short *buf, opus_int32 len, opus_int32 channels)
{
   opus_int32 i,j,k;
   opus_int32 *a,*b,*c,*d;
   a = (opus_int32 *)malloc(sizeof(opus_int32) * channels);
   b = (opus_int32 *)malloc(sizeof(opus_int32) * channels);
   c = (opus_int32 *)malloc(sizeof(opus_int32) * channels);
   d = (opus_int32 *)malloc(sizeof(opus_int32) * channels);
   memset(a, 0, sizeof(opus_int32) * channels);
   memset(b, 0, sizeof(opus_int32) * channels);
   memset(c, 0, sizeof(opus_int32) * channels);
   memset(d, 0, sizeof(opus_int32) * channels);
   j=0;

   for(i=0;i<len;i++)
   {
     for(k=0;k<channels;k++)
     {
      opus_uint32 r;
      opus_int32 v;
      v=(((j*((j>>12)^((j>>10|j>>12)&26&j>>7)))&128)+128)<<15;
      r=fast_rand();v+=r&65535;v-=r>>16;
      b[k]=v-a[k]+((b[k]*61+32)>>6);a[k]=v;
      c[k]=(30*(c[k]+b[k]+d[k])+32)>>6;d[k]=b[k];
      v=(c[k]+128)>>8;
      buf[i*channels+k]=v>32767?32767:(v<-32768?-32768:v);
      if(i%6==0)j++;
     }
   }

   free(a);
   free(b);
   free(c);
   free(d);
}

void test_encode_decode(opus_int32 bitrate, opus_int32 channels,
                        const int mapping_family)
{
  const opus_int32 Fs = 48000;
  const int application = OPUS_APPLICATION_AUDIO;

  OpusProjectionEncoder *st_enc;
  OpusProjectionDecoder *st_dec;
  int streams;
  int coupled;
  int error;
  short *buffer_in;
  short *buffer_out;
  unsigned char data[MAX_DATA_BYTES] = { 0 };
  int len;
  int out_samples;
  opus_int32 matrix_size = 0;
  unsigned char *matrix = NULL;

  buffer_in = (short *)malloc(sizeof(short) * BUFFER_SIZE * channels);
  buffer_out = (short *)malloc(sizeof(short) * BUFFER_SIZE * channels);

  st_enc = opus_projection_ambisonics_encoder_create(Fs, channels,
    mapping_family, &streams, &coupled, application, &error);
  if (error != OPUS_OK) {
    fprintf(stderr,
      "Couldn\'t create encoder with %d channels and mapping family %d.\n",
      channels, mapping_family);
    free(buffer_in);
    free(buffer_out);
    test_failed();
  }

  error = opus_projection_encoder_ctl(st_enc,
    OPUS_SET_BITRATE(bitrate * 1000 * (streams + coupled)));
  if (error != OPUS_OK)
  {
    goto bad_cleanup;
  }

  error = opus_projection_encoder_ctl(st_enc,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST, &matrix_size);
  if (error != OPUS_OK || !matrix_size)
  {
    goto bad_cleanup;
  }

  matrix = (unsigned char *)opus_alloc(matrix_size);
  error = opus_projection_encoder_ctl(st_enc,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST, matrix, matrix_size);

  st_dec = opus_projection_decoder_create(Fs, channels, streams, coupled,
    matrix, matrix_size, &error);
  opus_free(matrix);

  if (error != OPUS_OK) {
    fprintf(stderr,
      "Couldn\'t create decoder with %d channels, %d streams "
      "and %d coupled streams.\n", channels, streams, coupled);
    goto bad_cleanup;
  }

  generate_music(buffer_in, BUFFER_SIZE, channels);

  len = opus_projection_encode(
    st_enc, buffer_in, BUFFER_SIZE, data, MAX_DATA_BYTES);
  if(len<0 || len>MAX_DATA_BYTES) {
    fprintf(stderr,"opus_encode() returned %d\n", len);
    goto bad_cleanup;
  }

  out_samples = opus_projection_decode(
    st_dec, data, len, buffer_out, MAX_FRAME_SAMPLES, 0);
  if(out_samples!=BUFFER_SIZE) {
    fprintf(stderr,"opus_decode() returned %d\n", out_samples);
    goto bad_cleanup;
  }

  free(buffer_in);
  free(buffer_out);
  return;
bad_cleanup:
  free(buffer_in);
  free(buffer_out);
  test_failed();
}

int main(int _argc, char **_argv)
{
  unsigned int i;

  (void)_argc;
  (void)_argv;

  /* Test matrix creation/multiplication. */
  test_matrix_transform();

  /* Test full range of channels in creation arguments. */
  for (i = 0; i < 255; i++)
    test_creation_arguments(i, 253);

  /* Test encode/decode pipeline. */
  test_encode_decode(64 * 16, 16, 253);

  fprintf(stderr, "All projection tests passed.\n");
  return 0;
}

#else

int main(int _argc, char **_argv)
{
  (void)_argc;
  (void)_argv;
  fprintf(stderr, "Projection tests are disabled. "
          "Configure with --enable-ambisonics for support.\n");
  return 0;
}

#endif /* ENABLE_EXPERIMENTAL_AMBISONICS */
