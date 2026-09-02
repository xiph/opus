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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "float_cast.h"
#include "opus.h"
#include "test_opus_common.h"
#include "opus_projection.h"
#include "mathops.h"
#include "../src/mapping_matrix.h"
#include "mathops.h"

#define BUFFER_SIZE 960
#define MAX_DATA_BYTES 32768
#define MAX_FRAME_SAMPLES 5760
#define ERROR_TOLERANCE 1

#define SIMPLE_MATRIX_SIZE 12
#define SIMPLE_MATRIX_FRAME_SIZE 10
#define SIMPLE_MATRIX_INPUT_SIZE 30
#define SIMPLE_MATRIX_OUTPUT_SIZE 40

/* Synthetic ambisonic test material: how many plane-wave sources it is built from,
 * and how much of it to run through the codec per order.
 */
#define MAX_AMBISONIC_ORDER MAPPING_MATRIX_MAX_AMBISONIC_ORDER
#define MAX_AMBISONIC_CHANNELS \
  ((MAX_AMBISONIC_ORDER + 1) * (MAX_AMBISONIC_ORDER + 1) + 2)
#define NB_TEST_SOURCES 4
#define NB_TEST_DIRECTION_SETS 2
#define TEST_FRAME_SIZE 960
#define NB_TEST_FRAMES 4
#define TEST_PI 3.141592653589793

/* With no codec in the way the matrix pair is exact to the Q15 quantization floor, and
 * measures 46-64 dB across the orders: highest at 1st order, falling slowly as the
 * accumulated rounding of more channels adds up, and ~12 dB lower at 2nd order, whose
 * demixing table gives up two bits to its 3050 gain field. A mismatched pair collapses
 * to ~0 dB, so this bound is deliberately loose.
 */
#define MATRIX_ROUNDTRIP_MIN_SNR_DB 35.0

/* Through the codec, what is left is the coding noise: ~27 dB at every order, except
 * 2nd order, which loses ~9 dB of it to a mixing matrix that is much worse conditioned
 * than its neighbours (cond 11.7 against 1.0-3.3). Again a loose bound: a matrix pair
 * that does not invert takes this to ~0 dB.
 */
#define CODEC_ROUNDTRIP_MIN_SNR_DB 12.0

int assert_is_equal(
  const opus_res *a, const opus_int16 *b, int size, opus_int16 tolerance)
{
  int i;
  for (i = 0; i < size; i++)
  {
    if (abs(RES2INT16(a[i]) - b[i]) > tolerance)
      return 1;
  }
  return 0;
}

int assert_is_equal_short(
  const opus_int16 *a, const opus_int16 *b, int size, opus_int16 tolerance)
{
  int i;
  for (i = 0; i < size; i++)
    if (abs(a[i] - b[i]) > tolerance)
      return 1;
  return 0;
}

void test_simple_matrix(void)
{
  const MappingMatrix simple_matrix_params = {4, 3, 0};
  const opus_int16 simple_matrix_data[SIMPLE_MATRIX_SIZE] = {0, 32767, 0, 0, 32767, 0, 0, 0, 0, 0, 0, 32767};
  const opus_int16 input_int16[SIMPLE_MATRIX_INPUT_SIZE] = {
    32767, 0, -32768, 29491, -3277, -29491, 26214, -6554, -26214, 22938, -9830,
    -22938, 19661, -13107, -19661, 16384, -16384, -16384, 13107, -19661, -13107,
    9830, -22938, -9830, 6554, -26214, -6554, 3277, -29491, -3277};
  const opus_int16 expected_output_int16[SIMPLE_MATRIX_OUTPUT_SIZE] = {
    0, 32767, 0, -32768, -3277, 29491, 0, -29491, -6554, 26214, 0, -26214,
    -9830, 22938, 0, -22938, -13107, 19661, 0, -19661, -16384, 16384, 0, -16384,
    -19661, 13107, 0, -13107, -22938, 9830, 0, -9830, -26214, 6554, 0, -6554,
    -29491, 3277, 0, -3277};

  int i, ret;
  opus_int32 simple_matrix_size;
  opus_res *input_pcm;
  opus_res *output_pcm;
  opus_int16 *output_int16;
  MappingMatrix *simple_matrix;

  /* Allocate input/output buffers. */
  input_pcm = (opus_res *)opus_alloc(sizeof(opus_res) * SIMPLE_MATRIX_INPUT_SIZE);
  output_int16 = (opus_int16 *)opus_alloc(sizeof(opus_int16) * SIMPLE_MATRIX_OUTPUT_SIZE);
  output_pcm = (opus_res *)opus_alloc(sizeof(opus_res) * SIMPLE_MATRIX_OUTPUT_SIZE);

  /* Initialize matrix */
  simple_matrix_size = mapping_matrix_get_size(simple_matrix_params.rows,
    simple_matrix_params.cols);
  if (!simple_matrix_size)
    test_failed();

  simple_matrix = (MappingMatrix *)opus_alloc(simple_matrix_size);
  mapping_matrix_init(simple_matrix, simple_matrix_params.rows,
    simple_matrix_params.cols, simple_matrix_params.gain, simple_matrix_data,
    sizeof(simple_matrix_data));

  /* Copy inputs. */
  for (i = 0; i < SIMPLE_MATRIX_INPUT_SIZE; i++)
  {
    input_pcm[i] = INT16TORES(input_int16[i]);
  }

  /* _in_short */
  for (i = 0; i < SIMPLE_MATRIX_OUTPUT_SIZE; i++)
    output_pcm[i] = 0;
  for (i = 0; i < simple_matrix->rows; i++)
  {
    mapping_matrix_multiply_channel_in_short(simple_matrix,
      input_int16, simple_matrix->cols, &output_pcm[i], i,
      simple_matrix->rows, SIMPLE_MATRIX_FRAME_SIZE);
  }
  ret = assert_is_equal(output_pcm, expected_output_int16, SIMPLE_MATRIX_OUTPUT_SIZE, ERROR_TOLERANCE);
  if (ret)
    test_failed();

  /* _out_short */
  for (i = 0; i < SIMPLE_MATRIX_OUTPUT_SIZE; i++)
    output_int16[i] = 0;
  for (i = 0; i < simple_matrix->cols; i++)
  {
    mapping_matrix_multiply_channel_out_short(simple_matrix,
      &input_pcm[i], i, simple_matrix->cols, output_int16,
      simple_matrix->rows, SIMPLE_MATRIX_FRAME_SIZE);
  }
  ret = assert_is_equal_short(output_int16, expected_output_int16, SIMPLE_MATRIX_OUTPUT_SIZE, ERROR_TOLERANCE);
  if (ret)
    test_failed();

#if !defined(DISABLE_FLOAT_API) && !defined(FIXED_POINT)
  /* _in_float */
  for (i = 0; i < SIMPLE_MATRIX_OUTPUT_SIZE; i++)
    output_pcm[i] = 0;
  for (i = 0; i < simple_matrix->rows; i++)
  {
    mapping_matrix_multiply_channel_in_float(simple_matrix,
      input_pcm, simple_matrix->cols, &output_pcm[i], i,
      simple_matrix->rows, SIMPLE_MATRIX_FRAME_SIZE);
  }
  ret = assert_is_equal(output_pcm, expected_output_int16, SIMPLE_MATRIX_OUTPUT_SIZE, ERROR_TOLERANCE);
  if (ret)
    test_failed();

  /* _out_float */
  for (i = 0; i < SIMPLE_MATRIX_OUTPUT_SIZE; i++)
    output_pcm[i] = 0;
  for (i = 0; i < simple_matrix->cols; i++)
  {
    mapping_matrix_multiply_channel_out_float(simple_matrix,
      &input_pcm[i], i, simple_matrix->cols, output_pcm,
      simple_matrix->rows, SIMPLE_MATRIX_FRAME_SIZE);
  }
  ret = assert_is_equal(output_pcm, expected_output_int16, SIMPLE_MATRIX_OUTPUT_SIZE, ERROR_TOLERANCE);
  if (ret)
    test_failed();
#endif

  opus_free(input_pcm);
  opus_free(output_int16);
  opus_free(output_pcm);
  opus_free(simple_matrix);
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

  is_channels_valid = (order_plus_one >= 2 &&
    order_plus_one <= MAX_AMBISONIC_ORDER + 1) &&
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

/* Plane-wave directions for the synthetic ambisonic material: four sources spread over
 * the sphere, none on an axis or in the horizontal plane, so that no ambisonic channel
 * of any order is left empty by symmetry.
 *
 * Two independent sets, because a codec round trip depends on the geometry more than on
 * the matrix: how sparsely four sources happen to land on the sectors decides how
 * easily the streams code, which moves the result by 1-2 dB from one set of directions
 * to another, in either direction. Averaging the sets measures the matrix instead of
 * the luck of the geometry.
 */
static const double test_source_azimuth[NB_TEST_DIRECTION_SETS][NB_TEST_SOURCES] = {
  {0.35, 2.20, -1.40, -2.85},
  {1.10, -0.62, 2.90, 0.05}
};
static const double test_source_elevation[NB_TEST_DIRECTION_SETS][NB_TEST_SOURCES] = {
  {0.20, -0.55, 0.95, -0.15},
  {-0.80, 0.40, 0.10, 1.30}
};

/* Real spherical harmonics up to `order` at azimuth `az` and elevation `el`, in ACN
 * channel order with SN3D normalization -- the convention channel mapping family 3
 * carries (RFC 8486 section 3.1). Writes (order+1)^2 coefficients to `y`, which is
 * the encoding of a unit-amplitude plane wave arriving from that direction.
 */
static void sh_acn_sn3d(int order, double az, double el, double *y)
{
  /* Associated Legendre functions are evaluated in the polar argument. */
  double ct = sin(el);
  double st = cos(el);
  int n, m, k;

  for (m = 0; m <= order; m++)
  {
    double cos_maz = cos(m * az);
    double sin_maz = sin(m * az);
    double pmm = 1.0;
    double p_prev1 = 0.0;
    double p_prev2 = 0.0;

    /* P_m^m(x) = (2m-1)!! (1-x^2)^(m/2), without the Condon-Shortley phase. */
    for (k = 1; k <= m; k++)
      pmm *= (2 * k - 1) * st;

    for (n = m; n <= order; n++)
    {
      double p, norm;

      if (n == m)
        p = pmm;
      else if (n == m + 1)
        p = ct * (2 * m + 1) * pmm;
      else
        p = ((2 * n - 1) * ct * p_prev1 - (n + m - 1) * p_prev2) / (n - m);

      /* SN3D: sqrt((2 - delta_m0) (n-m)! / (n+m)!), accumulated as a product of
       * square roots so that neither factorial is ever formed.
       */
      norm = (m == 0) ? 1.0 : sqrt(2.0);
      for (k = n - m + 1; k <= n + m; k++)
        norm /= sqrt((double)k);

      y[n * n + n + m] = norm * p * cos_maz;
      if (m > 0)
        y[n * n + n - m] = norm * p * sin_maz;

      p_prev2 = p_prev1;
      p_prev1 = p;
    }
  }
}

/* One channel of test material: a fundamental plus three partials, deterministic and
 * scaled so that its peak is `amplitude` of full scale.
 *
 * Tonal rather than noise-like on purpose. These tests measure a waveform SNR, and a
 * perceptual codec preserves the waveform of tonal material (~38 dB at 64 kbit/s per
 * channel) but not of noise, which measures near 0 dB waveform SNR however well it
 * codes perceptually. Noise-like material would make the bounds below meaningless.
 */
static void generate_tone(double *buf, int frame_size, double f0, double amplitude)
{
  static const double partial_gain[4] = {1.0, 0.5, 0.33, 0.2};
  double norm = 0;
  int i, p, nb_partials = 0;

  for (p = 0; p < 4; p++)
  {
    if (f0 * (p + 1) > 18000.0)
      break;
    norm += partial_gain[p];
    nb_partials++;
  }
  for (i = 0; i < frame_size; i++)
    buf[i] = 0;
  for (p = 0; p < nb_partials; p++)
  {
    double w = 2.0 * TEST_PI * f0 * (p + 1) / 48000.0;
    double g = amplitude * 32767.0 * partial_gain[p] / norm;
    for (i = 0; i < frame_size; i++)
      buf[i] += g * sin(w * i);
  }
}

static void quantize_to_short(const double *acc, short *buf, int size)
{
  int i;

  for (i = 0; i < size; i++)
  {
    double v = floor(acc[i] + 0.5);
    buf[i] = (short)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
  }
}

/* Order-`order` ambisonic test material: NB_TEST_SOURCES sources, each with its own
 * fundamental, encoded as plane waves from the directions above, in ACN/SN3D. Any
 * non-diegetic channels carry sources verbatim, so the identity block of the matrix is
 * exercised as well. Everything is scaled to a quarter of full scale, to leave the
 * mixing matrix headroom: it is an isometry, but a single sector can still be louder
 * than any one input channel.
 *
 * Note that this is spatially sparse material, so its energy per channel falls with
 * order: an SN3D plane wave puts unit energy in each order n, spread over 2n+1
 * channels. Use generate_ambisonic_diffuse() where equal energy per channel matters.
 */
static void generate_ambisonic(short *buf, int frame_size, int channels, int order,
                               int dirset)
{
  double y[MAX_AMBISONIC_CHANNELS];
  double *acc;
  double *src;
  int nb_acn = (order + 1) * (order + 1);
  int i, c, s;

  src = (double *)malloc(sizeof(double) * frame_size);
  acc = (double *)calloc((size_t)frame_size * channels, sizeof(double));
  if (!src || !acc)
    test_failed();

  for (s = 0; s < NB_TEST_SOURCES; s++)
  {
    generate_tone(src, frame_size, 220.0 * pow(2.0, 0.5 * s),
      0.25 / NB_TEST_SOURCES);
    sh_acn_sn3d(order, test_source_azimuth[dirset][s],
      test_source_elevation[dirset][s], y);
    for (i = 0; i < frame_size; i++)
    {
      for (c = 0; c < nb_acn; c++)
        acc[i * channels + c] += y[c] * src[i];
      if (nb_acn + s < channels)
        acc[i * channels + nb_acn + s] += src[i];
    }
  }

  quantize_to_short(acc, buf, frame_size * channels);
  free(src);
  free(acc);
}

/* A diffuse ambisonic field: every channel gets its own fundamental, all at the same
 * level. Spatially this is the opposite extreme from a plane wave, and it is the
 * material that makes a mapping matrix round trip measure the matrix rather than the
 * signal, since no channel is left quiet.
 *
 * Kept well below full scale because mapping_matrix_multiply_channel_out_short()
 * accumulates into opus_int16 without saturating: the reconstruction is bounded by the
 * input, but with a matrix that is not an isometry the partial sums along the way need
 * not be, and with loud uncorrelated channels they can wrap.
 */
static void generate_ambisonic_diffuse(short *buf, int frame_size, int channels)
{
  double *acc;
  double *src;
  int i, c;

  src = (double *)malloc(sizeof(double) * frame_size);
  acc = (double *)calloc((size_t)frame_size * channels, sizeof(double));
  if (!src || !acc)
    test_failed();

  for (c = 0; c < channels; c++)
  {
    generate_tone(src, frame_size, 150.0 + 31.0 * c, 1.0 / 16.0);
    for (i = 0; i < frame_size; i++)
      acc[i * channels + c] = src[i];
  }

  quantize_to_short(acc, buf, frame_size * channels);
  free(src);
  free(acc);
}

/* Signal-to-noise ratio of `test` against `ref`, in dB, after applying `gain` (the
 * matrix gain field, which libopus reports but never applies itself).
 */static double compute_snr(const short *ref, const short *test, int size, double gain)
{
  double signal = 0;
  double noise = 0;
  int i;

  for (i = 0; i < size; i++)
  {
    double r = ref[i];
    double d = gain * test[i] - r;
    signal += r * r;
    noise += d * d;
  }
  if (signal == 0)
    test_failed();
  if (noise == 0)
    return 999.0;
  return 10.0 * log10(signal / noise);
}

/* The gain field is stored as dB in S7.8 format. */
static double matrix_gain_to_linear(int gain)
{
  return pow(10.0, gain / (256.0 * 20.0));
}

/* Push a synthetic ambisonic signal through the built-in mixing matrix and back
 * through the built-in demixing matrix, with no codec in between. This is what makes
 * a projection stream decodable at all: the two tables have to invert each other to
 * within Q15 quantization, and nothing else in the test suite checks that.
 */
void test_matrix_roundtrip(int order)
{
  const MappingMatrix *mixing_table;
  const MappingMatrix *demixing_table;
  const opus_int16 *mixing_data;
  const opus_int16 *demixing_data;
  opus_int32 mixing_data_size, demixing_data_size;
  MappingMatrix *mixing_matrix;
  MappingMatrix *demixing_matrix;
  short *input;
  short *output;
  opus_res *sectors;
  double snr;
  int channels = (order + 1) * (order + 1) + 2;
  int nb_sectors;
  int i;

  if (mapping_matrix_get_ambisonic(order, 0, &mixing_table, &mixing_data,
        &mixing_data_size) != OPUS_OK)
    test_failed();
  if (mapping_matrix_get_ambisonic(order, 1, &demixing_table, &demixing_data,
        &demixing_data_size) != OPUS_OK)
    test_failed();
  if (mixing_table->rows != channels || mixing_table->cols != channels ||
      demixing_table->rows != channels || demixing_table->cols != channels)
  {
    fprintf(stderr, "Order %d: matrix is %dx%d, expected %dx%d.\n", order,
      mixing_table->rows, mixing_table->cols, channels, channels);
    test_failed();
  }
  nb_sectors = channels;

  mixing_matrix = (MappingMatrix *)opus_alloc(
    mapping_matrix_get_size(mixing_table->rows, mixing_table->cols));
  demixing_matrix = (MappingMatrix *)opus_alloc(
    mapping_matrix_get_size(demixing_table->rows, demixing_table->cols));
  input = (short *)malloc(sizeof(short) * TEST_FRAME_SIZE * channels);
  output = (short *)calloc((size_t)TEST_FRAME_SIZE * channels, sizeof(short));
  sectors = (opus_res *)calloc((size_t)TEST_FRAME_SIZE * nb_sectors,
    sizeof(opus_res));
  if (!mixing_matrix || !demixing_matrix || !input || !output || !sectors)
    test_failed();

  mapping_matrix_init(mixing_matrix, mixing_table->rows, mixing_table->cols,
    mixing_table->gain, mixing_data, mixing_data_size);
  mapping_matrix_init(demixing_matrix, demixing_table->rows,
    demixing_table->cols, demixing_table->gain, demixing_data,
    demixing_data_size);

  generate_ambisonic_diffuse(input, TEST_FRAME_SIZE, channels);

  for (i = 0; i < nb_sectors; i++)
    mapping_matrix_multiply_channel_in_short(mixing_matrix, input, channels,
      &sectors[i], i, nb_sectors, TEST_FRAME_SIZE);
  for (i = 0; i < nb_sectors; i++)
    mapping_matrix_multiply_channel_out_short(demixing_matrix, &sectors[i], i,
      nb_sectors, output, channels, TEST_FRAME_SIZE);

  /* The mixing gain is rolled into the demixing gain, so undoing the latter undoes
   * the whole round trip.
   */
  snr = compute_snr(input, output, TEST_FRAME_SIZE * channels,
    matrix_gain_to_linear(demixing_table->gain));
  fprintf(stderr, "  order %2d: %3d channels, gain %5d, matrix round trip "
    "%6.2f dB SNR\n", order, channels, demixing_table->gain, snr);
  if (snr < MATRIX_ROUNDTRIP_MIN_SNR_DB)
  {
    fprintf(stderr, "Order %d: matrix round trip is only %.2f dB.\n", order, snr);
    test_failed();
  }

  opus_free(mixing_matrix);
  opus_free(demixing_matrix);
  free(input);
  free(output);
  free(sectors);
}

/* Encode and decode a synthetic ambisonic signal through the projection API and return
 * how much of it survives, in dB. `nondiegetic` adds the two non-diegetic channels that
 * family 3 allows; `dirset` picks the source directions.
 */
static double ambisonic_codec_snr(int order, opus_int32 bitrate_per_channel,
                                  int nondiegetic, int dirset, int *streams_out,
                                  int *coupled_out)
{
  const opus_int32 Fs = 48000;
  const int application = OPUS_APPLICATION_AUDIO;
  const int total_samples = NB_TEST_FRAMES * TEST_FRAME_SIZE;

  OpusProjectionEncoder *st_enc;
  OpusProjectionDecoder *st_dec;
  short *input;
  short *output;
  unsigned char *data;
  unsigned char *matrix = NULL;
  opus_int32 matrix_size = 0;
  opus_int32 max_data_bytes;
  int channels = (order + 1) * (order + 1) + (nondiegetic ? 2 : 0);
  int streams, coupled, error, gain, lookahead;
  int f, len, out_samples, start, count;
  double snr;

  st_enc = opus_projection_ambisonics_encoder_create(Fs, channels, 3, &streams,
    &coupled, application, &error);
  if (error != OPUS_OK || st_enc == NULL)
  {
    fprintf(stderr, "Couldn't create encoder for order %d (%d channels).\n",
      order, channels);
    test_failed();
  }

  if (opus_projection_encoder_ctl(st_enc,
        OPUS_SET_BITRATE(bitrate_per_channel * channels)) != OPUS_OK ||
      opus_projection_encoder_ctl(st_enc, OPUS_GET_LOOKAHEAD(&lookahead))
        != OPUS_OK ||
      opus_projection_encoder_ctl(st_enc,
        OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN(&gain)) != OPUS_OK ||
      opus_projection_encoder_ctl(st_enc,
        OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST, &matrix_size)
        != OPUS_OK ||
      !matrix_size)
    test_failed();

  matrix = (unsigned char *)opus_alloc(matrix_size);
  if (!matrix)
    test_failed();
  if (opus_projection_encoder_ctl(st_enc,
        OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST, matrix, matrix_size)
        != OPUS_OK)
    test_failed();

  st_dec = opus_projection_decoder_create(Fs, channels, streams, coupled, matrix,
    matrix_size, &error);
  opus_free(matrix);
  if (error != OPUS_OK || st_dec == NULL)
  {
    fprintf(stderr, "Couldn't create decoder for order %d (%d channels, %d "
      "streams, %d coupled).\n", order, channels, streams, coupled);
    test_failed();
  }

  /* One 20 ms frame of every stream, with room to spare. */
  max_data_bytes = (opus_int32)(streams + coupled) * 1500 + 1024;
  input = (short *)malloc(sizeof(short) * total_samples * channels);
  output = (short *)calloc((size_t)total_samples * channels, sizeof(short));
  data = (unsigned char *)malloc(max_data_bytes);
  if (!input || !output || !data)
    test_failed();

  generate_ambisonic(input, total_samples, channels, order, dirset);

  for (f = 0; f < NB_TEST_FRAMES; f++)
  {
    len = opus_projection_encode(st_enc, input + f * TEST_FRAME_SIZE * channels,
      TEST_FRAME_SIZE, data, max_data_bytes);
    if (len < 0 || len > max_data_bytes)
    {
      fprintf(stderr, "opus_projection_encode() returned %d\n", len);
      test_failed();
    }
    out_samples = opus_projection_decode(st_dec, data, len,
      output + f * TEST_FRAME_SIZE * channels, TEST_FRAME_SIZE, 0);
    if (out_samples != TEST_FRAME_SIZE)
    {
      fprintf(stderr, "opus_projection_decode() returned %d\n", out_samples);
      test_failed();
    }
  }

  /* The decoder output lags its input by the encoder lookahead; skip the first
   * frame as well, so that the measurement is not dominated by codec start-up.
   */
  start = TEST_FRAME_SIZE;
  count = total_samples - lookahead - start;
  snr = compute_snr(input + start * channels,
    output + (start + lookahead) * channels, count * channels,
    matrix_gain_to_linear(gain));
  *streams_out = streams;
  *coupled_out = coupled;

  opus_projection_decoder_destroy(st_dec);
  opus_projection_encoder_destroy(st_enc);
  free(input);
  free(output);
  free(data);
  return snr;
}

/* Average over the direction sets, so that what is reported and bounded does not depend
 * on which four directions were picked.
 */
void test_ambisonic_encode_decode(int order, opus_int32 bitrate_per_channel,
                                 int nondiegetic)
{
  double snr, best, worst, mean = 0;
  int channels = (order + 1) * (order + 1) + (nondiegetic ? 2 : 0);
  int streams = 0, coupled = 0, d;

  best = worst = 0;
  for (d = 0; d < NB_TEST_DIRECTION_SETS; d++)
  {
    snr = ambisonic_codec_snr(order, bitrate_per_channel, nondiegetic, d, &streams,
      &coupled);
    mean += snr / NB_TEST_DIRECTION_SETS;
    if (d == 0 || snr > best)
      best = snr;
    if (d == 0 || snr < worst)
      worst = snr;
  }

  fprintf(stderr, "  order %2d: %3d channels, %3d streams (%3d coupled), "
    "%3d kbit/s/ch, codec round trip %6.2f dB SNR (spread %4.2f)\n", order, channels,
    streams, coupled, bitrate_per_channel / 1000, mean, best - worst);
  if (mean < CODEC_ROUNDTRIP_MIN_SNR_DB)
  {
    fprintf(stderr, "Order %d: codec round trip is only %.2f dB.\n", order, mean);
    test_failed();
  }
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

  opus_projection_decoder_destroy(st_dec);
  opus_projection_encoder_destroy(st_enc);
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
  int order;

  (void)_argc;
  (void)_argv;

  /* Test simple matrix multiplication routines. */
  test_simple_matrix();

  /* Test full range of channels in creation arguments. */
  for (i = 0; i < 255; i++)
    test_creation_arguments(i, 3);

  /* Test that every built-in matrix pair inverts itself. */
  fprintf(stderr, "Mapping matrix round trip:\n");
  for (order = 1; order <= MAX_AMBISONIC_ORDER; order++)
    test_matrix_roundtrip(order);

  /* Test encode/decode pipeline. */
  test_encode_decode(64 * 18, 18, 3);

  /* Test encode/decode of synthetic ambisonic material at every order, with and
     without the two non-diegetic channels. */
  fprintf(stderr, "Ambisonic encode/decode:\n");
  for (order = 1; order <= MAX_AMBISONIC_ORDER; order++)
  {
    test_ambisonic_encode_decode(order, 64000, 1);
    test_ambisonic_encode_decode(order, 64000, 0);
  }

  fprintf(stderr, "All projection tests passed.\n");
  return 0;
}
