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

#include "arch.h"
#include "float_cast.h"
#include "opus_private.h"
#include "opus_defines.h"
#include "mapping_matrix.h"

#define MATRIX_INDEX(nb_rows, row, col) (nb_rows * col + row)

opus_int32 mapping_matrix_get_size(int rows, int cols)
{
  opus_int32 size;

  /* Mapping Matrix must only support up to 255 channels in or out.
   * Note that a matrix larger than MAPPING_MATRIX_MAX_OGG_HEADER_OCTETS cannot be
   * stored in an Ogg OpusHead; that is a container limit, checked by the muxer, and
   * it excludes only 13th- and 14th-order ambisonics.
   */
  if (rows > 255 || cols > 255)
      return 0;
  size = rows * (opus_int32)cols * sizeof(opus_int16);

  return align(sizeof(MappingMatrix)) + align(size);
}

opus_int16 *mapping_matrix_get_data(const MappingMatrix *matrix)
{
  /* void* cast avoids clang -Wcast-align warning */
  return (opus_int16*)(void*)((char*)matrix + align(sizeof(MappingMatrix)));
}

void mapping_matrix_init(MappingMatrix * const matrix,
  int rows, int cols, int gain, const opus_int16 *data, opus_int32 data_size)
{
  int i;
  opus_int16 *ptr;

#if !defined(ENABLE_ASSERTIONS)
  (void)data_size;
#endif
  celt_assert(align(data_size) == align(rows * cols * sizeof(opus_int16)));

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->gain = gain;
  ptr = mapping_matrix_get_data(matrix);
  for (i = 0; i < rows * cols; i++)
  {
     ptr[i] = data[i];
  }
}

void mapping_matrix_init_transposed(MappingMatrix * const matrix,
  int rows, int cols, int gain, const opus_int16 *data, opus_int32 data_size)
{
  int row, col;
  opus_int16 *ptr;

#if !defined(ENABLE_ASSERTIONS)
  (void)data_size;
#endif
  celt_assert(align(data_size) == align(rows * cols * sizeof(opus_int16)));

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->gain = gain;
  ptr = mapping_matrix_get_data(matrix);
  /* `data` is a cols x rows matrix, also stored col-wise. */
  for (col = 0; col < cols; col++)
  {
    for (row = 0; row < rows; row++)
    {
      ptr[MATRIX_INDEX(rows, row, col)] = data[MATRIX_INDEX(cols, col, row)];
    }
  }
}

#ifndef DISABLE_FLOAT_API
void mapping_matrix_multiply_channel_in_float(
    const MappingMatrix *matrix,
    const float *input,
    int input_rows,
    opus_res *output,
    int output_row,
    int output_rows,
    int frame_size)
{
  /* Matrix data is ordered col-wise. */
  opus_int16* matrix_data;
  int i, col;

  celt_assert(input_rows <= matrix->cols && output_rows <= matrix->rows);

  matrix_data = mapping_matrix_get_data(matrix);

  for (i = 0; i < frame_size; i++)
  {
    float tmp = 0;
    for (col = 0; col < input_rows; col++)
    {
      tmp +=
        matrix_data[MATRIX_INDEX(matrix->rows, output_row, col)] *
        input[MATRIX_INDEX(input_rows, col, i)];
    }
    output[output_rows * i] = FLOAT2RES((1/32768.f)*tmp);
  }
}

void mapping_matrix_multiply_channel_out_float(
    const MappingMatrix *matrix,
    const opus_res *input,
    int input_row,
    int input_rows,
    float *output,
    int output_rows,
    int frame_size
)
{
  /* Matrix data is ordered col-wise. */
  opus_int16* matrix_data;
  int i, row;
  float input_sample;

  celt_assert(input_rows <= matrix->cols && output_rows <= matrix->rows);

  matrix_data = mapping_matrix_get_data(matrix);

  for (i = 0; i < frame_size; i++)
  {
    input_sample = RES2FLOAT(input[input_rows * i]);
    for (row = 0; row < output_rows; row++)
    {
      float tmp =
        (1/32768.f)*matrix_data[MATRIX_INDEX(matrix->rows, row, input_row)] *
        input_sample;
      output[MATRIX_INDEX(output_rows, row, i)] += tmp;
    }
  }
}
#endif /* DISABLE_FLOAT_API */

void mapping_matrix_multiply_channel_in_short(
    const MappingMatrix *matrix,
    const opus_int16 *input,
    int input_rows,
    opus_res *output,
    int output_row,
    int output_rows,
    int frame_size)
{
  /* Matrix data is ordered col-wise. */
  opus_int16* matrix_data;
  int i, col;

  celt_assert(input_rows <= matrix->cols && output_rows <= matrix->rows);

  matrix_data = mapping_matrix_get_data(matrix);

  for (i = 0; i < frame_size; i++)
  {
    opus_val32 tmp = 0;
    for (col = 0; col < input_rows; col++)
    {
#if defined(FIXED_POINT)
      tmp +=
        ((opus_int32)matrix_data[MATRIX_INDEX(matrix->rows, output_row, col)] *
        (opus_int32)input[MATRIX_INDEX(input_rows, col, i)]) >> 8;
#else
      tmp +=
        matrix_data[MATRIX_INDEX(matrix->rows, output_row, col)] *
        input[MATRIX_INDEX(input_rows, col, i)];
#endif
    }
#if defined(FIXED_POINT)
#ifdef ENABLE_RES24
    output[output_rows * i] = SHL32(tmp, RES_SHIFT-7);
#else
    output[output_rows * i] = SATURATE16((tmp + 64) >> 7);
#endif
#else
    output[output_rows * i] = (1/(32768.f*32768.f))*tmp;
#endif
  }
}

void mapping_matrix_multiply_channel_out_short(
    const MappingMatrix *matrix,
    const opus_res *input,
    int input_row,
    int input_rows,
    opus_int16 *output,
    int output_rows,
    int frame_size)
{
  /* Matrix data is ordered col-wise. */
  opus_int16* matrix_data;
  int i, row;
  opus_int32 input_sample;

  celt_assert(input_rows <= matrix->cols && output_rows <= matrix->rows);

  matrix_data = mapping_matrix_get_data(matrix);

  for (i = 0; i < frame_size; i++)
  {
    input_sample = RES2INT16(input[input_rows * i]);
    for (row = 0; row < output_rows; row++)
    {
      opus_int32 tmp =
        (opus_int32)matrix_data[MATRIX_INDEX(matrix->rows, row, input_row)] *
        input_sample;
      output[MATRIX_INDEX(output_rows, row, i)] += (tmp + 16384) >> 15;
    }
  }
}

void mapping_matrix_multiply_channel_in_int24(
    const MappingMatrix *matrix,
    const opus_int32 *input,
    int input_rows,
    opus_res *output,
    int output_row,
    int output_rows,
    int frame_size)
{
  /* Matrix data is ordered col-wise. */
  opus_int16* matrix_data;
  int i, col;

  celt_assert(input_rows <= matrix->cols && output_rows <= matrix->rows);

  matrix_data = mapping_matrix_get_data(matrix);

  for (i = 0; i < frame_size; i++)
  {
    opus_val64 tmp = 0;
    for (col = 0; col < input_rows; col++)
    {
      tmp +=
        matrix_data[MATRIX_INDEX(matrix->rows, output_row, col)] *
        (opus_val64)input[MATRIX_INDEX(input_rows, col, i)];
    }
#if defined(FIXED_POINT)
    output[output_rows * i] = INT24TORES((tmp + 16384) >> 15);
#else
    output[output_rows * i] = INT24TORES((1/(32768.f))*tmp);
#endif
  }
}

void mapping_matrix_multiply_channel_out_int24(
    const MappingMatrix *matrix,
    const opus_res *input,
    int input_row,
    int input_rows,
    opus_int32 *output,
    int output_rows,
    int frame_size)
{
  /* Matrix data is ordered col-wise. */
  opus_int16* matrix_data;
  int i, row;
  opus_int32 input_sample;

  celt_assert(input_rows <= matrix->cols && output_rows <= matrix->rows);

  matrix_data = mapping_matrix_get_data(matrix);

  for (i = 0; i < frame_size; i++)
  {
    input_sample = RES2INT24(input[input_rows * i]);
    for (row = 0; row < output_rows; row++)
    {
      opus_int64 tmp =
        (opus_int64)matrix_data[MATRIX_INDEX(matrix->rows, row, input_row)] *
        input_sample;
      output[MATRIX_INDEX(output_rows, row, i)] += (tmp + 16384) >> 15;
    }
  }
}


/* Projection mixing matrices for 1st to 5th-order ambisonics; 6th to 14th order are
 * in mapping_matrix_hoa.c.
 *
 * These are maximum-determinant ("Fekete") node sets -- point sets of exactly
 * (N+1)^2 points chosen to maximize |det Y|, i.e. to optimize the very
 * spherical-harmonic interpolation matrix in question -- with the resulting matrix
 * replaced by its polar factor U.V^T, the nearest matrix with orthonormal columns.
 * That is energy-preserving ambisonic decoding (Zotter, Pomberger & Noisternig,
 * Acta Acustica 98(1), 2012) at the critically sampled limit L = (N+1)^2, and it
 * gives, at every order:
 *
 *   - cond(A) = 1, so the demixing step amplifies coding noise by 0 dB. The
 *     figure of merit is ||A||_F.||A^-1||_F/N, the factor by which per-sector
 *     coding noise comes back louder in the ambisonic domain; it is 0 dB if and
 *     only if the matrix is a scaled orthogonal one.
 *   - no gain field, since neither matrix exceeds unity magnitude;
 *   - a demixing matrix that is the exact transpose of the mixing matrix, so only
 *     the mixing matrix is stored and mapping_matrix_init_transposed() derives the
 *     other -- which halves the table .rodata and leaves the demixing matrix with
 *     no rounding error of its own relative to the matrix it has to invert;
 *   - a Q15 round trip whose error stays at the quantization floor (~1e-4) rather
 *     than growing with order.
 *
 * Generated with AmbisonicArrays.jl.
 */

/* Order-1 ambisonics: 4 ACN channels + 2 non-diegetic = 6 streams.
   maxDet (Fekete) nodes, degree 1 (4 points), orthogonalized: the mixing
   matrix has orthonormal columns, so cond = 1, no gain field is needed, and the
   demixing matrix is the exact transpose of the mixing matrix.
   Its demixing matrix is derived by transposing this table at init time. */
const MappingMatrix mapping_matrix_foa_mixing = { 6, 6, 0 };
const opus_int16 mapping_matrix_foa_mixing_data[36] = {
     16384,      0,  28377,      0,      0,      0,  16384,      0,
     -9459,  26754,      0,      0,  16384,  23170,  -9459, -13377,
         0,      0,  16384, -23170,  -9459, -13377,      0,      0,
         0,      0,      0,      0,  32767,      0,      0,      0,
         0,      0,      0,  32767
};

#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 2
/* Order-2 ambisonics: 9 ACN channels + 2 non-diegetic = 11 streams.
   maxDet (Fekete) nodes, degree 2 (9 points), orthogonalized: the mixing
   matrix has orthonormal columns, so cond = 1, no gain field is needed, and the
   demixing matrix is the exact transpose of the mixing matrix.
   Its demixing matrix is derived by transposing this table at init time. */
const MappingMatrix mapping_matrix_soa_mixing = { 11, 11, 0 };
const opus_int16 mapping_matrix_soa_mixing_data[121] = {
     11093,      0,  17235,      0,  -3001,      0,  25383,      0,
      -487,      0,      0,  10836,   -877,   7235,  18308,   1071,
     -1910,  -9199,  14044,  16772,      0,      0,  10836,  -8667,
    -14469, -10189,  12720,  12258,   6633,  14410,   2066,      0,
         0,  11093,  11369,  -8618,  -9671, -18149, -13498,  -1198,
     11482,  -2948,      0,      0,  10836,   8667, -14469,  10189,
     12720, -12258,   6633, -14410,   2066,      0,      0,  10836,
     18212,   7235,   2070,   6325,  14168,  -9199,    366, -15571,
         0,      0,  10836, -18212,   7235,  -2070,   6325, -14168,
     -9199,   -366, -15571,      0,      0,  10836,    877,   7235,
    -18308,   1071,   1910,  -9199, -14044,  16772,      0,      0,
     11093, -11369,  -8618,   9671, -18149,  13498,  -1198, -11482,
     -2948,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,  32767,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
     32767
};
#endif

#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 3
/* Order-3 ambisonics: 16 ACN channels + 2 non-diegetic = 18 streams.
   maxDet (Fekete) nodes, degree 3 (16 points), orthogonalized: the mixing
   matrix has orthonormal columns, so cond = 1, no gain field is needed, and the
   demixing matrix is the exact transpose of the mixing matrix.
   Its demixing matrix is derived by transposing this table at init time. */
const MappingMatrix mapping_matrix_toa_mixing = { 18, 18, 0 };
const opus_int16 mapping_matrix_toa_mixing_data[324] = {
      7921,    250,  13870,     79,     55,    305,  17870,     96,
       -79,  -1066,  -2145,   -125,  21953,    -39,   3075,  -1383,
         0,      0,   7921,    250,  -6179,  12417,    297,    -88,
     -3614, -12425,  12325,  -1582,  -1720,    541,   6901,    584,
    -15916,  13858,      0,      0,   7921,  13629,  -1356,  -2204,
     -4607,  -2835,  -8740,    521, -14612, -16537,  -2451, -11245,
      3890,   1960,   3349,   7935,      0,      0,   7921,   7310,
     -6179, -10040, -11670,  -7182,  -3614,  10139,   3975,  13805,
     15531,    778,   6901,   -170,  -3879,   1989,      0,      0,
      7921, -10469,   -985,   9047, -15265,   1780,  -8720,  -1647,
     -2127,  -8722,   3287,   9619,   3630, -11187,   3144, -13094,
         0,      0,   8956,   7186,   7913, -10663, -11406,   8464,
     -1724, -12559,   4619,   9446, -13537,   1416,  -8327,  -3638,
      4761,   1426,      0,      0,   7921,  -6757,   8521,  -8612,
      9297,  -9407,   1384, -11735,   2216,  -5763,  16009,  -3560,
     -8020, -10250,   3129,   4208,      0,      0,   8956, -14164,
     -2740,  -4453,   9388,   5778,  -8827,   1816, -13455,   9057,
     -3357,   9159,   5937,   2879,   4812,  11742,      0,      0,
      7921, -10469,   8521,   3197,  -5289, -14429,   1384,   4241,
     -7960,   5534,  -8462,  -8785,  -8020,   6369, -13945,  -4505,
         0,      0,   7921,  -3409, -11528,  -6922,   3735,   6253,
      9435,  12950,   3111,  -5068,  -8520,  -5460,  -3480, -17764,
     -6385,   1586,      0,      0,   8956,   -208,   7913,  12857,
      -398,   -245,  -1724,  15143,  12299,  -1020,    204,   -920,
     -8327,   3793,  14348,   9498,      0,      0,   7921,  -6757,
    -11528,   3726,  -4209,  12538,   9435,  -7043,  -2432,   2821,
      8934, -14642,  -3480,  11445,   5793,  -4499,      0,      0,
      8956,   7186, -13085,   2259,   2416, -13996,  12275,  -4400,
     -3463,  -1776,  -4287,  15904,  -6263,   5000,   6144,  -2303,
         0,      0,   7921,  -3409,   -985, -13410,   7265,    517,
     -8720,   2370,  13592, -10449,  -4085,   1487,   3630,  14679,
     -2000, -11761,      0,      0,   7921,   9916,  -1356,   9606,
     15304,  -2027,  -8740,  -2050,   -720,  11875,  -2298,  -8101,
      3890,  -8041,   3456, -13979,      0,      0,   7921,   9916,
      9186,   3117,   5086,  14576,   2640,   4582,  -7290,  -2606,
      9637,  10594,  -8593,   3331, -13813,  -3379,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
     32767,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,  32767
};
#endif

#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 4
/* Order-4 ambisonics: 25 ACN channels + 2 non-diegetic = 27 streams.
   maxDet (Fekete) nodes, degree 4 (25 points), orthogonalized: the mixing
   matrix has orthonormal columns, so cond = 1, no gain field is needed, and the
   demixing matrix is the exact transpose of the mixing matrix.
   Its demixing matrix is derived by transposing this table at init time. */
const MappingMatrix mapping_matrix_fourthoa_mixing = { 27, 27, 0 };
const opus_int16 mapping_matrix_fourthoa_mixing_data[729] = {
      6669,   -164,  11111,   -170,   1015,    405,  14742,    253,
       175,   -618,   1645,   -414,  17596,    383,  -1267,    -88,
     -1450,   -391,   3034,    356,  18781,   -301,  -1590,   1867,
      1433,      0,      0,   6694,     -4,  -5193,  10931,   -737,
       581,  -2974, -10392,  10047,   -778,    -74,    993,   8069,
      -701, -11640,   9843,    -82,   -574,    140,   1939,  -4047,
     11940,   3327, -11832,   7073,      0,      0,   6696,   3600,
      2010, -10515,  -7730,   2749,  -6464,  -4907,   9948,   9868,
     -4384,  -2457,  -4311,   6730,   4404,  -7230, -15013,   8110,
      3474,  -2768,   4589,   6419,  -4422,   -858,   4191,      0,
         0,   6558, -11451,    429,   1475,  -3218,   -914,  -7256,
      1186, -12598,  13417,  -2205,   9894,  -1913,   -448,  -1587,
     -6145,   7699,    121,   2072,   5011,   6107,   -141,   9392,
     -1751,  11359,      0,      0,   6472,  -8590,   5408,  -4774,
      8828,  -8233,  -1927,  -5564,  -4630,  -1067,  10280,   -795,
     -6760,   -577,  -6521,   9657,  -9793,    338,   4259,   8431,
     -1977,   5984,   -781,  13466,  -2830,      0,      0,   6669,
      5150,  -5310,  -8294,  -8466,  -6504,  -2792,   9242,   3379,
      8918,  11648,   1799,   8757,    190,  -4320,    895,  -4348,
    -13971,  -4956,   4286,  -5398,  -8405,   -401,  -2609,  -2944,
         0,      0,   5961,  -1787,  -9872,   4008,  -1802,   3563,
      9779,  -7626,   2042,  -1477,   3609,  -5982,  -9576,  12126,
     -4027,   -632,  -1340,   5377,  -7545,   7469,   5714, -14756,
      6752,  -1091,   -765,      0,      0,   6749,   2899, -11311,
     -2732,  -1744,  -5232,  12699,   5747,   -408,     22,   3703,
      8125, -12112,  -9996,   1005,   1733,    314,    542,  -5610,
     -9577,   7576,  11947,    369,  -1894,  -1280,      0,      0,
      6696,  -7271,   8475,   1695,  -3426, -13096,   4257,   2872,
     -4744,   3001,  -4768, -11228,  -2095,   1758,  -9104,  -3165,
      4315,   6357,  -6034,  -7877,  -7431,  -1900, -11861,  -7342,
      -332,      0,      0,   6749,  -6967,   1987,   9556, -12333,
     -2973,  -6292,   3380,   3594, -13415,  -5226,   5136,  -4572,
     -7118,   -346,  -5001,  -6495,  -3558,   7744,   5564,   5516,
     -4343,  -2293,  -3292, -10859,      0,      0,   6600,    960,
      2367,  10040,   3042,    405,  -6421,   5603,  11590,   4996,
      2882,  -1497,  -4613,  -7224,   7031,  10875,   4417,   2386,
      -900,  -4349,   4047, -11028,  -7647,   8282,  10943,      0,
         0,   6472,  -3636,   1207, -10542,   8209,  -1431,  -6753,
     -1588,   9180, -10988,   2654,   4394,  -2573,   9660,   1956,
     -6197,  15069,  -6669,  -5129,   3160,   4924,   4007,  -7510,
     -2959,   1254,      0,      0,   6472,  -9198,  -3223,  -5553,
      9817,   5300,  -6431,   2900,  -5253,  -1448,  -6402,   5817,
      5919,   3196,   4714,  12036,  -8477,   3515,  -6047, -10220,
      1438,  -6376,   1857,  -8960,  -7950,      0,      0,   6600,
      9798,   3181,   1100,   3140,   7037,  -5713,    656, -11131,
    -10798,   2454,  -6385,  -4853,   -490,  -8042,  -4982,  -7526,
    -10601,   -357, -10140,   2162,  -2133,   2203,   -662,  11970,
         0,      0,   5961,  -1847,   8037,   6980,  -2705,  -2254,
      4459,  11243,   3903,  -2236,  -6468,  -2518,  -1328,  12187,
      9833,   2485,  -2637,  -6637,  -6912,    591,  -6885,   8936,
     13323,   5876,   1657,      0,      0,   6696,  -6909,  -8810,
     -1486,   2338,  11780,   7168,   3430,  -4973,   3029,  -4701,
    -11982,   -795,  -3695,   8125,   1733,    -47,  -8922,   7129,
     10160,  -3828,   4367, -10990,  -3956,   2684,      0,      0,
      6558,  -1533,   8783,  -7347,   1569,  -2963,   5023, -12941,
      4475,   -794,   3126,  -3565,  -1998, -13033,  11059,  -2765,
      2560,  -2895,   3088,  -2963,  -7558,  -6369,  12464,  -7738,
       392,      0,      0,   5961,   9227,    950,  -5539, -10243,
      1088,  -5939,  -1281,  -5559,   1206,  -2420,  -8987,  -2684,
      4140,   -992,  13890,  13708,    709,   9006,  -2807,   5834,
      2329,   5047,   1475,  -9831,      0,      0,   6600,   4995,
     -8043,   4206,   5650,  -9366,   5476,  -8133,   -768,   2981,
     -9895,   9151,   1863,   8527,   1086,  -2963,    579,  -7480,
     13362,  -7907,  -5688,  -5543,  -3906,   7310,   -803,      0,
         0,   6669,  -8160,  -5298,   5371,  -8746,   8732,  -2836,
     -6930,  -3130,  -2014,  12296,  -1275,   8864,    484,   5014,
     -7335,   3627,   3372,  -4224, -10770,  -5467,   4279,  -2324,
     11616,  -5058,      0,      0,   6749,   7150,   8836,  -3822,
     -5514,  11389,   4617,  -6025,  -2673,    456, -10193,  12469,
     -1985,  -5080,  -5211,   3725,   2028,     96, -11701,   6828,
     -7433,  -1949,  -6276,   6430,  -1283,      0,      0,   6694,
     10767,  -5523,   -179,  -1298, -10419,  -3002,    756,  -9941,
    -10209,   1253,    943,   7426,   -403,  11719,    494,  -1200,
     13519,  -2085,  10086,  -3095,    242,  -2256,  -1923,   7449,
         0,      0,   6534,   8629,  -1366,   8406,  13170,  -2141,
     -7411,  -2085,   -345,   9225,  -4758,  -7589,   1661,  -6770,
       189,  -9800,    291,  -1398,  -8992,   4759,   5375,   3803,
      2651,   2060, -13283,      0,      0,   6558,  -2337,  -6787,
     -9053,   4182,   3465,    808,  11354,   8057,  -5196,  -7130,
     -3613,   5148,  -6525, -11638,  -5613,   3255,  10056,   5908,
        60,  -5490,  -2230,   9172,   8712,   2134,      0,      0,
      6694,   6663,   7957,   6224,   6911,   9045,   3232,   8864,
      -233,   3655,  12601,   7652,  -4491,   5830,   -914,  -3728,
      1779,   8319,  10684,   1094,  -7130,    543,  -1143,  -9211,
     -6258,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,  32767,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
     32767
};
#endif

#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 5
/* Order-5 ambisonics: 36 ACN channels + 2 non-diegetic = 38 streams.
   maxDet (Fekete) nodes, degree 5 (36 points), orthogonalized: the mixing
   matrix has orthonormal columns, so cond = 1, no gain field is needed, and the
   demixing matrix is the exact transpose of the mixing matrix.
   Its demixing matrix is derived by transposing this table at init time. */
const MappingMatrix mapping_matrix_fifthoa_mixing = { 38, 38, 0 };
const opus_int16 mapping_matrix_fifthoa_mixing_data[1444] = {
      5618,    294,   9453,    145,   -329,    181,  12291,    359,
       327,      2,   -522,   -449,  14778,    865,    853,    290,
      1017,   -357,  -1077,   -283,  17340,    239,   1114,   -358,
      -434,   -649,    366,  -1587,   -292,    438,  16424,   -781,
       898,   -979,   -753,    211,      0,      0,   5234,     27,
     -2241,   8712,   -104,    223,  -4500,  -4453,   9020,    -10,
       401,    155,   4969,  -5467,  -5536,  10012,    315,    533,
      -611,   -138,   2968,   8376,  -5149,  -8624,  11674,     82,
       530,  -2549,   -726,  -1218,  -5538,   2717,   8291,  -3351,
     -9864,  10824,      0,      0,   5408,  -5166,  -1976,  -8088,
      9664,   2815,  -5569,   3665,   4152, -11184,  -6139,   3589,
      3882,   5848,  -2198,   2262,   8016,   6550,  -3534,  -3781,
      3005,  -5830,  -1998,  -1202,  -7141,  -4281,  -6980,   3164,
      7453,   -795,  -3871,  -4404,   1927,  -2238,   6147,  10152,
         0,      0,   5538,   5506,  -1727,   7792,   9873,  -2702,
     -5296,  -3684,   3715,  10691,  -5724,  -4193,   4104,  -5738,
     -1968,  -3097,   7073,  -6157,  -4004,   3182,   3185,   5853,
     -2014,   2344,  -9776,   1872,  -4773,  -3049,   5623,   2359,
     -4353,   3190,   3111,   -652,   7262, -11514,      0,      0,
      5638,   7192,   3640,  -4796,  -8522,   6540,  -3162,  -4367,
     -3576,   2361,  -8542,  -1120,  -5853,   1154,  -3615,   8898,
      6830,   2184,  -2231,  -6199,  -1032,   4433,   -546,  10231,
     -6122,  -9265,   8739,    656,   4092,  -3655,   3260,   2419,
      1773,   6020,  -9013,  -1104,      0,      0,   5638,    -71,
     -5960,  -7243,    320,    231,   1244,  10742,   6338,     40,
      -537,   -418,   5207,  -6623, -10543,  -5003,    581,   1384,
       433,   -883,  -7527,  -1831,  10473,   8356,   3906,     51,
     -2076,  -3312,    133,   1558,   3632,  11418,  -3534, -10626,
     -6354,  -1993,      0,      0,   5487,  -3659,   7522,  -4240,
      3415,  -6682,   5398,  -7931,    -80,  -1631,   7771,  -7458,
       694,  -9111,   1124,   1843,   -192,  -3433,  12113,  -5508,
     -4319,  -8321,   1877,   2662,   -264,   -407,   -897,  -8617,
     12309,   -406,  -5560,  -4328,   1468,   4602,  -3394,   1267,
         0,      0,   5508,  -5684,   7894,    932,   -672,  -9938,
      6120,   1964,  -3200,   1968,  -2047, -11102,   1975,   2303,
     -7812,  -1449,     71,   4165,  -4205,  -9285,  -3485,   3519,
    -10372,  -1633,    369,    778,   2444,   9445,  -5763,  -5556,
     -7058,   3551,  -9731,  -2968,   3349,   -504,      0,      0,
      5706,   9303,  -2072,  -2413,  -5420,  -4826,  -5482,    727,
     -9251,  -7362,   2815,  -6218,   4251,   1945,   6193,   8242,
     10282,   6999,   3786,   6723,   3102,  -1897,   3352,  -5276,
      4350,   1182,  -7741,    205,  -3659,   2668,  -4588,   -467,
     -6087,  -2999,  -4125,  -9750,      0,      0,   5134,   3326,
      2883,  -7705,  -6088,   1811,  -4132,  -5335,   6263,   7880,
     -5018,  -1863,  -5624,   4162,   5699,  -3249,  -9807,   8838,
      2884,  -3417,    -52,   7397,  -1631,  -4372,    569,   9197,
    -11822,  -3674,   7526,    990,   5543,   1698,  -6441,  -2060,
      1445,   3575,      0,      0,   5514,   6980,   3785,   5725,
      8875,   6721,  -3312,   4560,  -2304,   3925,   9003,  -1032,
     -6327,   -855,  -1718,  -7938,  -2362,   4557,    716,  -6666,
      -994,  -5136,  -1189,  -9974,  -7268,  -8694,  -5216,  -2196,
     -7507,  -5450,   4223,  -3529,   1623,  -2494,  -9759,  -1323,
         0,      0,   5156,   8695,   3064,   -278,    -70,   6287,
     -3538,    439,  -8423,  -9442,    209,  -3591,  -5304,   -163,
     -7858,   -477,   -970, -10262,   1021,  -8502,   -181,    510,
      2198,   -604,   8440,  10782,   -634,   1971,    555,  -3423,
      4678,   -582,   9322,  -3735,  11578,   -789,      0,      0,
      5662,   5758,  -2370,  -7795, -10060,  -2548,  -5138,   4385,
      2953,  10036,   5941,  -3857,   4802,   4488,  -2723,   4218,
     -5045,  -8895,   5072,   4336,   2894,  -5776,    205,  -2022,
     -9837,  -2591,   4970,  -1046,  -6978,    612,  -4906,  -1482,
      2361,  -3554,   7433,   9363,      0,      0,   5591,  -4491,
      1197,   8637,  -8561,  -1026,  -6089,   2053,   6559, -11084,
     -2352,   3546,  -2702,  -7391,   1699,   1814, -11313,  -4314,
      7592,   1871,   4488,  -3286,  -4750,   2231,  -3913,  -8620,
     -4650,   9191,   2812,  -2627,   2343,   4765,  -2889,  -1185,
      2129,  -7701,      0,      0,   5514,  -2747,  -9071,   2448,
     -1076,   5960,  10273,  -4338,    -16,   -357,   3441,  -8394,
     -8473,   6962,    968,   -786,   -354,    755,  -6520,  10617,
      5594,  -8005,   -852,     33,    601,   1192,     66,  -2530,
      6311, -13129,  -2624,   9806,   1694,  -1917,    -86,    417,
         0,      0,   5678,  -4350,  -5232,   6822,  -7442,   5317,
      -907,  -8529,   3140,  -7229,  10072,  -2002,   6026,   2655,
     -4461,  -1444,  -3813,  11600,  -5700,  -3712,  -5010,   3897,
      3004,   2012,  -5937,    507,   6695,  -7501,  -1324,   6186,
       515,  -6718,    737,  -1561,   7071,  -5895,      0,      0,
      5706,   5281,  -7387,  -3768,  -4346,  -9418,   4079,   6507,
     -1502,    535,   9480,   8875,   2208,  -6030,   3297,   2807,
       623,  -1479, -12019,  -3049,  -7516,   2063,  -4411,  -6955,
      -752,   -487,  -2717,   2559,   7403,  -3494,   6172,   2471,
      2215,  11066,   2912,   -205,      0,      0,   5487,  -7053,
      3035,  -5384,   9175,  -5648,  -3864,  -3393,  -2419,  -4009,
      7980,   2754,  -5391,   2331,  -2751,   8799,  -5160,  -2491,
     -3207,   7059,    623,   5634,  -1543,   8872,  -9016,  11319,
     -3738,   1798,  -8438,    878,   4498,   1598,   2385,    730,
     -8166,    -68,      0,      0,   5156,   -151,  -8627,  -3258,
      -118,    878,   9278,   5969,    834,   -114,   -547,   -505,
     -9953,  -9352,  -2262,   -101,    245,  -1395,   1475,   1585,
      8765,  12343,   4288,   1412,   -207,  -1435,    -52,   1855,
      -150,  -2251,  -4454, -17865,  -5802,  -2564,    259,    630,
         0,      0,   5508,  -9224,   3215,   -260,    534,  -6196,
     -4135,   -721,  -9720,  10010,    133,   3347,  -5909,     70,
     -7486,   1122,   -379,   8405,   1693,   8092,    381,   -177,
      2547,   1516,  10395, -12117,  -3582,  -1741,    496,   2257,
      5174,  -1028,   7065,     22,   9561,   -316,      0,      0,
      5529,  -5760,  -6853,  -3542,   5088,   8709,   3805,   5908,
     -1803,  -1415,  -9355,  -8602,   2247,  -5759,   3270,   3550,
     -1680,   3770,  10263,   3446,  -6290,   1958,  -5076,  -7358,
     -2593,   2508,   3928,  -2550,  -7970,   2816,   4826,   4141,
      3929,  11675,   3584,  -1436,      0,      0,   5624,  -2409,
      3949,  -8344,   4583,  -1696,  -3077,  -7655,   7867,  -5883,
      4604,    451,  -6033,   1630,   8258,  -6172,   8169,  -8199,
      -298,   1723,  -1386,   8337,   2649,  -7556,   3589,  -8972,
      9466,  -1672,  -4092,    950,   3751,   6525,  -5554,  -3361,
      4551,   -591,      0,      0,   5134,    150,  -1314,  -8775,
      -624,   -406,  -5373,   2268,   9515,    795,    870,     97,
      2448,   7040,  -3047, -10794,  -1110,   -266,   -469,   1007,
      5168,  -4300,  -9069,   4525,  10983,   2195,   2707,   1308,
     -1908,  -1012,  -3712,  -7612,   5564,  10396,  -5904, -11203,
         0,      0,   5502,   7320,  -6215,   2656,   4334,  -9431,
      1082,  -3628,  -4818,  -2603,  -7357,   6492,   5279,   2445,
      8493,  -4874,  -3631,   3137,   6591,   2325,  -6937,   1302,
     -7007,   7512,    427,   -366,   6305,  -4251,  -1903, -11053,
      3170,  -4963,   3153, -10090,   -133,   2915,      0,      0,
      5538,   1416,  -7131,   6415,   2630,  -2727,   3345, -10706,
      4476,   2659,  -4979,   2665,   1929,   9928,  -8957,   2528,
      1654,  -6515,   5316,    -41,  -6944,  -4425,  10164,  -4155,
      1355,    -31,  -4374,   8496,  -2387,  -1189,   6157,  -5356,
     -8170,   7845,  -1847,    756,      0,      0,   5191,   2747,
     -7930,   1267,    608,  -5993,   9331,  -2610,   -866,    222,
     -1891,   8092,  -9235,   3663,   1492,    102,    445,   1652,
      3883, -12332,   8243,  -4955,  -3165,    894,   -624,    175,
       229,  -1740,  -5018,  15938,  -4873,   8458,   4341,  -1615,
      -138,   -920,      0,      0,   5678,   1920,   3055,   8934,
      3907,   1147,  -4514,   6814,   9149,   5507,   3258,   -726,
     -6420,  -3981,   7684,   8092,   6385,   5081,  -2008,  -1331,
       -59, -10247,  -1001,   8292,   6068,   7380,   7094,    583,
     -2081,    765,   5729,  -3212,  -8297,   1036,   6045,   3853,
         0,      0,   5413,  -8821,  -1073,   4464,  -7981,   2325,
     -5842,   -785,  -6488,   1876,   2268,   8193,   3114,  -3752,
      2141, -11214,  10095,    -20,   5618,  -4081,   5559,   2713,
      4675,   3292,  -2442,   7058,  -3142,  -2407,  -5565,  -8893,
     -4516,   5222,  -2442,   8151,     34,   6857,      0,      0,
      5404,   2597,   7477,   4455,   2835,   4268,   5777,   8795,
      1892,   2258,   7028,   4956,   1467,  10016,   3370,   -263,
       536,   4818,  10093,   3976,  -4018,   9103,   5911,    288,
     -1226,   2384,   4883,  11605,   8927,   1670,  -6920,   3314,
      6588,    -36,    -42,  -1668,      0,      0,   5191,   7857,
     -1177,   2988,   6264,  -2527,  -5544,   -679,  -7355,  -4774,
     -1510,  -7210,   2271,  -2700,   1534,  -8843, -11331,   1839,
     -6466,   4023,   4865,    696,   6331,   3728,   2423,  -3112,
      5312,   4179,   3192,   8397,  -3150,   3369,  -5219,   9436,
     -2136,  11034,      0,      0,   5408,   2177,   7729,  -5615,
     -2669,   4145,   5904,  -9708,   3098,   2099,  -5889,   4933,
      1129, -11680,   5563,  -1668,  -1897,   4445,  -8325,   3754,
     -4329,  -7387,   7109,  -1642,   -266,   2423,  -1738,  10162,
     -8780,   -790,  -7117,  -1950,   7952,  -2636,   1184,  -1133,
         0,      0,   5529,   5435,   7944,    229,    338,  10212,
      5887,   -264,  -3903,  -2255,   -239,  12438,   1124,   -315,
     -7987,    163,   -526,  -4162,   -494,   9636,  -4288,  -1314,
    -11379,    313,   1972,   -285,  -2008,  -9167,    233,   4788,
     -5913,    411, -12354,   1356,   2030,   1391,      0,      0,
      5404,  -7178,  -5442,   1162,  -2764,   9414,    272,  -1959,
     -6355,   5186,   4193,  -5753,   4135,   1196,   9445,  -2761,
      3621,  -9439,  -2543,  -1933,  -6155,    629,  -8237,   6667,
      3705,  -3637,  -7139,   8575,   2089,   9811,   3311,  -4217,
       981,  -6886,  -4703,   3763,      0,      0,   5618,  -8682,
     -2410,  -2878,   5711,   4695,  -5513,   1582,  -7994,   6445,
     -3184,   7061,   5023,   2067,   4954,   8052,  -9875,  -4509,
     -4273,  -7725,   3601,  -2688,   6086,  -6522,   3027,   1688,
      8309,  -2511,   4522,  -3703,  -5211,   -334,  -6092,  -3877,
     -2468,  -8844,      0,      0,   5037,  -5835,   3386,   4035,
     -7570,  -6388,  -2624,   4308,  -2712,  -1713,  -7928,    110,
     -6109,   -391,  -3512,  -7165,   5211,  -2033,  -2661,   6825,
     -2352,  -5311,   -697, -10823,  -5924,   8702,   9026,   -678,
      7465,   7373,   4541,  -5092,   2540,  -5140,  -8313,    987,
         0,      0,   5413,  -2713,   7000,   6521,  -3718,  -3776,
      3499,   9654,   3525,  -3885,  -7621,  -3425,  -1986,   8817,
      7739,   1847,  -2751,  -7074,  -7658,  -1786,  -5512,   2887,
      7539,   3165,   1418,  -4232,  -8344, -12393,  -5655,   -263,
     -4840,  -2712,   4483,   4236,   -571,    569,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,  32767,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,      0,      0,      0,      0,      0,
         0,      0,      0,  32767
};
#endif


/* Built-in projection mixing matrices by ambisonic order, so that callers do not need
 * a branch per order. Index 0 is unused: order 0 (mono) has no projection matrix.
 */
typedef struct {
  const MappingMatrix *matrix;
  const opus_int16 *data;
  opus_int32 data_size;
} AmbisonicMatrix;

#define MAPPING_MATRIX_ENTRY(stem) \
  { &mapping_matrix_ ## stem ## _mixing, mapping_matrix_ ## stem ## _mixing_data, \
    sizeof(mapping_matrix_ ## stem ## _mixing_data) }

static const AmbisonicMatrix ambisonic_mixing_matrices[
    MAPPING_MATRIX_MAX_AMBISONIC_ORDER + 1] = {
  { NULL, NULL, 0 },
  MAPPING_MATRIX_ENTRY(foa),
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 2
  MAPPING_MATRIX_ENTRY(soa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 3
  MAPPING_MATRIX_ENTRY(toa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 4
  MAPPING_MATRIX_ENTRY(fourthoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 5
  MAPPING_MATRIX_ENTRY(fifthoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 6
  MAPPING_MATRIX_ENTRY(sixthoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 7
  MAPPING_MATRIX_ENTRY(seventhoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 8
  MAPPING_MATRIX_ENTRY(eighthoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 9
  MAPPING_MATRIX_ENTRY(ninthoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 10
  MAPPING_MATRIX_ENTRY(tenthoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 11
  MAPPING_MATRIX_ENTRY(eleventhoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 12
  MAPPING_MATRIX_ENTRY(twelfthoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 13
  MAPPING_MATRIX_ENTRY(thirteenthoa),
#endif
#if MAPPING_MATRIX_MAX_AMBISONIC_ORDER >= 14
  MAPPING_MATRIX_ENTRY(fourteenthoa)
#endif
};

int mapping_matrix_get_ambisonic(int order, const MappingMatrix **matrix,
  const opus_int16 **data, opus_int32 *data_size)
{
  const AmbisonicMatrix *entry;

  if (order < 1 || order > MAPPING_MATRIX_MAX_AMBISONIC_ORDER)
    return OPUS_BAD_ARG;

  entry = &ambisonic_mixing_matrices[order];
  if (matrix)
    *matrix = entry->matrix;
  if (data)
    *data = entry->data;
  if (data_size)
    *data_size = entry->data_size;
  return OPUS_OK;
}
