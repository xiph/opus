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

/**
 * @file mapping_matrix.h
 * @brief Opus reference implementation mapping matrix API
 */

#ifndef MAPPING_MATRIX_H
#define MAPPING_MATRIX_H

#include "opus_types.h"
#include "opus_projection.h"

typedef struct MappingMatrix
{
    int rows; /* number of channels outputted from matrix. */
    int cols; /* number of channels inputted to matrix. */
    int gain; /* in dB. S7.8-format. */
    /* Matrix cell data goes here using col-wise ordering. */
} MappingMatrix;

opus_int32 mapping_matrix_get_size(int rows, int cols);

opus_int16 *mapping_matrix_get_data(const MappingMatrix *matrix);

void mapping_matrix_init(
    MappingMatrix * const matrix,
    int rows,
    int cols,
    int gain,
    const opus_int16 *data,
    opus_int32 data_size
);

/* As mapping_matrix_init(), but initializes `matrix` to the transpose of `data`, a
 * `cols` x `rows` matrix.
 */
void mapping_matrix_init_transposed(
    MappingMatrix * const matrix,
    int rows,
    int cols,
    int gain,
    const opus_int16 *data,
    opus_int32 data_size
);

#ifndef DISABLE_FLOAT_API
void mapping_matrix_multiply_channel_in_float(
    const MappingMatrix *matrix,
    const float *input,
    int input_rows,
    opus_res *output,
    int output_row,
    int output_rows,
    int frame_size
);

void mapping_matrix_multiply_channel_out_float(
    const MappingMatrix *matrix,
    const opus_res *input,
    int input_row,
    int input_rows,
    float *output,
    int output_rows,
    int frame_size
);
#endif /* DISABLE_FLOAT_API */

void mapping_matrix_multiply_channel_in_short(
    const MappingMatrix *matrix,
    const opus_int16 *input,
    int input_rows,
    opus_res *output,
    int output_row,
    int output_rows,
    int frame_size
);

void mapping_matrix_multiply_channel_out_short(
    const MappingMatrix *matrix,
    const opus_res *input,
    int input_row,
    int input_rows,
    opus_int16 *output,
    int output_rows,
    int frame_size
);


void mapping_matrix_multiply_channel_in_int24(
    const MappingMatrix *matrix,
    const opus_int32 *input,
    int input_rows,
    opus_res *output,
    int output_row,
    int output_rows,
    int frame_size
);

void mapping_matrix_multiply_channel_out_int24(
    const MappingMatrix *matrix,
    const opus_res *input,
    int input_row,
    int input_rows,
    opus_int32 *output,
    int output_rows,
    int frame_size
);
/* Highest ambisonic order with a built-in projection matrix, 1 to 12, set with
 * --enable-ambisonics-max-order (autotools), OPUS_AMBISONICS_MAX_ORDER (CMake) or
 * ambisonics-max-order (meson). This limits only the encoder: the decoder takes its
 * demixing matrix from the stream.
 */
#ifndef OPUS_AMBISONICS_MAX_ORDER
# define OPUS_AMBISONICS_MAX_ORDER 5
#endif
#if OPUS_AMBISONICS_MAX_ORDER < 1 || OPUS_AMBISONICS_MAX_ORDER > 12
# error "OPUS_AMBISONICS_MAX_ORDER must be between 1 and 12."
#endif
#define MAPPING_MATRIX_MAX_AMBISONIC_ORDER OPUS_AMBISONICS_MAX_ORDER

/* Get the built-in projection mixing matrix for ambisonic `order`; the demixing
 * matrix is its transpose. Any output may be NULL. Returns OPUS_BAD_ARG, leaving the
 * outputs untouched, if there is no matrix for that order.
 */
int mapping_matrix_get_ambisonic(
    int order,
    const MappingMatrix **matrix,
    const opus_int16 **data,
    opus_int32 *data_size
);

/* Pre-computed projection mixing matrices for 1st- to 5th-order (mapping_matrix.c)
 * and 6th- to 12th-order ambisonics (mapping_matrix_hoa.c). Only orders up to
 * MAPPING_MATRIX_MAX_AMBISONIC_ORDER are compiled in.
 *   foa: first-order ambisonics
 *   soa: second-order ambisonics
 *   toa: third-order ambisonics
 */
extern const MappingMatrix mapping_matrix_foa_mixing;
extern const opus_int16 mapping_matrix_foa_mixing_data[36];

extern const MappingMatrix mapping_matrix_soa_mixing;
extern const opus_int16 mapping_matrix_soa_mixing_data[121];

extern const MappingMatrix mapping_matrix_toa_mixing;
extern const opus_int16 mapping_matrix_toa_mixing_data[324];

extern const MappingMatrix mapping_matrix_fourthoa_mixing;
extern const opus_int16 mapping_matrix_fourthoa_mixing_data[729];

extern const MappingMatrix mapping_matrix_fifthoa_mixing;
extern const opus_int16 mapping_matrix_fifthoa_mixing_data[1444];

extern const MappingMatrix mapping_matrix_sixthoa_mixing;
extern const opus_int16 mapping_matrix_sixthoa_mixing_data[2601];

extern const MappingMatrix mapping_matrix_seventhoa_mixing;
extern const opus_int16 mapping_matrix_seventhoa_mixing_data[4356];

extern const MappingMatrix mapping_matrix_eighthoa_mixing;
extern const opus_int16 mapping_matrix_eighthoa_mixing_data[6889];

extern const MappingMatrix mapping_matrix_ninthoa_mixing;
extern const opus_int16 mapping_matrix_ninthoa_mixing_data[10404];

extern const MappingMatrix mapping_matrix_tenthoa_mixing;
extern const opus_int16 mapping_matrix_tenthoa_mixing_data[15129];

extern const MappingMatrix mapping_matrix_eleventhoa_mixing;
extern const opus_int16 mapping_matrix_eleventhoa_mixing_data[21316];

extern const MappingMatrix mapping_matrix_twelfthoa_mixing;
extern const opus_int16 mapping_matrix_twelfthoa_mixing_data[29241];

#endif /* MAPPING_MATRIX_H */
