/* Copyright (c) 2026 Lynne */
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

/*
 * Pre-baked lookup tables for the AArch64 NEON inverse MDCT in
 * celt_tx_neon.S / celt_mdct_tx.c, for the CELT inverse MDCT sizes
 * N/2 = {120, 240, 480, 960, 1920} (the 48 kHz mode's four shifts plus the
 * 96 kHz/qext mode's larger transform; both modes share the middle sizes)
 * and the power-of-two sizes N/2 = {64, 128, 256, 512, 1024} used by Opus
 * custom modes (sub-FFT = fft32/fft_sr directly, no PFA, so those have no
 * pfa/p2 tables and need no scratch buffer).
 *
 * The tables are a bit-exact dump of what FFmpeg's libavutil/tx generates at
 * runtime for  av_tx_init(AV_TX_FLOAT_MDCT, inv=1, len=N/2, scale=-1.0f)  on
 * AArch64 (FFmpeg commit d229b4c1242870a8b80942f149a4b81bd25b7c5e):
 *
 *   celt_tx_mdct_map_<L>  L ints     first L/2: input gather map for the
 *                                    pre-rotation, composed with the PFA/
 *                                    fft15 input permutation and pre-doubled;
 *                                    second L/2: its inverse, used by the
 *                                    stride==4 contiguous path to scatter
 *   celt_tx_mdct_exp_<L>  L/2 cx     pre/post-rotation twiddles in natural
 *                                    index order, sqrt(|scale|) folded in;
 *                                    interleaved re/im floats. FFmpeg keeps a
 *                                    second copy pre-permuted by the gather
 *                                    map; the values are identical, so the
 *                                    asm indexes the one table via the map
 *   celt_tx_pfa_map_<L/2> L/2 ints   PFA compound out map (Ruritanian/CRT);
 *                                    the in map FFmpeg generates alongside
 *                                    it is unread by the preshuffled 15xM
 *                                    kernel (it equals the gather map >> 1)
 *   celt_tx_p2_map_<M>    M ints     split-radix parity scatter map of the
 *                                    M-point sub-FFT, read by the PFA kernel
 *
 * The N/2 = 1920 tables (96 kHz/qext mode) are gated behind ENABLE_QEXT,
 * and the power-of-two tables behind CUSTOM_MODES; neither size family is
 * reachable in builds without the matching option.
 *
 * scale=-1.0f makes the FFmpeg transform numerically identical to libopus's
 * clt_mdct_backward_c() (verified to ~2e-7 max relative error across all
 * sizes, strides 1/2/4/8 and overlaps); FFmpeg's own CELT decoder uses
 * -1.0f/32768 only because its internal signal scale is 1/32768 of ours.
 */

/* This file is grid-generated. Do not edit. */
#ifndef CELT_TX_TABLES_H
#define CELT_TX_TABLES_H

/* Define OPUS_ARM_TX_MDCT helper if we are building for ARM Neon float */
#if !defined(FIXED_POINT) && defined(__aarch64__) && \
    (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) || defined(OPUS_ARM_PRESUME_NEON_INTR))
#define OPUS_ARM_TX_MDCT (1)
#endif

/* Tables are needed if either C PFA is enabled or ARM Neon TX MDCT is enabled */
#if defined(OPUS_USE_PFA_MDCT) || defined(OPUS_ARM_TX_MDCT)
#define NEED_CELT_TX_TABLES (1)
#endif

#if defined(NEED_CELT_TX_TABLES)

#include "opus_types.h"
#include "kiss_fft.h"

extern const opus_int16 celt_tx_mdct_map_120[120];
extern const float celt_tx_mdct_exp_120[120];
extern const opus_int16 celt_tx_pfa_map_60[60];
extern const opus_int16 celt_tx_p2_map_4[4];
extern const opus_int16 celt_tx_mdct_map_240[240];
extern const float celt_tx_mdct_exp_240[240];
extern const opus_int16 celt_tx_pfa_map_120[120];
extern const opus_int16 celt_tx_p2_map_8[8];
extern const opus_int16 celt_tx_mdct_map_480[480];
extern const float celt_tx_mdct_exp_480[480];
extern const opus_int16 celt_tx_pfa_map_240[240];
extern const opus_int16 celt_tx_p2_map_16[16];
extern const opus_int16 celt_tx_mdct_map_960[960];
extern const float celt_tx_mdct_exp_960[960];
extern const opus_int16 celt_tx_pfa_map_480[480];
extern const opus_int16 celt_tx_p2_map_32[32];
extern const opus_int16 celt_tx_mdct_map_1920[1920];
extern const float celt_tx_mdct_exp_1920[1920];
extern const opus_int16 celt_tx_pfa_map_960[960];
extern const opus_int16 celt_tx_p2_map_64[64];
extern const opus_int16 celt_tx_mdct_map_64[64];
extern const float celt_tx_mdct_exp_64[64];
extern const opus_int16 celt_tx_mdct_map_128[128];
extern const float celt_tx_mdct_exp_128[128];
extern const opus_int16 celt_tx_mdct_map_256[256];
extern const float celt_tx_mdct_exp_256[256];
extern const opus_int16 celt_tx_mdct_map_512[512];
extern const float celt_tx_mdct_exp_512[512];
extern const opus_int16 celt_tx_mdct_map_1024[1024];
extern const float celt_tx_mdct_exp_1024[1024];

/* FFT Twiddle Tables from assembly */

extern const float celt_tx_tab_53_float[12];
extern const float celt_tx_tab_32_float[9];
extern const float celt_tx_tab_64_float[17];
extern const float celt_tx_tab_128_float[33];
extern const float celt_tx_tab_256_float[65];
extern const float celt_tx_tab_512_float[129];

#endif /* NEED_CELT_TX_TABLES */

#endif /* CELT_TX_TABLES_H */