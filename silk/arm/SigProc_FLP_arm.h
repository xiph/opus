/* Copyright (c) 2026 Xiph.Org Foundation */
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

#ifndef SILK_SIGPROC_FLP_ARM_H
#define SILK_SIGPROC_FLP_ARM_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef FIXED_POINT

#if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)

double silk_inner_product_FLP_neon(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
);

#if defined(OPUS_ARM_PRESUME_NEON_INTR)

#define OVERRIDE_inner_product_FLP
#define silk_inner_product_FLP(data1, data2, dataSize, arch) \
    ((void)arch, silk_inner_product_FLP_neon(data1, data2, dataSize))

#elif defined(OPUS_HAVE_RTCD)

#define OVERRIDE_inner_product_FLP
extern double (*const SILK_INNER_PRODUCT_FLP_IMPL[OPUS_ARCHMASK + 1])(
    const silk_float    *data1,
    const silk_float    *data2,
    opus_int            dataSize
);
#define silk_inner_product_FLP(data1, data2, dataSize, arch) \
    ((void)arch, (*SILK_INNER_PRODUCT_FLP_IMPL[(arch) & OPUS_ARCHMASK])(data1, data2, dataSize))

#endif

void silk_warped_autocorrelation_FLP_neon(
          silk_float    *corr,
    const silk_float    *input,
    const silk_float    warping,
    const opus_int      length,
    const opus_int      order
);

/* silk_warped_autocorrelation_FLP has no arch argument at its call site, so it
   cannot be dispatched through an arch-indexed RTCD table.  We therefore only
   override it on PRESUME-NEON targets (e.g. aarch64, where NEON is baseline);
   ARMv7 runtime-detection builds keep the C reference.  Adding an arch
   parameter to the signature would enable RTCD here too. */
#if defined(OPUS_ARM_PRESUME_NEON_INTR)

#define OVERRIDE_warped_autocorrelation_FLP
#define silk_warped_autocorrelation_FLP(corr, input, warping, length, order) \
    silk_warped_autocorrelation_FLP_neon(corr, input, warping, length, order)

#endif

/* silk_energy_FLP also takes no arch argument -> PRESUME only. */
double silk_energy_FLP_neon(
    const silk_float    *data,
    opus_int            dataSize
);

#if defined(OPUS_ARM_PRESUME_NEON_INTR)

#define OVERRIDE_energy_FLP
#define silk_energy_FLP(data, dataSize) silk_energy_FLP_neon(data, dataSize)

#endif

#endif /* OPUS_ARM_MAY_HAVE_NEON_INTR */

#endif /* !FIXED_POINT */

#endif /* SILK_SIGPROC_FLP_ARM_H */
