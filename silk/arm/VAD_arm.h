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

#ifndef SILK_VAD_ARM_H
#define SILK_VAD_ARM_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)

/* When NEON is enabled, silk_VAD_GetNoiseLevels is exported (rather than a
   static inline in VAD.c) so the NEON silk_VAD_GetSA_Q8 can call it, mirroring
   the x86 path. */
void silk_VAD_GetNoiseLevels(
    const opus_int32            pX[ VAD_N_BANDS ],
    silk_VAD_state              *psSilk_VAD
);

opus_int silk_VAD_GetSA_Q8_neon(
    silk_encoder_state          *psEncC,
    const opus_int16            pIn[]
);

#if defined(OPUS_ARM_PRESUME_NEON_INTR)

#define OVERRIDE_silk_VAD_GetSA_Q8
#define silk_VAD_GetSA_Q8(psEnC, pIn, arch) \
    ((void)(arch), silk_VAD_GetSA_Q8_neon(psEnC, pIn))

#elif defined(OPUS_HAVE_RTCD)

#define OVERRIDE_silk_VAD_GetSA_Q8
extern opus_int (*const SILK_VAD_GETSA_Q8_IMPL[OPUS_ARCHMASK + 1])(
    silk_encoder_state          *psEncC,
    const opus_int16            pIn[]
);
#define silk_VAD_GetSA_Q8(psEnC, pIn, arch) \
    ((*SILK_VAD_GETSA_Q8_IMPL[(arch) & OPUS_ARCHMASK])(psEnC, pIn))

#endif

#endif /* OPUS_ARM_MAY_HAVE_NEON_INTR */

#endif /* SILK_VAD_ARM_H */
