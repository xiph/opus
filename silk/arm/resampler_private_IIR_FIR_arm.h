/***********************************************************************
Copyright (c) 2026, Xiph.Org Foundation. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifndef SILK_RESAMPLER_PRIVATE_IIR_FIR_ARM_H
#define SILK_RESAMPLER_PRIVATE_IIR_FIR_ARM_H

# include "celt/arm/armcpu.h"

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
opus_int16 *silk_resampler_private_IIR_FIR_INTERPOL_neon(
    opus_int16                  *out,               /* O    Output signal                                               */
    opus_int16                  *buf,               /* I    Buffer of 2x upsampled input                                */
    opus_int32                  max_index_Q16,      /* I    Interpolation end point, Q16                                */
    opus_int32                  index_increment_Q16 /* I    Interpolation step, Q16                                     */
);

#  if !defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_silk_resampler_private_IIR_FIR_INTERPOL (1)
#   define silk_resampler_private_IIR_FIR_INTERPOL(out, buf, max_index_Q16, index_increment_Q16, arch) \
    ((void)(arch), PRESUME_NEON(silk_resampler_private_IIR_FIR_INTERPOL)(out, buf, max_index_Q16, index_increment_Q16))
#  endif
# endif

# if !defined(OVERRIDE_silk_resampler_private_IIR_FIR_INTERPOL)
/*Is run-time CPU detection enabled on this platform?*/
#  if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern opus_int16 *(*const SILK_RESAMPLER_PRIVATE_IIR_FIR_INTERPOL_IMPL[OPUS_ARCHMASK+1])(
    opus_int16                  *out,
    opus_int16                  *buf,
    opus_int32                  max_index_Q16,
    opus_int32                  index_increment_Q16);
#   define OVERRIDE_silk_resampler_private_IIR_FIR_INTERPOL (1)
#   define silk_resampler_private_IIR_FIR_INTERPOL(out, buf, max_index_Q16, index_increment_Q16, arch) \
    ((*SILK_RESAMPLER_PRIVATE_IIR_FIR_INTERPOL_IMPL[(arch)&OPUS_ARCHMASK])(out, buf, max_index_Q16, index_increment_Q16))
#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_silk_resampler_private_IIR_FIR_INTERPOL (1)
#   define silk_resampler_private_IIR_FIR_INTERPOL(out, buf, max_index_Q16, index_increment_Q16, arch) \
    ((void)(arch), silk_resampler_private_IIR_FIR_INTERPOL_neon(out, buf, max_index_Q16, index_increment_Q16))
#  endif
# endif

#endif /* SILK_RESAMPLER_PRIVATE_IIR_FIR_ARM_H */
