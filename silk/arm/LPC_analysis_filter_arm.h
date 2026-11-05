/***********************************************************************
Copyright (c) 2026 LPC NEON Pilot
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

#if !defined(SILK_LPC_ANALYSIS_FILTER_ARM_H)
# define SILK_LPC_ANALYSIS_FILTER_ARM_H

# include "celt/cpu_support.h"

/* NEON implementation (only). */
extern void silk_LPC_analysis_filter_neon(
        opus_int16                  *out,               /* O    Output signal                                               */
        const opus_int16            *in,                /* I    Input signal                                                */
        const opus_int16            *B,                 /* I    MA prediction coefficients, Q12 [order]                     */
        const opus_int32            len,                /* I    Signal length                                               */
        const opus_int32            d,                  /* I    Filter order                                                */
        int                         arch                /* I    Run-time architecture                                       */
);

/* NEON-presumed build (aarch64): route LPC_analysis_filter straight to NEON. */
# define silk_LPC_analysis_filter(out, in, B, len, d, arch) \
    silk_LPC_analysis_filter_neon(out, in, B, len, d, (arch))

#endif /* SILK_LPC_ANALYSIS_FILTER_ARM_H */
