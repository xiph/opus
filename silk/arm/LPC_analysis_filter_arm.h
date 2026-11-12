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

#ifndef SILK_LPC_ANALYSIS_FILTER_ARM_H
# define SILK_LPC_ANALYSIS_FILTER_ARM_H

# include "celt/arm/armcpu.h"

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void silk_LPC_analysis_filter_neon(     /* O    output signal                                        */
    opus_int16                  *out,   /* O    Output signal                                         */
    const opus_int16            *in,    /* I    Input signal                                          */
    const opus_int16            *B,     /* I    MA prediction coefficients, Q12 [order]               */
    const opus_int32            len,    /* I    Signal length                                         */
    const opus_int32            d,      /* I    Filter order                                          */
    int                         arch    /* I    Run-time architecture                                 */
);

#  if !defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_silk_LPC_analysis_filter            (1)
#   define silk_LPC_analysis_filter(out,in,B,len,d,arch) ((void)(arch), PRESUME_NEON(silk_LPC_analysis_filter)(out,in,B,len,d,arch))
#  endif
# endif

# if !defined(OVERRIDE_silk_LPC_analysis_filter)
/*Is run-time CPU detection enabled on this platform?*/
#  if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern void (*const SILK_LPC_ANALYSIS_FILTER_IMPL[OPUS_ARCHMASK+1])(
        opus_int16                  *out,   /* O    Output signal                                         */
        const opus_int16            *in,    /* I    Input signal                                          */
        const opus_int16            *B,     /* I    MA prediction coefficients, Q12 [order]               */
        const opus_int32            len,    /* I    Signal length                                         */
        const opus_int32            d,      /* I    Filter order                                          */
        int                         arch    /* I    Run-time architecture                                 */
);
#   define OVERRIDE_silk_LPC_analysis_filter            (1)
#   define silk_LPC_analysis_filter(out,in,B,len,d,arch) ((*SILK_LPC_ANALYSIS_FILTER_IMPL[(arch)&OPUS_ARCHMASK])(out,in,B,len,d,arch))
#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_silk_LPC_analysis_filter            (1)
#   define silk_LPC_analysis_filter(out,in,B,len,d,arch) ((void)(arch), silk_LPC_analysis_filter_neon(out,in,B,len,d,arch))
#  endif
# endif

#endif /* SILK_LPC_ANALYSIS_FILTER_ARM_H */
