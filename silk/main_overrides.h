/* Copyright (c) 2025, Samsung R&D Institute Poland
   Written by Marek Pikula

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

#ifndef SILK_MAIN_OVERRIDES_H
#define SILK_MAIN_OVERRIDES_H

/** Override implementation with a single function. */
#define OVERRIDE_IMPL_SINGLE(func, impl, arch, ...) \
    ((void)(arch), func ## _ ## impl(__VA_ARGS__))

/** Override implementation with a multiple arch-specific functions.
 *
 * It uses an array declared with OVERRIDE_IMPL_ARRAY_DECL().
 */
#define OVERRIDE_IMPL_ARRAY(func, arch, ...) \
    ((*OVERRIDE_MAP_ ## func[(arch) & OPUS_ARCHMASK])(__VA_ARGS__))

/** Declare a mapping array for use with OVERRIDE_IMPL_ARRAY(). */
#define OVERRIDE_IMPL_ARRAY_DECL(func) \
    const func ## _t OVERRIDE_MAP_ ## func[ OPUS_ARCHMASK + 1 ]


// Function declarations.

/* Entropy constrained matrix-weighted VQ, for a single input data vector */
#define SILK_VQ_WMAT_EC_DECL(impl, ...) \
    void (__VA_ARGS__ silk_VQ_WMat_EC_ ## impl)( \
        opus_int8           *ind,           /* O    index of best codebook vector               */ \
        opus_int32          *res_nrg_Q15,   /* O    best residual energy                        */ \
        opus_int32          *rate_dist_Q8,  /* O    best total bitrate                          */ \
        opus_int            *gain_Q7,       /* O    sum of absolute LTP coefficients            */ \
        const opus_int32    *XX_Q17,        /* I    correlation matrix                          */ \
        const opus_int32    *xX_Q17,        /* I    correlation vector                          */ \
        const opus_int8     *cb_Q7,         /* I    codebook                                    */ \
        const opus_uint8    *cb_gain_Q7,    /* I    codebook effective gain                     */ \
        const opus_uint8    *cl_Q5,         /* I    code length for each codebook vector        */ \
        const opus_int      subfr_len,      /* I    number of samples per subframe              */ \
        const opus_int32    max_gain_Q7,    /* I    maximum sum of absolute LTP coefficients    */ \
        const opus_int      L               /* I    number of vectors in codebook               */ \
    )
typedef SILK_VQ_WMAT_EC_DECL(t, *const);

/* Noise shaping quantization (NSQ) */
#define SILK_NSQ_DECL(impl, ...) \
    void (__VA_ARGS__ silk_NSQ_ ## impl)( \
        const silk_encoder_state    *psEncC,                                        /* I    Encoder State               */ \
        silk_nsq_state              *NSQ,                                           /* I/O  NSQ state                   */ \
        SideInfoIndices             *psIndices,                                     /* I/O  Quantization Indices        */ \
        const opus_int16            x16[],                                          /* I    Input                       */ \
        opus_int8                   pulses[],                                       /* O    Quantized pulse signal      */ \
        const opus_int16            *PredCoef_Q12,                                  /* I    Short term prediction coefs */ \
        const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],        /* I    Long term prediction coefs  */ \
        const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ],   /* I    Noise shaping coefs         */ \
        const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],              /* I    Long term shaping coefs     */ \
        const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                       /* I    Spectral tilt               */ \
        const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                     /* I    Low frequency shaping coefs */ \
        const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                      /* I    Quantization step sizes     */ \
        const opus_int              pitchL[ MAX_NB_SUBFR ],                         /* I    Pitch lags                  */ \
        const opus_int              Lambda_Q10,                                     /* I    Rate/distortion tradeoff    */ \
        const opus_int              LTP_scale_Q14                                   /* I    LTP state scaling           */ \
    )
typedef SILK_NSQ_DECL(t, *const);

#endif // SILK_MAIN_OVERRIDES_H
