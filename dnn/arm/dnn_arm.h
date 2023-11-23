/* Copyright (c) 2011-2019 Mozilla
                 2023 Amazon */
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

#ifndef DNN_ARM_H
#define DNN_ARM_H

#include "cpu_support.h"
#include "opus_types.h"

void compute_linear_dotprod(const LinearLayer *linear, float *out, const float *in);
void compute_linear_neon(const LinearLayer *linear, float *out, const float *in);


#if defined(OPUS_ARM_PRESUME_DOTPROD)

#define OVERRIDE_COMPUTE_LINEAR
#define compute_linear(linear, out, in, arch) ((void)(arch),compute_linear_dotprod(linear, out, in))

#elif defined(OPUS_ARM_PRESUME_NEON_INTR) && !defined(OPUS_ARM_MAY_HAVE_DOTPROD)

#define OVERRIDE_COMPUTE_LINEAR
#define compute_linear(linear, out, in, arch) ((void)(arch),compute_linear_neon(linear, out, in))

#elif defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_DOTPROD) || defined(OPUS_ARM_MAY_HAVE_NEON))

extern void (*const DNN_COMPUTE_LINEAR_IMPL[OPUS_ARCHMASK + 1])(
                    const LinearLayer *linear,
                    float *out,
                    const float *in
                    );
#define OVERRIDE_COMPUTE_LINEAR
#define compute_linear(linear, out, in, arch) \
    ((*DNN_COMPUTE_LINEAR_IMPL[(arch) & OPUS_ARCHMASK])(linear, out, in))


#endif



#endif /* DNN_ARM_H */
