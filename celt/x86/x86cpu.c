/* Copyright (c) 2014, Cisco Systems, INC
   Written by XiangMingZhu WeiZhou MinPeng YanWang

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

#include "cpu_support.h"
#include "macros.h"
#include "main.h"
#include "pitch.h"
#include "x86cpu.h"

#if defined(OPUS_HAVE_RTCD) && \
  ((defined(OPUS_X86_MAY_HAVE_SSE) && !defined(OPUS_X86_PRESUME_SSE)) || \
  (defined(OPUS_X86_MAY_HAVE_SSE2) && !defined(OPUS_X86_PRESUME_SSE2)) || \
  (defined(OPUS_X86_MAY_HAVE_SSE4_1) && !defined(OPUS_X86_PRESUME_SSE4_1)) || \
  (defined(OPUS_X86_MAY_HAVE_AVX2) && !defined(OPUS_X86_PRESUME_AVX2)) || \
  (defined(OPUS_X86_MAY_HAVE_AVX512VNNI) && !defined(OPUS_X86_PRESUME_AVX512VNNI)))

#if defined(_MSC_VER)

#include <intrin.h>
static _inline void cpuid(unsigned int CPUInfo[4], unsigned int InfoType)
{
    __cpuid((int*)CPUInfo, InfoType);
}

#else

#if defined(CPU_INFO_BY_C)
#include <cpuid.h>
#endif

static void cpuid(unsigned int CPUInfo[4], unsigned int InfoType)
{
#if defined(CPU_INFO_BY_ASM)
#if defined(__i386__) && defined(__PIC__)
/* %ebx is PIC register in 32-bit, so mustn't clobber it. */
    __asm__ __volatile__ (
        "xchg %%ebx, %1\n"
        "cpuid\n"
        "xchg %%ebx, %1\n":
        "=a" (CPUInfo[0]),
        "=r" (CPUInfo[1]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        /* We clear ECX to avoid a valgrind false-positive prior to v3.17.0. */
        "0" (InfoType), "2" (0)
    );
#else
    __asm__ __volatile__ (
        "cpuid":
        "=a" (CPUInfo[0]),
        "=b" (CPUInfo[1]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        /* We clear ECX to avoid a valgrind false-positive prior to v3.17.0. */
        "0" (InfoType), "2" (0)
    );
#endif
#elif defined(CPU_INFO_BY_C)
    /* We use __get_cpuid_count to clear ECX to avoid a valgrind false-positive
        prior to v3.17.0.*/
    if (!__get_cpuid_count(InfoType, 0, &(CPUInfo[0]), &(CPUInfo[1]), &(CPUInfo[2]), &(CPUInfo[3]))) {
        /* Our function cannot fail, but __get_cpuid{_count} can.
           Returning all zeroes will effectively disable all SIMD, which is
            what we want on CPUs that don't support CPUID. */
        CPUInfo[3] = CPUInfo[2] = CPUInfo[1] = CPUInfo[0] = 0;
    }
#else
# error "Configured to use x86 RTCD, but no CPU detection method available. " \
 "Reconfigure with --disable-rtcd (or send patches)."
#endif
}

#endif

#if defined(OPUS_X86_MAY_HAVE_AVX512VNNI)
/* Returns the low 32 bits of the extended control register XCR0, which tells
   us whether the OS has enabled saving/restoring of the relevant vector state.
   This must only be called once OSXSAVE (CPUID.1:ECX[27]) has been confirmed,
   otherwise XGETBV is an illegal instruction. */
static opus_uint32 get_xcr0(void)
{
#if defined(_MSC_VER)
    return (opus_uint32)_xgetbv(0);
#elif defined(CPU_INFO_BY_ASM) || defined(CPU_INFO_BY_C)
    opus_uint32 eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    (void)edx;
    return eax;
#else
    return 0;
#endif
}
#endif

typedef struct CPU_Feature{
    /*  SIMD: 128-bit */
    int HW_SSE;
    int HW_SSE2;
    int HW_SSE41;
    /*  SIMD: 256-bit */
    int HW_AVX2;
    /*  256-bit VNNI int8 dot product (EVEX-encoded vpdpbusd), requires
        AVX512F + AVX512VL + AVX512_VNNI and OS support for AVX-512 state. */
    int HW_AVX512VNNI;
} CPU_Feature;

static void opus_cpu_feature_check(CPU_Feature *cpu_feature)
{
    unsigned int info[4];
    unsigned int nIds = 0;

    cpuid(info, 0);
    nIds = info[0];

    if (nIds >= 1){
        unsigned int leaf1_ecx;
        cpuid(info, 1);
        leaf1_ecx = info[2];
        cpu_feature->HW_SSE = (info[3] & (1 << 25)) != 0;
        cpu_feature->HW_SSE2 = (info[3] & (1 << 26)) != 0;
        cpu_feature->HW_SSE41 = (info[2] & (1 << 19)) != 0;
        cpu_feature->HW_AVX2 = (info[2] & (1 << 28)) != 0 && (info[2] & (1 << 12)) != 0;
        cpu_feature->HW_AVX512VNNI = 0;
        if (cpu_feature->HW_AVX2 && nIds >= 7) {
            cpuid(info, 7);
            cpu_feature->HW_AVX2 = cpu_feature->HW_AVX2 && (info[1] & (1 << 5)) != 0;
#if defined(OPUS_X86_MAY_HAVE_AVX512VNNI)
            /* leaf 7, sub-leaf 0: EBX[16]=AVX512F, EBX[31]=AVX512VL,
               ECX[11]=AVX512_VNNI. The 256-bit EVEX vpdpbusd we emit for this
               tier needs F+VL+VNNI. We also require that the OS has enabled
               AVX-512 register state (XCR0 bits 5,6,7, on top of the SSE/AVX
               bits 1,2), otherwise the EVEX-encoded instructions fault. */
            if (cpu_feature->HW_AVX2
             && (info[1] & (1u << 16)) != 0   /* AVX512F     */
             && (info[1] & (1u << 31)) != 0   /* AVX512VL    */
             && (info[2] & (1u << 11)) != 0   /* AVX512_VNNI */
             && (leaf1_ecx & (1u << 27)) != 0 /* OSXSAVE     */) {
                opus_uint32 xcr0 = get_xcr0();
                unsigned int avx512_state = (1u << 1) | (1u << 2)
                                          | (1u << 5) | (1u << 6) | (1u << 7);
                cpu_feature->HW_AVX512VNNI =
                    (xcr0 & avx512_state) == avx512_state;
            }
#endif
        } else {
            cpu_feature->HW_AVX2 = 0;
        }
    }
    else {
        cpu_feature->HW_SSE = 0;
        cpu_feature->HW_SSE2 = 0;
        cpu_feature->HW_SSE41 = 0;
        cpu_feature->HW_AVX2 = 0;
        cpu_feature->HW_AVX512VNNI = 0;
    }
}

static int opus_select_arch_impl(void)
{
    CPU_Feature cpu_feature;
    int arch;

    opus_cpu_feature_check(&cpu_feature);

    arch = 0;
    if (!cpu_feature.HW_SSE)
    {
       return arch;
    }
    arch++;

    if (!cpu_feature.HW_SSE2)
    {
       return arch;
    }
    arch++;

    if (!cpu_feature.HW_SSE41)
    {
        return arch;
    }
    arch++;

    if (!cpu_feature.HW_AVX2)
    {
        return arch;
    }
    arch++;

    if (!cpu_feature.HW_AVX512VNNI)
    {
        return arch;
    }
    arch++;

    return arch;
}

int opus_select_arch(void) {
    int arch = opus_select_arch_impl();
#ifdef FUZZING
    /* Randomly downgrade the architecture. */
    arch = rand()%(arch+1);
#endif
    return arch;
}

#endif
