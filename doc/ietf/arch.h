/* Copyright (C) 2003-2008 Jean-Marc Valin */
/**
   @file arch.h
   @brief Various architecture definitions for CELT
*/
/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
   
- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
   
- Neither the name of the Xiph.org Foundation nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.
   
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ARCH_H
#define ARCH_H

#include "celt_types.h"

#define CELT_SIG_SCALE 32768.

#define celt_fatal(str) _celt_fatal(str, __FILE__, __LINE__);
#ifdef ENABLE_ASSERTIONS
#define celt_assert(cond) {if (!(cond)) \
		{celt_fatal("assertion failed: " #cond);}}
#define celt_assert2(cond, message) {if (!(cond)) \
		{celt_fatal("assertion failed: " #cond "\n" message);}}
#else
#define celt_assert(cond)
#define celt_assert2(cond, message)
#endif


#define ABS(x) ((x) < 0 ? (-(x)) : (x))
#define ABS16(x) ((x) < 0 ? (-(x)) : (x))
#define MIN16(a,b) ((a) < (b) ? (a) : (b))
#define MAX16(a,b) ((a) > (b) ? (a) : (b))
#define ABS32(x) ((x) < 0 ? (-(x)) : (x))
#define MIN32(a,b) ((a) < (b) ? (a) : (b))
#define MAX32(a,b) ((a) > (b) ? (a) : (b))
#define IMIN(a,b) ((a) < (b) ? (a) : (b))
#define IMAX(a,b) ((a) > (b) ? (a) : (b))

#define float2int(flt) ((int)(floor(.5+flt)))

#define SCALEIN(a)	((a)*CELT_SIG_SCALE)
#define SCALEOUT(a)	((a)*(1/CELT_SIG_SCALE))

#ifndef GLOBAL_STACK_SIZE
#ifdef FIXED_POINT
#define GLOBAL_STACK_SIZE 25000
#else
#define GLOBAL_STACK_SIZE 40000
#endif
#endif

#endif /* ARCH_H */
