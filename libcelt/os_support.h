/* Copyright (C) 2007 Jean-Marc Valin
      
   File: os_support.h
   This is the (tiny) OS abstraction layer. Aside from math.h, this is the
   only place where system headers are allowed.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef OS_SUPPORT_H
#define OS_SUPPORT_H

#ifdef CUSTOM_SUPPORT
#  include "custom_support.h"
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/** CELT wrapper for calloc(). To do your own dynamic allocation, all you need to do is replace this function, celt_realloc and celt_free 
    NOTE: celt_alloc needs to CLEAR THE MEMORY */
#ifndef OVERRIDE_CELT_ALLOC
static inline void *celt_alloc (int size)
{
   /* WARNING: this is not equivalent to malloc(). If you want to use malloc() 
      or your own allocator, YOU NEED TO CLEAR THE MEMORY ALLOCATED. Otherwise
      you will experience strange bugs */
   return calloc(size,1);
}
#endif

/** Same as celt_alloc(), except that the area is only needed inside a CELT call (might cause problem with wideband though) */
#ifndef OVERRIDE_CELT_ALLOC_SCRATCH
static inline void *celt_alloc_scratch (int size)
{
   /* Scratch space doesn't need to be cleared */
   return calloc(size,1);
}
#endif

/** CELT wrapper for free(). To do your own dynamic allocation, all you need to do is replace this function, celt_realloc and celt_alloc */
#ifndef OVERRIDE_CELT_FREE
static inline void celt_free (void *ptr)
{
   free(ptr);
}
#endif

/** Same as celt_free(), except that the area is only needed inside a CELT call (might cause problem with wideband though) */
#ifndef OVERRIDE_CELT_FREE_SCRATCH
static inline void celt_free_scratch (void *ptr)
{
   free(ptr);
}
#endif

/** Copy n bytes of memory from src to dst. The 0* term provides compile-time type checking  */
#ifndef OVERRIDE_CELT_COPY
#define CELT_COPY(dst, src, n) (memcpy((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Copy n bytes of memory from src to dst, allowing overlapping regions. The 0* term 
    provides compile-time type checking */
#ifndef OVERRIDE_CELT_MOVE
#define CELT_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Set n bytes of memory to value of c, starting at address s */
#ifndef OVERRIDE_CELT_MEMSET
#define CELT_MEMSET(dst, c, n) (memset((dst), (c), (n)*sizeof(*(dst))))
#endif

/*#ifdef __GNUC__
#pragma GCC poison printf sprintf
#pragma GCC poison malloc free realloc calloc
#endif*/

#endif /* OS_SUPPORT_H */

