/* celt_types.h taken from libogg */
/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2002             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: #ifdef jail to whip a few platforms into the UNIX ideal.
 last mod: $Id: os_types.h 7524 2004-08-11 04:20:36Z conrad $

 ********************************************************************/
/**
   @file celt_types.h
   @brief CELT types
*/
#ifndef _CELT_TYPES_H
#define _CELT_TYPES_H

/* Use the real stdint.h if it's there (taken from Paul Hsieh's pstdint.h) */
#if (defined(__STDC__) && __STDC__ && __STDC_VERSION__ >= 199901L) || (defined(__GNUC__) && (defined(_STDINT_H) || defined(_STDINT_H_)) || defined (HAVE_STDINT_H))
#include <stdint.h>

   typedef int16_t celt_int16_t;
   typedef uint16_t celt_uint16_t;
   typedef int32_t celt_int32_t;
   typedef uint32_t celt_uint32_t;
   typedef int64_t celt_int64_t;
   typedef uint64_t celt_uint64_t;
#elif defined(_WIN32) 

#  if defined(__CYGWIN__)
#    include <_G_config.h>
     typedef _G_int32_t celt_int32_t;
     typedef _G_uint32_t celt_uint32_t;
     typedef _G_int16_t celt_int16_t;
     typedef _G_uint16_t celt_uint16_t;
     typedef _G_int64_t celt_int64_t;
     typedef _G_uint64_t celt_uint64_t;
#  elif defined(__MINGW32__)
     typedef short celt_int16_t;
     typedef unsigned short celt_uint16_t;
     typedef int celt_int32_t;
     typedef unsigned int celt_uint32_t;
     typedef long long celt_int64_t;
     typedef unsigned long long celt_uint64_t;
#  elif defined(__MWERKS__)
     typedef int celt_int32_t;
     typedef unsigned int celt_uint32_t;
     typedef short celt_int16_t;
     typedef unsigned short celt_uint16_t;
     typedef long long celt_int64_t;
     typedef unsigned long long celt_uint64_t;
#  else
     /* MSVC/Borland */
     typedef __int32 celt_int32_t;
     typedef unsigned __int32 celt_uint32_t;
     typedef __int16 celt_int16_t;
     typedef unsigned __int16 celt_uint16_t;
     typedef __int64 celt_int64_t;
     typedef unsigned __int64 celt_uint64_t;
#  endif

#elif defined(__MACOS__)

#  include <sys/types.h>
   typedef SInt16 celt_int16_t;
   typedef UInt16 celt_uint16_t;
   typedef SInt32 celt_int32_t;
   typedef UInt32 celt_uint32_t;
   typedef SInt64 celt_int64_t;
   typedef UInt64 celt_uint64_t;

#elif (defined(__APPLE__) && defined(__MACH__)) /* MacOS X Framework build */

#  include <sys/types.h>
   typedef int16_t celt_int16_t;
   typedef u_int16_t celt_uint16_t;
   typedef int32_t celt_int32_t;
   typedef u_int32_t celt_uint32_t;
   typedef int64_t celt_int64_t;
   typedef u_int64_t celt_uint64_t;

#elif defined(__BEOS__)

   /* Be */
#  include <inttypes.h>
   typedef int16_t celt_int16_t;
   typedef u_int16_t celt_uint16_t;
   typedef int32_t celt_int32_t;
   typedef u_int32_t celt_uint32_t;
   typedef int64_t celt_int64_t;
   typedef u_int64_t celt_uint64_t;

#elif defined (__EMX__)

   /* OS/2 GCC */
   typedef short celt_int16_t;
   typedef unsigned short celt_uint16_t;
   typedef int celt_int32_t;
   typedef unsigned int celt_uint32_t;
   typedef long long celt_int64_t;
   typedef unsigned long long celt_uint64_t;

#elif defined (DJGPP)

   /* DJGPP */
   typedef short celt_int16_t;
   typedef int celt_int32_t;
   typedef unsigned int celt_uint32_t;
   typedef long long celt_int64_t;
   typedef unsigned long long celt_uint64_t;

#elif defined(R5900)

   /* PS2 EE */
   typedef int celt_int32_t;
   typedef unsigned celt_uint32_t;
   typedef short celt_int16_t;
   typedef long celt_int64_t;
   typedef unsigned long celt_uint64_t;

#elif defined(__SYMBIAN32__)

   /* Symbian GCC */
   typedef signed short celt_int16_t;
   typedef unsigned short celt_uint16_t;
   typedef signed int celt_int32_t;
   typedef unsigned int celt_uint32_t;
   typedef long long int celt_int64_t;
   typedef unsigned long long int celt_uint64_t;

#elif defined(CONFIG_TI_C54X) || defined (CONFIG_TI_C55X)

   typedef short celt_int16_t;
   typedef unsigned short celt_uint16_t;
   typedef long celt_int32_t;
   typedef unsigned long celt_uint32_t;
   typedef long long celt_int64_t;
   typedef unsigned long long celt_uint64_t;

#elif defined(CONFIG_TI_C6X)

   typedef short celt_int16_t;
   typedef unsigned short celt_uint16_t;
   typedef int celt_int32_t;
   typedef unsigned int celt_uint32_t;
   typedef long long int celt_int64_t;
   typedef unsigned long long int celt_uint64_t;

#else

   /* Give up, take a reasonable guess */
   typedef short celt_int16_t;
   typedef unsigned short celt_uint16_t;
   typedef int celt_int32_t;
   typedef unsigned int celt_uint32_t;
   typedef long long celt_int64_t;
   typedef unsigned long long celt_uint64_t;

#endif

#endif  /* _CELT_TYPES_H */
