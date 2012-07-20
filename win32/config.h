#ifndef CONFIG_H
#define CONFIG_H

#define CELT_BUILD            1

#define restrict
#define inline __inline
#define getpid _getpid

#define USE_ALLOCA            1

/* Comment out the next line for floating-point code */
/*#define FIXED_POINT           1 */

#define OPUS_BUILD            1

/* Get rid of the CELT VS compile warnings */
#if 1
#pragma warning(disable : 4996)/* This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details. */
#endif

#include "version.h"

#endif CONFIG_H
