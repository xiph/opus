#ifndef CONFIG_H
#define CONFIG_H

#define CELT_BUILD            1

#define restrict 
#define inline __inline

#define USE_ALLOCA            1

/* Comment out the next line for floating-point code */
/*#define FIXED_POINT           1 */

#define OPUS_BUILD            1

/* Get rid of the CELT VS compile warnings */
#if 1
#pragma warning(disable : 4018)/* signed/unsigned mismatch */
#pragma warning(disable : 4244)/* conversion from 'double' to 'opus_val16', possible loss of data */
#pragma warning(disable : 4267)/* conversion from 'size_t' to 'int', possible loss of data */
#pragma warning(disable : 4305)/* truncation from 'double' to 'const float' */
#pragma warning(disable : 4311)/* pointer truncation from 'char *' to 'long' */
#pragma warning(disable : 4554)/* check operator precedence for possible error; use parentheses to clarify precedence */
#pragma warning(disable : 4996)/* This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details. */
#endif

#endif CONFIG_H
