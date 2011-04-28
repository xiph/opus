/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved. 
Redistribution and use in source and binary forms, with or without 
modification, (subject to the limitations in the disclaimer below) 
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright 
notice, this list of conditions and the following disclaimer in the 
documentation and/or other materials provided with the distribution.
- Neither the name of Skype Limited, nor the names of specific 
contributors, may be used to endorse or promote products derived from 
this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED 
BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND 
CONTRIBUTORS ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifndef _SKP_DEBUG_H_
#define _SKP_DEBUG_H_

#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE    1
#endif

#include "SKP_Silk_typedef.h"
#include <stdio.h>      /* file writing */
#include <string.h>     /* strcpy, strcmp */

#ifdef  __cplusplus
extern "C"
{
#endif

unsigned long GetHighResolutionTime(void); /* O: time in usec*/

/* make SKP_DEBUG dependent on compiler's _DEBUG */
#if defined _WIN32
    #ifdef _DEBUG
        #define SKP_DEBUG  1
    #else
        #define SKP_DEBUG  0
    #endif

    /* overrule the above */
    #if 0
    //  #define NO_ASSERTS
    #undef  SKP_DEBUG  
    #define SKP_DEBUG  1
    #endif
#else
    #define SKP_DEBUG  0
#endif

/* Flag for using timers */
#define SKP_TIC_TOC 0

#if SKP_TIC_TOC

#if (defined(_WIN32) || defined(_WINCE)) 
#include <windows.h>    /* timer */
#pragma warning( disable : 4996 )       // stop bitching about strcpy in TIC()
#else   // Linux or Mac
#include <sys/time.h>
#endif

/*********************************/
/* timer functions for profiling */
/*********************************/
/* example:                                                         */
/*                                                                  */
/* TIC(LPC)                                                         */
/* do_LPC(in_vec, order, acoef);    // do LPC analysis              */
/* TOC(LPC)                                                         */
/*                                                                  */
/* and call the following just before exiting (from main)           */
/*                                                                  */
/* SKP_TimerSave("SKP_TimingData.txt");                             */
/*                                                                  */
/* results are now in SKP_TimingData.txt                            */

void SKP_TimerSave(char *file_name);

/* max number of timers (in different locations) */
#define SKP_NUM_TIMERS_MAX                  50
/* max length of name tags in TIC(..), TOC(..) */
#define SKP_NUM_TIMERS_MAX_TAG_LEN          30

extern int           SKP_Timer_nTimers;
extern int           SKP_Timer_depth_ctr;
extern char          SKP_Timer_tags[SKP_NUM_TIMERS_MAX][SKP_NUM_TIMERS_MAX_TAG_LEN];
#ifdef _WIN32
extern LARGE_INTEGER SKP_Timer_start[SKP_NUM_TIMERS_MAX];
#else
extern unsigned long SKP_Timer_start[SKP_NUM_TIMERS_MAX];
#endif  
extern unsigned int  SKP_Timer_cnt[SKP_NUM_TIMERS_MAX];
extern SKP_int64     SKP_Timer_sum[SKP_NUM_TIMERS_MAX];
extern SKP_int64     SKP_Timer_max[SKP_NUM_TIMERS_MAX];
extern SKP_int64     SKP_Timer_min[SKP_NUM_TIMERS_MAX];
extern SKP_int64     SKP_Timer_depth[SKP_NUM_TIMERS_MAX];

/* WARNING: TIC()/TOC can measure only up to 0.1 seconds at a time */
#ifdef _WIN32
#define TIC(TAG_NAME) {                                     \
    static int init = 0;                                    \
    static int ID = -1;                                     \
    if( init == 0 )                                         \
    {                                                       \
        int k;                                              \
        init = 1;                                           \
        for( k = 0; k < SKP_Timer_nTimers; k++ ) {          \
            if( strcmp(SKP_Timer_tags[k], #TAG_NAME) == 0 ) {   \
                ID = k;                                     \
                break;                                      \
            }                                               \
        }                                                   \
        if (ID == -1) {                                     \
            ID = SKP_Timer_nTimers;                         \
            SKP_Timer_nTimers++;                            \
            SKP_Timer_depth[ID] = SKP_Timer_depth_ctr;      \
            strcpy(SKP_Timer_tags[ID], #TAG_NAME);          \
            SKP_Timer_cnt[ID] = 0;                          \
            SKP_Timer_sum[ID] = 0;                          \
            SKP_Timer_min[ID] = 0xFFFFFFFF;                 \
            SKP_Timer_max[ID] = 0;                          \
        }                                                   \
    }                                                       \
    SKP_Timer_depth_ctr++;                                  \
    QueryPerformanceCounter(&SKP_Timer_start[ID]);          \
}
#else
#define TIC(TAG_NAME) {                                     \
    static int init = 0;                                    \
    static int ID = -1;                                     \
    if( init == 0 )                                         \
    {                                                       \
        int k;                                              \
        init = 1;                                           \
        for( k = 0; k < SKP_Timer_nTimers; k++ ) {          \
        if( strcmp(SKP_Timer_tags[k], #TAG_NAME) == 0 ) {   \
                ID = k;                                     \
                break;                                      \
            }                                               \
        }                                                   \
        if (ID == -1) {                                     \
            ID = SKP_Timer_nTimers;                         \
            SKP_Timer_nTimers++;                            \
            SKP_Timer_depth[ID] = SKP_Timer_depth_ctr;      \
            strcpy(SKP_Timer_tags[ID], #TAG_NAME);          \
            SKP_Timer_cnt[ID] = 0;                          \
            SKP_Timer_sum[ID] = 0;                          \
            SKP_Timer_min[ID] = 0xFFFFFFFF;                 \
            SKP_Timer_max[ID] = 0;                          \
        }                                                   \
    }                                                       \
    SKP_Timer_depth_ctr++;                                  \
    SKP_Timer_start[ID] = GetHighResolutionTime();          \
}
#endif

#ifdef _WIN32
#define TOC(TAG_NAME) {                                             \
    LARGE_INTEGER lpPerformanceCount;                               \
    static int init = 0;                                            \
    static int ID = 0;                                              \
    if( init == 0 )                                                 \
    {                                                               \
        int k;                                                      \
        init = 1;                                                   \
        for( k = 0; k < SKP_Timer_nTimers; k++ ) {                  \
            if( strcmp(SKP_Timer_tags[k], #TAG_NAME) == 0 ) {       \
                ID = k;                                             \
                break;                                              \
            }                                                       \
        }                                                           \
    }                                                               \
    QueryPerformanceCounter(&lpPerformanceCount);                   \
    lpPerformanceCount.QuadPart -= SKP_Timer_start[ID].QuadPart;    \
    if((lpPerformanceCount.QuadPart < 100000000) &&                 \
        (lpPerformanceCount.QuadPart >= 0)) {                       \
        SKP_Timer_cnt[ID]++;                                        \
        SKP_Timer_sum[ID] += lpPerformanceCount.QuadPart;           \
        if( lpPerformanceCount.QuadPart > SKP_Timer_max[ID] )       \
            SKP_Timer_max[ID] = lpPerformanceCount.QuadPart;        \
        if( lpPerformanceCount.QuadPart < SKP_Timer_min[ID] )       \
            SKP_Timer_min[ID] = lpPerformanceCount.QuadPart;        \
    }                                                               \
    SKP_Timer_depth_ctr--;                                          \
}
#else
#define TOC(TAG_NAME) {                                             \
    unsigned long endTime;                                          \
    static int init = 0;                                            \
    static int ID = 0;                                              \
    if( init == 0 )                                                 \
    {                                                               \
        int k;                                                      \
        init = 1;                                                   \
        for( k = 0; k < SKP_Timer_nTimers; k++ ) {                  \
            if( strcmp(SKP_Timer_tags[k], #TAG_NAME) == 0 ) {       \
                ID = k;                                             \
                break;                                              \
            }                                                       \
        }                                                           \
    }                                                               \
    endTime = GetHighResolutionTime();                              \
    endTime -= SKP_Timer_start[ID];                                 \
    if((endTime < 100000000) &&                                     \
        (endTime >= 0)) {                                           \
        SKP_Timer_cnt[ID]++;                                        \
        SKP_Timer_sum[ID] += endTime;                               \
        if( endTime > SKP_Timer_max[ID] )                           \
            SKP_Timer_max[ID] = endTime;                            \
        if( endTime < SKP_Timer_min[ID] )                           \
            SKP_Timer_min[ID] = endTime;                            \
    }                                                               \
        SKP_Timer_depth_ctr--;                                      \
}
#endif

#else /* SKP_TIC_TOC */

/* define macros as empty strings */
#define TIC(TAG_NAME)
#define TOC(TAG_NAME)
#define SKP_TimerSave(FILE_NAME)

#endif /* SKP_TIC_TOC */



#if SKP_DEBUG
/************************************/
/* write data to file for debugging */
/************************************/
/* opens an empty file if this file has not yet been open, then writes to the file and closes it            */
/* if file has been open previously it is opened again and the fwrite is appending, finally it is closed    */
#define SAVE_DATA( FILE_NAME, DATA_PTR, N_BYTES ) {                 \
    static SKP_int32 init = 0;                                      \
    FILE *fp;                                                       \
    if (init == 0)	{                                               \
        init = 1;                                                   \
        fp = fopen(#FILE_NAME, "wb");                               \
    } else {                                                        \
        fp = fopen(#FILE_NAME, "ab+");                              \
    }	                                                            \
    fwrite((DATA_PTR), (N_BYTES), 1, fp);                           \
    fclose(fp);                                                     \
}	

/* Example: DEBUG_STORE_DATA(testfile.pcm, &RIN[0], 160*sizeof(SKP_int16)); */

#if 0
/* Ensure that everything is written to files when an assert breaks */
#define DEBUG_STORE_DATA(FILE_NAME, DATA_PTR, N_BYTES) SAVE_DATA(FILE_NAME, DATA_PTR, N_BYTES)
#define DEBUG_STORE_CLOSE_FILES

#else

#define SKP_NUM_STORES_MAX                                  100
extern FILE *SKP_debug_store_fp[ SKP_NUM_STORES_MAX ];
extern int SKP_debug_store_count;

/* Faster way of storing the data */
#define DEBUG_STORE_DATA( FILE_NAME, DATA_PTR, N_BYTES ) {          \
    static SKP_int init = 0, cnt = 0;                               \
    static FILE **fp;                                               \
    if (init == 0) {                                                \
        init = 1;											        \
        cnt = SKP_debug_store_count++;                              \
        SKP_debug_store_fp[ cnt ] = fopen(#FILE_NAME, "wb");        \
    }                                                               \
    fwrite((DATA_PTR), (N_BYTES), 1, SKP_debug_store_fp[ cnt ]);    \
}

/* Call this at the end of main() */
#define DEBUG_STORE_CLOSE_FILES {                                   \
    SKP_int i;                                                      \
    for( i = 0; i < SKP_debug_store_count; i++ ) {                  \
        fclose( SKP_debug_store_fp[ i ] );                          \
    }                                                               \
}
#endif

/* micro sec */
#define SKP_GETTIME(void)       time = (SKP_int64) GetHighResolutionTime();     

#else /* SKP_DEBUG */

/* define macros as empty strings */
#define DEBUG_STORE_DATA(FILE_NAME, DATA_PTR, N_BYTES)
#define SAVE_DATA(FILE_NAME, DATA_PTR, N_BYTES)
#define DEBUG_STORE_CLOSE_FILES

#endif /* SKP_DEBUG */

#ifdef  __cplusplus
}
#endif

#endif /* _SKP_DEBUG_H_ */
