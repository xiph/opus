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

/*                                                                      *
 * SKP_debug.c                                                          *
 *                                                                      *
 * This contains code to help debugging                                 *
 *                                                                      *
 * Copyright 2009 (c), Skype Limited                                    *
 * Date: 090629                                                         *
 *                                                                      */

#include "SKP_debug.h"
#include "../src_SigProc_FIX/SKP_Silk_SigProc_FIX.h"

#if SKP_TIC_TOC

#ifdef _WIN32

#if (defined(_WIN32) || defined(_WINCE)) 
#include <windows.h>    /* timer */
#else   // Linux or Mac
#include <sys/time.h>
#endif

unsigned long GetHighResolutionTime(void) /* O: time in usec*/
{
    /* Returns a time counter in microsec   */
    /* the resolution is platform dependent */
    /* but is typically 1.62 us resolution  */
    LARGE_INTEGER lpPerformanceCount;
    LARGE_INTEGER lpFrequency;
    QueryPerformanceCounter(&lpPerformanceCount);
    QueryPerformanceFrequency(&lpFrequency);
    return (unsigned long)((1000000*(lpPerformanceCount.QuadPart)) / lpFrequency.QuadPart);
}
#else   // Linux or Mac
unsigned long GetHighResolutionTime(void) /* O: time in usec*/
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return((tv.tv_sec*1000000)+(tv.tv_usec));
}
#endif

int           SKP_Timer_nTimers = 0;
int           SKP_Timer_depth_ctr = 0;
char          SKP_Timer_tags[SKP_NUM_TIMERS_MAX][SKP_NUM_TIMERS_MAX_TAG_LEN];
#ifdef WIN32
LARGE_INTEGER SKP_Timer_start[SKP_NUM_TIMERS_MAX];
#else
unsigned long SKP_Timer_start[SKP_NUM_TIMERS_MAX];
#endif  
unsigned int  SKP_Timer_cnt[SKP_NUM_TIMERS_MAX];
SKP_int64     SKP_Timer_min[SKP_NUM_TIMERS_MAX];
SKP_int64     SKP_Timer_sum[SKP_NUM_TIMERS_MAX];
SKP_int64     SKP_Timer_max[SKP_NUM_TIMERS_MAX];
SKP_int64     SKP_Timer_depth[SKP_NUM_TIMERS_MAX];

#ifdef WIN32
void SKP_TimerSave(char *file_name)
{
    if( SKP_Timer_nTimers > 0 ) 
    {
        int k;
        FILE *fp;
        LARGE_INTEGER lpFrequency;
        LARGE_INTEGER lpPerformanceCount1, lpPerformanceCount2;
        int del = 0x7FFFFFFF;
        double avg, sum_avg;
        /* estimate overhead of calling performance counters */
        for( k = 0; k < 1000; k++ ) {
            QueryPerformanceCounter(&lpPerformanceCount1);
            QueryPerformanceCounter(&lpPerformanceCount2);
            lpPerformanceCount2.QuadPart -= lpPerformanceCount1.QuadPart;
            if( (int)lpPerformanceCount2.LowPart < del )
                del = lpPerformanceCount2.LowPart;
        }
        QueryPerformanceFrequency(&lpFrequency);
        /* print results to file */
        sum_avg = 0.0f;
        for( k = 0; k < SKP_Timer_nTimers; k++ ) {
            if (SKP_Timer_depth[k] == 0) {
                sum_avg += (1e6 * SKP_Timer_sum[k] / SKP_Timer_cnt[k] - del) / lpFrequency.QuadPart * SKP_Timer_cnt[k];
            }
        }
        fp = fopen(file_name, "w");
        fprintf(fp, "                                min         avg     %%         max      count\n");
        for( k = 0; k < SKP_Timer_nTimers; k++ ) {
            if (SKP_Timer_depth[k] == 0) {
                fprintf(fp, "%-28s", SKP_Timer_tags[k]);
            } else if (SKP_Timer_depth[k] == 1) {
                fprintf(fp, " %-27s", SKP_Timer_tags[k]);
            } else if (SKP_Timer_depth[k] == 2) {
                fprintf(fp, "  %-26s", SKP_Timer_tags[k]);
            } else if (SKP_Timer_depth[k] == 3) {
                fprintf(fp, "   %-25s", SKP_Timer_tags[k]);
            } else {
                fprintf(fp, "    %-24s", SKP_Timer_tags[k]);
            }
            avg = (1e6 * SKP_Timer_sum[k] / SKP_Timer_cnt[k] - del) / lpFrequency.QuadPart;
            fprintf(fp, "%8.2f", (1e6 * (SKP_max_64(SKP_Timer_min[k] - del, 0))) / lpFrequency.QuadPart);
            fprintf(fp, "%12.2f %6.2f", avg, 100.0 * avg / sum_avg * SKP_Timer_cnt[k]);
            fprintf(fp, "%12.2f", (1e6 * (SKP_max_64(SKP_Timer_max[k] - del, 0))) / lpFrequency.QuadPart);
            fprintf(fp, "%10d\n", SKP_Timer_cnt[k]);
        }
        fprintf(fp, "                                microseconds\n");
        fclose(fp);
    }
}
#else
void SKP_TimerSave(char *file_name)
{
    if( SKP_Timer_nTimers > 0 ) 
    {
        int k;
        FILE *fp;
        /* print results to file */
        fp = fopen(file_name, "w");
        fprintf(fp, "                                min         avg         max      count\n");
        for( k = 0; k < SKP_Timer_nTimers; k++ )
        {
            if (SKP_Timer_depth[k] == 0) {
                fprintf(fp, "%-28s", SKP_Timer_tags[k]);
            } else if (SKP_Timer_depth[k] == 1) {
                fprintf(fp, " %-27s", SKP_Timer_tags[k]);
            } else if (SKP_Timer_depth[k] == 2) {
                fprintf(fp, "  %-26s", SKP_Timer_tags[k]);
            } else if (SKP_Timer_depth[k] == 3) {
                fprintf(fp, "   %-25s", SKP_Timer_tags[k]);
            } else {
                fprintf(fp, "    %-24s", SKP_Timer_tags[k]);
            }
            fprintf(fp, "%d ", SKP_Timer_min[k]);
            fprintf(fp, "%f ", (double)SKP_Timer_sum[k] / (double)SKP_Timer_cnt[k]);
            fprintf(fp, "%d ", SKP_Timer_max[k]);
            fprintf(fp, "%10d\n", SKP_Timer_cnt[k]);
        }
        fprintf(fp, "                                microseconds\n");
        fclose(fp);
    }
}
#endif

#endif /* SKP_TIC_TOC */

#if SKP_DEBUG
FILE *SKP_debug_store_fp[ SKP_NUM_STORES_MAX ];
int SKP_debug_store_count = 0;
#endif /* SKP_DEBUG */

