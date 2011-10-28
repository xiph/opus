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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

/* Filter coefficients for IIR/FIR polyphase resampling     *
 * Total size: < 600 Words (1.2 kB)                         */

#include "resampler_private.h"

/* Tables for 2x downsampler */
const opus_int16 silk_resampler_down2_0 = 9872;
const opus_int16 silk_resampler_down2_1 = 39809 - 65536;

/* Tables for 2x upsampler, high quality */
const opus_int16 silk_resampler_up2_hq_0[ 2 ] = {  4280, 33727 - 65536 };
const opus_int16 silk_resampler_up2_hq_1[ 2 ] = { 16295, 54015 - 65536 };
const opus_int16 silk_resampler_up2_hq_notch[ 4 ] = { 6554,  -3932,   6554,  30573 };

/* Tables with IIR and FIR coefficients for fractional downsamplers (90 Words) */
silk_DWORD_ALIGN const opus_int16 silk_Resampler_3_4_COEFS[ 2 + 3 * RESAMPLER_DOWN_ORDER_FIR / 2 ] = {
    -20253, -13986,
        86,      7,   -151,    368,   -542,    232,  11041,  21904,
        39,     90,   -181,    216,    -17,   -877,   6408,  19695,
         2,    113,   -108,      2,    314,   -977,   2665,  15787,
};

silk_DWORD_ALIGN const opus_int16 silk_Resampler_2_3_COEFS[ 2 + 2 * RESAMPLER_DOWN_ORDER_FIR / 2 ] = {
    -13997, -14120,
        60,   -174,     71,    298,   -800,    659,   9238,  17461,
        48,    -40,   -150,    314,   -155,   -845,   4188,  14293,
};

silk_DWORD_ALIGN const opus_int16 silk_Resampler_1_2_COEFS[ 2 + RESAMPLER_DOWN_ORDER_FIR / 2 ] = {
      1233, -14293,
       -91,    162,    169,   -342,   -505,   1332,   5281,   8742,
};

silk_DWORD_ALIGN const opus_int16 silk_Resampler_1_3_COEFS[ 2 + RESAMPLER_DOWN_ORDER_FIR / 2 ] = {
     16306, -14409,
        99,   -201,   -220,    -16,    572,   1483,   2433,   3043,
};

silk_DWORD_ALIGN const opus_int16 silk_Resampler_2_3_COEFS_LQ[ 2 + 2 * 2 ] = {
     -2797,  -6507,
      4697,  10739,
      1567,   8276,
};

/* Table with interplation fractions of 1/288 : 2/288 : 287/288 (432 Words) */
silk_DWORD_ALIGN const opus_int16 silk_resampler_frac_FIR_144[ 144 ][ RESAMPLER_ORDER_FIR_144 / 2 ] = {
    {  -25,    58, 32526},
    {   -8,   -69, 32461},
    {    8,  -195, 32393},
    {   25,  -318, 32321},
    {   41,  -439, 32244},
    {   57,  -557, 32163},
    {   72,  -673, 32079},
    {   88,  -787, 31990},
    {  103,  -899, 31897},
    {  118, -1009, 31801},
    {  133, -1116, 31700},
    {  148, -1221, 31596},
    {  162, -1324, 31488},
    {  177, -1424, 31376},
    {  191, -1522, 31260},
    {  205, -1618, 31140},
    {  218, -1712, 31017},
    {  231, -1803, 30890},
    {  245, -1893, 30759},
    {  257, -1980, 30625},
    {  270, -2065, 30487},
    {  282, -2147, 30346},
    {  295, -2228, 30201},
    {  306, -2306, 30052},
    {  318, -2382, 29900},
    {  330, -2456, 29745},
    {  341, -2528, 29586},
    {  352, -2597, 29424},
    {  362, -2664, 29259},
    {  373, -2730, 29090},
    {  383, -2793, 28918},
    {  393, -2854, 28743},
    {  402, -2912, 28565},
    {  411, -2969, 28384},
    {  421, -3024, 28199},
    {  429, -3076, 28012},
    {  438, -3126, 27822},
    {  446, -3175, 27628},
    {  454, -3221, 27432},
    {  462, -3265, 27233},
    {  469, -3307, 27031},
    {  476, -3348, 26826},
    {  483, -3386, 26619},
    {  490, -3422, 26409},
    {  496, -3456, 26196},
    {  502, -3488, 25981},
    {  508, -3518, 25763},
    {  514, -3547, 25543},
    {  519, -3573, 25320},
    {  524, -3597, 25095},
    {  529, -3620, 24867},
    {  533, -3640, 24637},
    {  538, -3659, 24405},
    {  541, -3676, 24171},
    {  545, -3691, 23934},
    {  548, -3704, 23696},
    {  552, -3716, 23455},
    {  554, -3726, 23212},
    {  557, -3733, 22967},
    {  559, -3740, 22721},
    {  561, -3744, 22472},
    {  563, -3747, 22222},
    {  565, -3748, 21970},
    {  566, -3747, 21716},
    {  567, -3745, 21460},
    {  568, -3741, 21203},
    {  568, -3735, 20944},
    {  568, -3728, 20684},
    {  568, -3719, 20422},
    {  568, -3708, 20159},
    {  568, -3697, 19894},
    {  567, -3683, 19628},
    {  566, -3668, 19361},
    {  564, -3652, 19093},
    {  563, -3634, 18823},
    {  561, -3614, 18552},
    {  559, -3594, 18280},
    {  557, -3571, 18008},
    {  554, -3548, 17734},
    {  552, -3523, 17459},
    {  549, -3497, 17183},
    {  546, -3469, 16907},
    {  542, -3440, 16630},
    {  539, -3410, 16352},
    {  535, -3379, 16074},
    {  531, -3346, 15794},
    {  527, -3312, 15515},
    {  522, -3277, 15235},
    {  517, -3241, 14954},
    {  513, -3203, 14673},
    {  507, -3165, 14392},
    {  502, -3125, 14110},
    {  497, -3085, 13828},
    {  491, -3043, 13546},
    {  485, -3000, 13264},
    {  479, -2957, 12982},
    {  473, -2912, 12699},
    {  466, -2867, 12417},
    {  460, -2820, 12135},
    {  453, -2772, 11853},
    {  446, -2724, 11571},
    {  439, -2675, 11289},
    {  432, -2625, 11008},
    {  424, -2574, 10727},
    {  417, -2522, 10446},
    {  409, -2470, 10166},
    {  401, -2417,  9886},
    {  393, -2363,  9607},
    {  385, -2309,  9328},
    {  376, -2253,  9050},
    {  368, -2198,  8773},
    {  359, -2141,  8497},
    {  351, -2084,  8221},
    {  342, -2026,  7946},
    {  333, -1968,  7672},
    {  324, -1910,  7399},
    {  315, -1850,  7127},
    {  305, -1791,  6856},
    {  296, -1731,  6586},
    {  286, -1670,  6317},
    {  277, -1609,  6049},
    {  267, -1548,  5783},
    {  257, -1486,  5517},
    {  247, -1424,  5254},
    {  237, -1362,  4991},
    {  227, -1300,  4730},
    {  217, -1237,  4470},
    {  207, -1174,  4212},
    {  197, -1110,  3956},
    {  187, -1047,  3701},
    {  176,  -984,  3448},
    {  166,  -920,  3196},
    {  155,  -856,  2946},
    {  145,  -792,  2698},
    {  134,  -728,  2452},
    {  124,  -664,  2207},
    {  113,  -600,  1965},
    {  102,  -536,  1724},
    {   92,  -472,  1486},
    {   81,  -408,  1249},
    {   70,  -345,  1015},
    {   60,  -281,   783},
    {   49,  -217,   553},
    {   38,  -154,   325},
};
