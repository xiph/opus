/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "opus.h"
#include "silk_debug.h"
#include "opus_types.h"

#define MAX_PACKET 1500

void print_usage( char* argv[] )
{
    fprintf(stderr, "Usage: %s [-e | -d] <application (0/1)> <sampling rate (Hz)> <channels (1/2)> "
        "<bits per second>  [options] <input> <output>\n\n", argv[0]);
    fprintf(stderr, "mode: 0 for VoIP, 1 for audio:\n" );
    fprintf(stderr, "options:\n" );
    fprintf(stderr, "-e                   : only runs the encoder (output the bit-stream)\n" );
    fprintf(stderr, "-d                   : only runs the decoder (reads the bit-stream as input)\n" );
    fprintf(stderr, "-cbr                 : enable constant bitrate; default: variable bitrate\n" );
    fprintf(stderr, "-cvbr                : enable constraint variable bitrate; default: unconstraint\n" );
    fprintf(stderr, "-bandwidth <NB|MB|WB|SWB|FB> : audio bandwidth (from narrowband to fullband); default: sampling rate\n" );
    fprintf(stderr, "-framesize <2.5|5|10|20|40|60> : frame size in ms; default: 20 \n" );
    fprintf(stderr, "-max_payload <bytes> : maximum payload size in bytes, default: 1024\n" );
    fprintf(stderr, "-complexity <comp>   : complexity, 0 (lowest) ... 10 (highest); default: 10\n" );
    fprintf(stderr, "-inbandfec           : enable SILK inband FEC\n" );
    fprintf(stderr, "-forcemono           : force mono encoding, even for stereo input\n" );
    fprintf(stderr, "-dtx                 : enable SILK DTX\n" );
    fprintf(stderr, "-loss <perc>         : simulate packet loss, in percent (0-100); default: 0\n" );
}

#ifdef _WIN32
#   define STR_CASEINSENSITIVE_COMPARE(x, y) _stricmp(x, y)
#else
#   define STR_CASEINSENSITIVE_COMPARE(x, y) strcasecmp(x, y)
#endif

static void int_to_char(opus_uint32 i, unsigned char ch[4])
{
    ch[0] = i>>24;
    ch[1] = (i>>16)&0xFF;
    ch[2] = (i>>8)&0xFF;
    ch[3] = i&0xFF;
}

static opus_uint32 char_to_int(unsigned char ch[4])
{
    return ((opus_uint32)ch[0]<<24) | ((opus_uint32)ch[1]<<16)
         | ((opus_uint32)ch[2]<< 8) |  (opus_uint32)ch[3];
}

int main(int argc, char *argv[])
{
    int err;
    char *inFile, *outFile;
    FILE *fin, *fout;
    OpusEncoder *enc;
    OpusDecoder *dec;
    int args;
    int len[2];
    int frame_size, channels;
    int bitrate_bps;
    unsigned char *data[2];
    int sampling_rate;
    int use_vbr;
    int max_payload_bytes;
    int complexity;
    int use_inbandfec;
    int use_dtx;
    int forcemono;
    int cvbr = 0;
    int packet_loss_perc;
    int count=0, count_act=0, k;
    int skip;
    int stop=0;
    short *in, *out;
    int application;
    double bits=0.0, bits_act=0.0, bits2=0.0, nrg;
    int bandwidth=-1;
    const char *bandwidth_string;
    int lost = 0, lost_prev = 1;
    int toggle = 0;
    int enc_final_range[2];
    int encode_only=0, decode_only=0;

    if (argc < 7 )
    {
       print_usage( argv );
       return 1;
    }

    fprintf(stderr, "%s\n", opus_get_version_string());

    if (strcmp(argv[1], "-e")==0)
    {
        encode_only = 1;
        argv++;
        argc--;
    } else if (strcmp(argv[1], "-d")==0)
    {
        decode_only = 1;
        argv++;
        argc--;
    }
    application = atoi(argv[1]) + OPUS_APPLICATION_VOIP;
    sampling_rate = atoi(argv[2]);
    channels = atoi(argv[3]);
    bitrate_bps = atoi(argv[4]);

    if (sampling_rate != 8000 && sampling_rate != 12000 && sampling_rate != 16000
     && sampling_rate != 24000 && sampling_rate != 48000)
    {
        fprintf(stderr, "Supported sampling rates are 8000, 12000, 16000, "
                "24000 and 48000.\n");
        return 1;
    }
    frame_size = sampling_rate/50;

    /* defaults: */
    use_vbr = 1;
    bandwidth = OPUS_BANDWIDTH_AUTO;
    max_payload_bytes = MAX_PACKET;
    complexity = 10;
    use_inbandfec = 0;
    forcemono = 0;
    use_dtx = 0;
    packet_loss_perc = 0;
    int max_frame_size = 960*6;
    int curr_read=0;

    args = 5;
    while( args < argc - 2 ) {
        /* process command line options */
        if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-cbr" ) == 0 ) {
            use_vbr = 0;
            args++;
        } else if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-bandwidth" ) == 0 ) {
            if (strcmp(argv[ args + 1 ], "NB")==0)
                bandwidth = OPUS_BANDWIDTH_NARROWBAND;
            else if (strcmp(argv[ args + 1 ], "MB")==0)
                bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
            else if (strcmp(argv[ args + 1 ], "WB")==0)
                bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            else if (strcmp(argv[ args + 1 ], "SWB")==0)
                bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
            else if (strcmp(argv[ args + 1 ], "FB")==0)
                bandwidth = OPUS_BANDWIDTH_FULLBAND;
            else {
                fprintf(stderr, "Unknown bandwidth %s. Supported are NB, MB, WB, SWB, FB.\n", argv[ args + 1 ]);
                return 1;
            }
            args += 2;
        } else if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-framesize" ) == 0 ) {
            if (strcmp(argv[ args + 1 ], "2.5")==0)
                frame_size = sampling_rate/400;
            else if (strcmp(argv[ args + 1 ], "5")==0)
                frame_size = sampling_rate/200;
            else if (strcmp(argv[ args + 1 ], "10")==0)
                frame_size = sampling_rate/100;
            else if (strcmp(argv[ args + 1 ], "20")==0)
                frame_size = sampling_rate/50;
            else if (strcmp(argv[ args + 1 ], "40")==0)
                frame_size = sampling_rate/25;
            else if (strcmp(argv[ args + 1 ], "60")==0)
                frame_size = 3*sampling_rate/50;
            else {
                fprintf(stderr, "Unsupported frame size: %s ms. Supported are 2.5, 5, 10, 20, 40, 60.\n", argv[ args + 1 ]);
                return 1;
            }
            args += 2;
        } else if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-max_payload" ) == 0 ) {
            max_payload_bytes = atoi( argv[ args + 1 ] );
            args += 2;
        } else if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-complexity" ) == 0 ) {
            complexity = atoi( argv[ args + 1 ] );
            args += 2;
        } else if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-inbandfec" ) == 0 ) {
            use_inbandfec = 1;
            args++;
        } else if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-forcemono" ) == 0 ) {
            forcemono = 1;
            args++;
        } else if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-cvbr" ) == 0 ) {
            cvbr = 1;
            args++;
        } else if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-dtx") == 0 ) {
            use_dtx = 1;
            args++;
        } else if( STR_CASEINSENSITIVE_COMPARE( argv[ args ], "-loss" ) == 0 ) {
            packet_loss_perc = atoi( argv[ args + 1 ] );
            args += 2;
        } else {
            printf( "Error: unrecognized setting: %s\n\n", argv[ args ] );
            print_usage( argv );
            return 1;
        }
    }

    if( application < OPUS_APPLICATION_VOIP || application > OPUS_APPLICATION_AUDIO) {
        fprintf (stderr, "mode must be: 0 or 1\n");
        return 1;
    }

    if (max_payload_bytes < 0 || max_payload_bytes > MAX_PACKET)
    {
        fprintf (stderr, "max_payload_bytes must be between 0 and %d\n",
                          MAX_PACKET);
        return 1;
    }
    if (bitrate_bps < 0 || bitrate_bps*frame_size/sampling_rate > max_payload_bytes*8)
    {
        fprintf (stderr, "bytes per packet must be between 0 and %d\n",
                          max_payload_bytes);
        return 1;
    }

    inFile = argv[argc-2];
    fin = fopen(inFile, "rb");
    if (!fin)
    {
        fprintf (stderr, "Could not open input file %s\n", argv[argc-2]);
        return 1;
    }
    outFile = argv[argc-1];
    fout = fopen(outFile, "wb+");
    if (!fout)
    {
        fprintf (stderr, "Could not open output file %s\n", argv[argc-1]);
        return 1;
    }

    enc = opus_encoder_create(sampling_rate, channels, application);
    dec = opus_decoder_create(sampling_rate, channels);

    if (enc==NULL)
    {
        fprintf(stderr, "Failed to create an encoder\n");
        exit(1);
    }
    if (dec==NULL)
    {
        fprintf(stderr, "Failed to create a decoder\n");
        exit(1);
    }

    opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate_bps));
    opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(bandwidth));
    opus_encoder_ctl(enc, OPUS_SET_VBR_FLAG(use_vbr));
    opus_encoder_ctl(enc, OPUS_SET_VBR_CONSTRAINT(cvbr));
    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(complexity));
    opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC_FLAG(use_inbandfec));
    opus_encoder_ctl(enc, OPUS_SET_FORCE_MONO(forcemono));
    opus_encoder_ctl(enc, OPUS_SET_DTX_FLAG(use_dtx));
    opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(packet_loss_perc));

    skip = 5*sampling_rate/1000;
    /* When SILK resamples, add 18 samples delay */
    /*if (mode != MODE_SILK_ONLY || sampling_rate > 16000)
        skip += 18;*/

    switch(bandwidth)
    {
    case OPUS_BANDWIDTH_NARROWBAND:
         bandwidth_string = "narrowband";
         break;
    case OPUS_BANDWIDTH_MEDIUMBAND:
         bandwidth_string = "mediumband";
         break;
    case OPUS_BANDWIDTH_WIDEBAND:
         bandwidth_string = "wideband";
         break;
    case OPUS_BANDWIDTH_SUPERWIDEBAND:
         bandwidth_string = "superwideband";
         break;
    case OPUS_BANDWIDTH_FULLBAND:
         bandwidth_string = "fullband";
         break;
    case OPUS_BANDWIDTH_AUTO:
         bandwidth_string = "auto";
         break;
    default:
         bandwidth_string = "unknown";
    }

    fprintf(stderr, "Encoding %d Hz input at %.3f kb/s in %s mode with %d-sample frames.\n", sampling_rate, bitrate_bps*0.001, bandwidth_string, frame_size);

    in = (short*)malloc(frame_size*channels*sizeof(short));
    out = (short*)malloc(max_frame_size*channels*sizeof(short));
    data[0] = (unsigned char*)calloc(max_payload_bytes,sizeof(char));
    if ( use_inbandfec ) {
        data[1] = (unsigned char*)calloc(max_payload_bytes,sizeof(char));
    }
    while (!stop)
    {
        if (decode_only)
        {
            unsigned char ch[4];
            err = fread(ch, 1, 4, fin);
            len[toggle] = char_to_int(ch);
            if (len[toggle]>max_payload_bytes || len[toggle]<0)
            {
                fprintf(stderr, "Invalid payload length\n");
                break;
            }
            err = fread(ch, 1, 4, fin);
            enc_final_range[toggle] = char_to_int(ch);
            err = fread(data[toggle], 1, len[toggle], fin);
            if (feof(fin))
                break;
        } else {
            err = fread(in, sizeof(short)*channels, frame_size, fin);
            curr_read = err;
            if (curr_read < frame_size)
            {
                int i;
                for (i=curr_read*channels;i<frame_size*channels;i++)
                   in[i] = 0;
                stop = 1;
            }

            len[toggle] = opus_encode(enc, in, frame_size, data[toggle], max_payload_bytes);
            enc_final_range[toggle] = opus_encoder_get_final_range( enc );
            if (len[toggle] < 0)
            {
                fprintf (stderr, "opus_encode() returned %d\n", len[toggle]);
                return 1;
            }
        }

        if (encode_only)
        {
            unsigned char int_field[4];
            int_to_char(len[toggle], int_field);
            fwrite(int_field, 1, 4, fout);
            int_to_char(enc_final_range[toggle], int_field);
            fwrite(int_field, 1, 4, fout);
            fwrite(data[toggle], 1, len[toggle], fout);
        } else {
            int output_samples;
            lost = rand()%100 < packet_loss_perc || len[toggle]==0;
            if( count >= use_inbandfec ) {
                /* delay by one packet when using in-band FEC */
                if( use_inbandfec  ) {
                    if( lost_prev ) {
                        /* attempt to decode with in-band FEC from next packet */
                        output_samples = opus_decode(dec, lost ? NULL : data[toggle], len[toggle], out, max_frame_size, 1);
                    } else {
                        /* regular decode */
                        output_samples = opus_decode(dec, data[1-toggle], len[1-toggle], out, max_frame_size, 0);
                    }
                } else {
                    output_samples = opus_decode(dec, lost ? NULL : data[toggle], len[toggle], out, max_frame_size, 0);
                }
                if (output_samples>0)
                {
                    fwrite(out+skip, sizeof(short)*channels, output_samples-skip, fout);
                    skip = 0;
                }
            }
        }

        /* compare final range encoder rng values of encoder and decoder */
        if( enc_final_range[toggle^use_inbandfec]!=0  && !encode_only && !lost && !lost_prev &&
             opus_decoder_get_final_range( dec ) != enc_final_range[toggle^use_inbandfec] ) {
            fprintf (stderr, "Error: Range coder state mismatch between encoder and decoder in frame %d: 0x%8x vs 0x%8x\n", count,  enc_final_range[toggle^use_inbandfec], opus_decoder_get_final_range( dec ));
            return 0;
        }

        lost_prev = lost;

        /* count bits */
        bits += len[toggle]*8;
        if( count >= use_inbandfec ) {
            nrg = 0.0;
            if (!decode_only)
            {
                for ( k = 0; k < frame_size * channels; k++ ) {
                    nrg += in[ k ] * (double)in[ k ];
                }
            }
            if ( ( nrg / ( frame_size * channels ) ) > 1e5 ) {
                bits_act += len[toggle]*8;
                count_act++;
            }
	    /* Variance */
            bits2 += len[toggle]*len[toggle]*64;
        }
        count++;
        toggle = (toggle + use_inbandfec) & 1;
    }
    fprintf (stderr, "average bitrate:             %7.3f kb/s\n", 1e-3*bits*sampling_rate/(frame_size*(double)count));
    fprintf (stderr, "active bitrate:              %7.3f kb/s\n", 1e-3*bits_act*sampling_rate/(frame_size*(double)count_act));
    fprintf (stderr, "bitrate standard deviation:  %7.3f kb/s\n", 1e-3*sqrt(bits2/count - bits*bits/(count*(double)count))*sampling_rate/frame_size);
    /* Close any files to which intermediate results were stored */
    SILK_DEBUG_STORE_CLOSE_FILES
    silk_TimerSave("opus_timing.txt");
    opus_encoder_destroy(enc);
    opus_decoder_destroy(dec);
    free(data[0]);
    if (use_inbandfec)
        free(data[1]);
    fclose(fin);
    fclose(fout);
    free(in);
    free(out);
    return 0;
}
