/* Copyright (c) 2010 Xiph.Org Foundation, Skype Limited
   Written by Jean-Marc Valin and Koen Vos */
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

static int decode_length(unsigned char *c, int len)
{
    int tmp;
    tmp = c[0];
    if (len < 1)
        return -1;
    if (tmp >= 252)
    {
        if (len >= 2)
            return 4*c[1] + (tmp&0x3) + 252;
        else
            return -1;
    } else {
        return tmp;
    }
}

int count_frames(unsigned char *packet, int len)
{
    int sz = packet[0]&0x7;

    if (sz == 0)
        return 1;
    else if (sz == 1 || sz == 4)
        return 2;
    else if (sz == 2 || sz == 5)
        return 3;
    else if (sz == 3)
    {
        /* Many packets, same size */
        int count, payload;
        int flen = decode_length(packet+1, len-1);
        if (flen<=0)
            return -1;
        payload = len - 2;
        if (flen>=252)
            payload--;
        count = payload/flen;
        if (count*flen==payload)
            return count;
        else
            return -1;
    } else /* if (sz == 6 || sz == 7) */
    {
        /* Many packets, different sizes */
        int count = 0;
        int pos = 1;
        int bytes = 1;
        int extra = 0;
        if (sz==7)
            extra = 1;
        while (bytes < len)
        {
            int tmp=extra+1;
            int flen = decode_length(packet+pos, len-bytes);
            if (flen==-1)
                return -1;
            if (flen >= 252)
                tmp = 2;
            pos += tmp;
            bytes += tmp+flen;
            count++;
        }
        if (bytes != len)
            return -1;
        else
            return count;
    }
}

#define MAX_FRAMES 256
int opus_merge_packets(unsigned char **packets, int *plen, int nb_packets,
        unsigned *output, int maxlen)
{
    int i;
    unsigned char cfg[MAX_FRAMES];
    unsigned char flen[MAX_FRAMES];
    int nb_frames=0;

    for (i=0;i<nb_packets;i++)
    {
        int tmp = count_frames(packets[i], plen[i]);
        if (tmp<=0)
            return -1;
        nb_frames += tmp;
    }
    return nb_frames;
}
