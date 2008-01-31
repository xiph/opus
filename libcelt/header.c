/* (C) 2008 Jean-Marc Valin, CSIRO
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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "celt_header.h"
#include "os_support.h"

/*typedef struct {
   char         codec_id[8];
   char         codec_version[20];
   celt_int32_t version_id;
   celt_int32_t header_size;
   celt_int32_t mode;
   celt_int32_t sample_rate;
   celt_int32_t nb_channels;
   celt_int32_t bytes_per_packet;
   celt_int32_t extra_headers;
} CELTHeader;*/

void celt_header_init(CELTHeader *header, celt_int32_t rate, celt_int32_t nb_channels, const CELTMode *m)
{
   CELT_COPY(header->codec_id, "CELT    ", 8);
   CELT_COPY(header->codec_version, "experimental        ", 20);

   header->version_id = 0x80000000;
   header->header_size = 56;
   header->mode = 0;
   header->sample_rate = rate;
   header->nb_channels = nb_channels;
   header->bytes_per_packet = -1;
   header->extra_headers = 0xdeadbeef;
}

int celt_header_to_packet(const CELTHeader *header, unsigned char *packet, celt_uint32_t size)
{
   CELT_MEMSET(packet, 0, sizeof(*header));
   /* FIXME: Do it in a endian-safe, alignment-safe, overflow-safe manner */
   CELT_COPY(packet, (unsigned char*)header, sizeof(*header));
   return sizeof(*header);
}

int celt_header_from_packet(const unsigned char *packet, celt_uint32_t size, CELTHeader *header)
{
   CELT_COPY((unsigned char*)header, packet, sizeof(*header));
   return sizeof(*header);
}

