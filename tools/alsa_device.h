/*
   Copyright (C) 2004-2006 Jean-Marc Valin
   Copyright (C) 2006 Commonwealth Scientific and Industrial Research
                      Organisation (CSIRO) Australia
   
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

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

#ifndef ALSA_DEVICE_H
#define ALSA_DEVICE_H

#include <sys/poll.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AlsaDevice_;

typedef struct AlsaDevice_ AlsaDevice;

AlsaDevice *alsa_device_open(char *device_name, unsigned int rate, int channels, int period);

void alsa_device_close(AlsaDevice *dev);

int alsa_device_read(AlsaDevice *dev, short *pcm, int len);

int alsa_device_write(AlsaDevice *dev, const short *pcm, int len);

int alsa_device_capture_ready(AlsaDevice *dev, struct pollfd *pfds, unsigned int nfds);

int alsa_device_playback_ready(AlsaDevice *dev, struct pollfd *pfds, unsigned int nfds);

void alsa_device_start(AlsaDevice *dev);

int alsa_device_nfds(AlsaDevice *dev);

void alsa_device_getfds(AlsaDevice *dev, struct pollfd *pfds, unsigned int nfds);

#ifdef __cplusplus
}
#endif

#endif
