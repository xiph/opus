/***************************************************************************
   Copyright (C) 2004-2006 by Jean-Marc Valin
   Copyright (C) 2006 Commonwealth Scientific and Industrial Research
                      Organisation (CSIRO) Australia
   Copyright (C) 2008-2009 Gregory Maxwell
   Copyright (c) 2007-2009 Xiph.Org Foundation

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
   
****************************************************************************/

/* Compile with something like:
 * gcc -oceltclient celtclient.c alsa_device.c -I../libcelt/ -lspeexdsp  -lasound -lcelt -lm
 */
 
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h> /* close() */
#include <string.h> /* memset() */

#include "alsa_device.h"
#include <celt.h>
#include <speex/speex_jitter.h>

#include <sched.h>

#define MAX_MSG 1500
#define SAMPLING_RATE 48000
#define FRAME_SIZE 256
#define PACKETSIZE 43
#define CHANNELS 1
#define HAS_SPEEX_AEC 

#if CHANNELS == 2
/* FIXME: The Speex AEC has multichannel support; but that API isn't being
   used here yet. */
#undef HAS_SPEEX_AEC   
#endif

#ifdef HAS_SPEEX_AEC 
#include <speex/speex_echo.h>
#endif

int main(int argc, char *argv[])
{
   
   int sd, rc, n;
   int i;
   struct sockaddr_in cliAddr, remoteAddr;
   char msg[MAX_MSG];
   struct hostent *h;
   int local_port, remote_port;
   int nfds;
   struct pollfd *pfds;
   AlsaDevice *audio_dev;
   int tmp;

   if (argc != 5)
   {
      fprintf(stderr, "Usage %s plughw:0,0 remote_host local_udp_port remote_udp_port\n",argv[0]);
      exit(1);
   }
  
   h = gethostbyname(argv[2]);
   if(h==NULL) {
      fprintf(stderr, "%s: unknown host '%s' \n", argv[0], argv[2]);
      exit(1);
   }

   local_port = atoi(argv[3]);
   remote_port = atoi(argv[4]);
   
   printf("%s: sending data to '%s' (IP : %s) \n", argv[0], h->h_name,
          inet_ntoa(*(struct in_addr *)h->h_addr_list[0]));

   {
      remoteAddr.sin_family = h->h_addrtype;
      memcpy((char *) &remoteAddr.sin_addr.s_addr,
            h->h_addr_list[0], h->h_length);
      remoteAddr.sin_port = htons(remote_port);
   }
   /* socket creation */
   sd=socket(AF_INET, SOCK_DGRAM, 0);
   if(sd<0) {
      printf("%s: cannot open socket \n",argv[0]);
      exit(1);
   }

   /* bind any port */
   cliAddr.sin_family = AF_INET;
   cliAddr.sin_addr.s_addr = htonl(INADDR_ANY);
   cliAddr.sin_port = htons(local_port);

   rc = bind(sd, (struct sockaddr *) &cliAddr, sizeof(cliAddr));
   if(rc<0) {
      printf("%s: cannot bind port\n", argv[0]);
      exit(1);
   }

   /* Setup audio device */
   audio_dev = alsa_device_open(argv[1], SAMPLING_RATE, CHANNELS, FRAME_SIZE);
   
   /* Setup the encoder and decoder in wideband */
   CELTEncoder *enc_state;
   CELTDecoder *dec_state;
   CELTMode *mode = celt_mode_create(SAMPLING_RATE, FRAME_SIZE, NULL);
   enc_state = celt_encoder_create_custom(mode, CHANNELS, NULL);
   dec_state = celt_decoder_create_custom(mode, CHANNELS, NULL);
   struct sched_param param;
   /*param.sched_priority = 40; */
   param.sched_priority = sched_get_priority_min(SCHED_FIFO);
   if (sched_setscheduler(0,SCHED_FIFO,&param))
      perror("sched_setscheduler");

   int send_timestamp = 0;
   int recv_started=0;
   
   /* Setup all file descriptors for poll()ing */
   nfds = alsa_device_nfds(audio_dev);
   pfds = malloc(sizeof(*pfds)*(nfds+1));
   alsa_device_getfds(audio_dev, pfds, nfds);
   pfds[nfds].fd = sd;
   pfds[nfds].events = POLLIN;

   /* Setup jitter buffer using decoder */
   JitterBuffer *jitter;
   jitter = jitter_buffer_init(FRAME_SIZE);
   tmp = FRAME_SIZE;
   jitter_buffer_ctl(jitter, JITTER_BUFFER_SET_MARGIN, &tmp);
#ifdef HAS_SPEEX_AEC
   /* Echo canceller with 200 ms tail length */
   SpeexEchoState *echo_state = speex_echo_state_init(FRAME_SIZE, 10*FRAME_SIZE);
   tmp = SAMPLING_RATE;
   speex_echo_ctl(echo_state, SPEEX_ECHO_SET_SAMPLING_RATE, &tmp);
#endif   
   alsa_device_start(audio_dev);
   
   /* Infinite loop on capture, playback and receiving packets */
   while (1)
   {
      /* Wait for either 1) capture 2) playback 3) socket data */
      poll(pfds, nfds+1, -1);
      /* Received packets */
      if (pfds[nfds].revents & POLLIN)
      {
         n = recv(sd, msg, MAX_MSG, 0);
         int recv_timestamp = ((int*)msg)[0];
   
         JitterBufferPacket packet;
         packet.data = msg+4;
         packet.len = n-4;
         packet.timestamp = recv_timestamp;
         packet.span = FRAME_SIZE;
         packet.sequence = 0;
         /* Put content of the packet into the jitter buffer, except for the pseudo-header */
         jitter_buffer_put(jitter, &packet);
         recv_started = 1;

      }
      /* Ready to play a frame (playback) */
      if (alsa_device_playback_ready(audio_dev, pfds, nfds))
      {
         short pcm[FRAME_SIZE*CHANNELS];
         if (recv_started)
         {
            JitterBufferPacket packet;
            /* Get audio from the jitter buffer */
            packet.data = msg;
            packet.len  = MAX_MSG;
            jitter_buffer_tick(jitter);
            jitter_buffer_get(jitter, &packet, FRAME_SIZE, NULL);
            if (packet.len==0)
              packet.data=NULL;
            celt_decode(dec_state, packet.data, packet.len, pcm, FRAME_SIZE);
         } else {
            for (i=0;i<FRAME_SIZE*CHANNELS;i++)
               pcm[i] = 0;
         }
         /* Playback the audio and reset the echo canceller if we got an underrun */

#ifdef HAS_SPEEX_AEC
         if (alsa_device_write(audio_dev, pcm, FRAME_SIZE)) 
            speex_echo_state_reset(echo_state);
         /* Put frame into playback buffer */
         speex_echo_playback(echo_state, pcm);
#else
         alsa_device_write(audio_dev, pcm, FRAME_SIZE);
#endif
      }
      /* Audio available from the soundcard (capture) */
      if (alsa_device_capture_ready(audio_dev, pfds, nfds))
      {
         short pcm[FRAME_SIZE*CHANNELS], pcm2[FRAME_SIZE*CHANNELS];
         char outpacket[MAX_MSG];
         /* Get audio from the soundcard */
         alsa_device_read(audio_dev, pcm, FRAME_SIZE);
         
#ifdef HAS_SPEEX_AEC
         /* Perform echo cancellation */
         speex_echo_capture(echo_state, pcm, pcm2);
         for (i=0;i<FRAME_SIZE*CHANNELS;i++)
            pcm[i] = pcm2[i];
#endif   
         /* Encode */
         celt_encode(enc_state, pcm, FRAME_SIZE, outpacket+4, PACKETSIZE);
         
         /* Pseudo header: four null bytes and a 32-bit timestamp */
         ((int*)outpacket)[0] = send_timestamp;
         send_timestamp += FRAME_SIZE;
         rc = sendto(sd, outpacket, PACKETSIZE+4, 0,
                (struct sockaddr *) &remoteAddr,
                sizeof(remoteAddr));
         
         if(rc<0) {
            perror("cannot send to socket");
            close(sd);
            exit(1);
         }
      }
      

   }


   return 0;
}
