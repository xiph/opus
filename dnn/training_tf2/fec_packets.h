#ifndef _FEC_PACKETS_H
#define _FEC_PACKETS_H

int get_fec_frame(const char * const filename, float *features, int packet_index, int subframe_index);
int get_fec_rate(const char * const filename, int packet_index);

#endif