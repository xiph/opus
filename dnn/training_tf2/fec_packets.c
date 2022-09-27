#include <stdio.h>
#include <inttypes.h>

#include "fec_packets.h"

int get_fec_frame(const char * const filename, float *features, int packet_index, int subframe_index)
{

    int16_t version;
    int16_t header_size;
    int16_t num_packets;
    int16_t packet_size;
    int16_t subframe_size;
    int16_t subframes_per_packet;
    int16_t num_features;
    long offset;

    FILE *fid = fopen(filename, "rb");
    
    /* read header */
    if (fread(&version, sizeof(version), 1, fid) != 1) goto error;
    if (fread(&header_size, sizeof(header_size), 1, fid) != 1) goto error;
    if (fread(&num_packets, sizeof(num_packets), 1, fid) != 1) goto error;
    if (fread(&packet_size, sizeof(packet_size), 1, fid) != 1) goto error;
    if (fread(&subframe_size, sizeof(subframe_size), 1, fid) != 1) goto error;
    if (fread(&subframes_per_packet, sizeof(subframes_per_packet), 1, fid) != 1) goto error;
    if (fread(&num_features, sizeof(num_features), 1, fid) != 1) goto error;

    /* check if indices are valid */
    if (packet_index >= num_packets || subframe_index >= subframes_per_packet)
    {
        fprintf(stderr, "get_fec_frame: index out of bounds\n");
        goto error;
    }

    /* calculate offset in file (+ 2 is for rate) */
    offset = header_size + packet_index * packet_size + 2 + subframe_index * subframe_size;
    fseek(fid, offset, SEEK_SET);

    /* read features */
    if (fread(features, sizeof(*features), num_features, fid) != num_features) goto error;

    fclose(fid);
    return 0;

error:
    fclose(fid);
    return 1;
}

int get_fec_rate(const char * const filename, int packet_index)
{
    int16_t version;
    int16_t header_size;
    int16_t num_packets;
    int16_t packet_size;
    int16_t subframe_size;
    int16_t subframes_per_packet;
    int16_t num_features;
    long offset;
    int16_t rate;

    FILE *fid = fopen(filename, "rb");
    
    /* read header */
    if (fread(&version, sizeof(version), 1, fid) != 1) goto error;
    if (fread(&header_size, sizeof(header_size), 1, fid) != 1) goto error;
    if (fread(&num_packets, sizeof(num_packets), 1, fid) != 1) goto error;
    if (fread(&packet_size, sizeof(packet_size), 1, fid) != 1) goto error;
    if (fread(&subframe_size, sizeof(subframe_size), 1, fid) != 1) goto error;
    if (fread(&subframes_per_packet, sizeof(subframes_per_packet), 1, fid) != 1) goto error;
    if (fread(&num_features, sizeof(num_features), 1, fid) != 1) goto error;

    /* check if indices are valid */
    if (packet_index >= num_packets)
    {
        fprintf(stderr, "get_fec_rate: index out of bounds\n");
        goto error;
    }

    /* calculate offset in file (+ 2 is for rate) */
    offset = header_size + packet_index * packet_size;
    fseek(fid, offset, SEEK_SET);

    /* read rate */
    if (fread(&rate, sizeof(rate), 1, fid) != 1) goto error;

    fclose(fid);
    return (int) rate;

error:
    fclose(fid);
    return -1;
}

#if 0
int main()
{
    float features[20];
    int i;

    if (get_fec_frame("../test.fec", &features[0], 0, 127))
    {
        return 1;
    }

    for (i = 0; i < 20; i ++)
    {
        printf("%d %f\n", i, features[i]);
    }

    printf("rate: %d\n", get_fec_rate("../test.fec", 0));

}
#endif