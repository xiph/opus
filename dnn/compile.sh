#!/bin/sh

gcc -Wall -W -O3 -g -I../include dump_data.c freq.c kiss_fft.c pitch.c celt_lpc.c -o dump_data -lm
gcc -o test_lpcnet -mavx2 -mfma -g -O3 -Wall -W -Wextra test_lpcnet.c lpcnet.c nnet.c nnet_data.c freq.c kiss_fft.c pitch.c celt_lpc.c -lm
