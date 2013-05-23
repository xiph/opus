CFLAGS=-O2 -Wall -Wextra -DHAVE_CONFIG_H
INCLUDES=-I../ -I../.. -I../../include

all: dump_modes

dump_modes:
	$(CC) $(CFLAGS) $(INCLUDES) -DCUSTOM_MODES dump_modes.c ../modes.c ../cwrs.c ../rate.c ../entenc.c ../entdec.c ../mathops.c ../mdct.c ../kiss_fft.c -o dump_modes -lm 

clean:
	rm -f dump_modes
