all:
	gcc -DHAVE_CONFIG_H -I../celt -W -Wextra -Wall -O3 -g -I../ -I../celt/libcelt -L../celt/libcelt/.libs/ -o test_hybrid test_hybrid.c hybrid_decoder.c hybrid_encoder.c ../silk/test/SKP_debug.o ../silk/libSKP_SILK_SDK.a -lcelt0 -I../silk/interface -lm
