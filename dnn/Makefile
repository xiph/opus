# Makefile for LPCNet

CC=gcc
CFLAGS+=-Wall -W -Wextra -Wno-unused-function -O3 -g -I../include 

AVX2:=$(shell cat /proc/cpuinfo | grep -c avx2)
AVX:=$(shell cat /proc/cpuinfo | grep -c avx)
NEON:=$(shell cat /proc/cpuinfo | grep -c neon)

ifneq ($(AVX2),0)
CFLAGS+=-mavx2 -mfma 
else
# AVX2 machines will also match on AVX
ifneq ($(AVX),0)
CFLAGS+=-mavx
endif
endif

ifneq ($(NEON),0)
CFLAGS+=-mfpu=neon -march=armv8-a -mtune=cortex-a53
endif

all: dump_data test_lpcnet test_vec

dump_data_objs := src/dump_data.o src/freq.o src/kiss_fft.o src/pitch.o src/celt_lpc.o
dump_data_deps := $(dump_data_objs:.o=.d)
dump_data: $(dump_data_objs)
	gcc -o $@ $(CFLAGS) $(dump_data_objs) -lm

-include $dump_data_deps(_deps)

test_lpcnet_objs := src/test_lpcnet.o src/lpcnet.o src/nnet.o src/nnet_data.o src/freq.o src/kiss_fft.o src/pitch.o src/celt_lpc.o
test_lpcnet_deps := $(test_lpcnet_objs:.o=.d)
test_lpcnet: $(test_lpcnet_objs)
	gcc -o $@ $(CFLAGS) $(test_lpcnet_objs) -lm

-include $(test_lpcnet_deps)

test_vec_objs := src/test_vec.o
test_vec_deps := $(test_vec_objs:.o=.d)
test_vec: $(test_vec_objs)
	gcc -o $@ $(CFLAGS) $(test_vec_objs) -lm

-include $(test_vec_deps)

test: test_vec
	./test_vec

clean:
	rm -f dump_data test_lpcnet test_vec
	rm -f $(dump_data_objs) $(dump_data_deps) 
	rm -f $(test_lpcnet_objs) $(test_lpcnet_deps) 
	rm -f $(test_vec_objs) $(test_vec_deps) 
