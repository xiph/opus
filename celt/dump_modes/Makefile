
CFLAGS=-O2 -Wall -Wextra -DHAVE_CONFIG_H
INCLUDES=-I. -I../ -I../.. -I../../include

SOURCES = dump_modes.c \
          ../modes.c \
          ../cwrs.c \
          ../rate.c \
          ../entenc.c \
          ../entdec.c \
          ../mathops.c \
          ../mdct.c \
          ../kiss_fft.c

ifdef HAVE_ARM_NE10
CC = gcc
CFLAGS += -mfpu=neon
INCLUDES += -I$(NE10_INCDIR) -DHAVE_ARM_NE10 -DOPUS_ARM_NEON_INTR
LIBDIR = -l:$(NE10_LIBDIR)/libNE10.so
SOURCES += ../arm/celt_ne10_fft.c \
           dump_modes_arm_ne10.c \
           ../arm/armcpu.c
endif

all: dump_modes

dump_modes:
	$(PREFIX)$(CC) $(CFLAGS) $(INCLUDES) -DCUSTOM_MODES_ONLY -DCUSTOM_MODES $(SOURCES) -o $@ $(LIBDIR) -lm

clean:
	rm -f dump_modes
