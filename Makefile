#					
# Makefile for Silk SDK			
#
# Copyright (c) 2010, Skype Limited
# All rights reserved.
#

#Platform detection and settings

BUILD_OS := $(shell uname | sed -e 's/^.*Darwin.*/MacOS-X/ ; s/^.*CYGWIN.*/Windows/')

BUILD_ARCHITECTURE := $(shell uname -m | sed -e 's/i686/i386/')

EXESUFFIX = 
LIBPREFIX = lib
LIBSUFFIX = .a
OBJSUFFIX = .o

CC     = $(TOOLCHAIN_PREFIX)gcc$(TOOLCHAIN_SUFFIX)
AR     = $(TOOLCHAIN_PREFIX)ar
RANLIB = $(TOOLCHAIN_PREFIX)ranlib
CP     = $(TOOLCHAIN_PREFIX)cp

cflags-from-defines    = $(addprefix -D,$(1))
cflags-from-includes   = $(addprefix -I,$(1))
ldflags-from-ldlibdirs = $(addprefix -L,$(1))
ldlibs-from-libs       = $(addprefix -l,$(1))

CFLAGS	+= -Wall -enable-threads -O3

CFLAGS  += $(call cflags-from-defines,$(CDEFINES))
CFLAGS  += $(call cflags-from-includes,$(CINCLUDES))
LDFLAGS += $(call ldflags-from-ldlibdirs,$(LDLIBDIRS))
LDLIBS  += $(call ldlibs-from-libs,$(LIBS))

COMPILE.c.cmdline   = $(CC) -c $(CFLAGS) -o $@ $<
LINK.o.cmdline      = $(LINK.o) -lm $^ $(LDLIBS) -o $@$(EXESUFFIX) 
ARCHIVE.cmdline     = $(AR) $(ARFLAGS) $@ $^ && $(RANLIB) $@

%$(OBJSUFFIX):%.c
	$(COMPILE.c.cmdline)

ifneq (,$(filter FIXED_POINT, $(CDEFINES)))

# Directives

CINCLUDES += interface src_common src_fix src_SigProc_FIX test

# VPATH e.g. VPATH = src:../headers
VPATH = ./ \
        interface \
	  src_common \
	  src_fix \
	  src_SigProc_FIX \
        test 

SRCS_C = $(wildcard src_common/*.c src_fix/*.c src_SigProc_FIX/*.c )

else

# Directives

CINCLUDES += interface src_common src_fix src_SigProc_FIX src_SigProc_FLP test

# VPATH e.g. VPATH = src:../headers
VPATH = ./ \
        interface \
	  src_common \
	  src_flp \
	  src_SigProc_FIX \
	  src_SigProc_FLP \
        test 

SRCS_C = $(wildcard src_common/*.c src_flp/*.c src_SigProc_FIX/*.c  src_SigProc_FLP/*.c )

endif

# Variable definitions
LIB_NAME = SKP_SILK_SDK
TARGET = $(LIBPREFIX)$(LIB_NAME)$(LIBSUFFIX)


OBJS := $(patsubst %.c,%$(OBJSUFFIX),$(SRCS_C))

ENCODER_SRCS_C = test/Encoder.c test/SKP_debug.c
ENCODER_OBJS := $(patsubst %.c,%$(OBJSUFFIX),$(ENCODER_SRCS_C))

DECODER_SRCS_C = test/Decoder.c test/SKP_debug.c
DECODER_OBJS := $(patsubst %.c,%$(OBJSUFFIX),$(DECODER_SRCS_C))

SIGNALCMP_SRCS_C = test/signalCompare.c
SIGNALCMP_OBJS := $(patsubst %.c,%$(OBJSUFFIX),$(SIGNALCMP_SRCS_C))

LIBS = \
	$(LIB_NAME)

LDLIBDIRS = ./

# Rules
default: all

all: $(TARGET) encoder decoder signalcompare

lib: $(TARGET)

$(TARGET): $(OBJS)
	$(ARCHIVE.cmdline)

encoder$(EXESUFFIX): $(ENCODER_OBJS)	
	$(LINK.o.cmdline)

decoder$(EXESUFFIX): $(DECODER_OBJS)	
	$(LINK.o.cmdline)

signalcompare$(EXESUFFIX): $(SIGNALCMP_OBJS)	
	$(LINK.o.cmdline)

clean:
	$(RM) $(TARGET)* $(OBJS) $(ENCODER_OBJS) $(DECODER_OBJS) \
		  $(SIGNALCMP_OBJS) $(TEST_OBJS) \
		  encoder$(EXESUFFIX) decoder$(EXESUFFIX) signalcompare$(EXESUFFIX)

