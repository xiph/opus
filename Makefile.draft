# Copyright (c) 2011-2012 IETF Trust, Skype Limited, Jean-Marc Valin. All rights reserved.
#
#  This file is extracted from RFC6716. Please see that RFC for additional
#  information.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  - Redistributions of source code must retain the above copyright
#  notice, this list of conditions and the following disclaimer.
#
#  - Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
#  - Neither the name of Internet Society, IETF or IETF Trust, nor the
#  names of specific contributors, may be used to endorse or promote
#  products derived from this software without specific prior written
#  permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#################### COMPILE OPTIONS #######################

# Uncomment this for fixed-point build
#FIXED_POINT=1

# It is strongly recommended to uncomment one of these
# VAR_ARRAYS: Use C99 variable-length arrays for stack allocation
# USE_ALLOCA: Use alloca() for stack allocation
# If none is defined, then the fallback is a non-threadsafe global array
CFLAGS := -DUSE_ALLOCA $(CFLAGS)
#CFLAGS := -DVAR_ARRAYS $(CFLAGS)

# These options affect performance
# HAVE_LRINTF: Use C99 intrinsics to speed up float-to-int conversion
#      inline: Don't use the 'inline' keyword (for ANSI C compilers)
#    restrict: Don't use the 'restrict' keyword (for pre-C99 compilers)
#CFLAGS := -DHAVE_LRINTF $(CFLAGS)
#CFLAGS := -Dinline= $(CFLAGS)
CFLAGS := -Drestrict= $(CFLAGS)

###################### END OF OPTIONS ######################

CFLAGS += -DOPUS_VERSION='"1.0.0"'
include silk_sources.mk
include celt_sources.mk
include opus_sources.mk

ifdef FIXED_POINT
SILK_SOURCES += $(SILK_SOURCES_FIXED)
else
SILK_SOURCES += $(SILK_SOURCES_FLOAT)
endif

EXESUFFIX =
LIBPREFIX = lib
LIBSUFFIX = .a
OBJSUFFIX = .o

CC     = $(TOOLCHAIN_PREFIX)cc$(TOOLCHAIN_SUFFIX)
AR     = $(TOOLCHAIN_PREFIX)ar
RANLIB = $(TOOLCHAIN_PREFIX)ranlib
CP     = $(TOOLCHAIN_PREFIX)cp

cppflags-from-defines   = $(addprefix -D,$(1))
cppflags-from-includes  = $(addprefix -I,$(1))
ldflags-from-ldlibdirs  = $(addprefix -L,$(1))
ldlibs-from-libs                = $(addprefix -l,$(1))

WARNINGS = -Wall -W -Wstrict-prototypes -Wextra -Wcast-align -Wnested-externs -Wshadow
CFLAGS  += -O2 -g $(WARNINGS) -DOPUS_BUILD
ifdef FIXED_POINT
CFLAGS += -DFIXED_POINT=1 -DDISABLE_FLOAT_API
endif

CINCLUDES += include/ \
	silk/ \
	silk/float/ \
	silk/fixed/ \
	celt/ \
	src/

# VPATH e.g. VPATH = src:../headers
VPATH = ./ \
	silk/interface \
	silk/src_FIX \
	silk/src_FLP \
	silk/src_SigProc_FIX \
	silk/src_SigProc_FLP \
	test

LIBS = m

LDLIBDIRS = ./

CFLAGS  += $(call cppflags-from-defines,$(CDEFINES))
CFLAGS  += $(call cppflags-from-includes,$(CINCLUDES))
LDFLAGS += $(call ldflags-from-ldlibdirs,$(LDLIBDIRS))
LDLIBS  += $(call ldlibs-from-libs,$(LIBS))

COMPILE.c.cmdline   = $(CC) -c $(CFLAGS) -o $@ $<
LINK.o              = $(CC) $(LDPREFLAGS) $(LDFLAGS)
LINK.o.cmdline      = $(LINK.o) $^ $(LDLIBS) -o $@$(EXESUFFIX)

ARCHIVE.cmdline     = $(AR) $(ARFLAGS) $@ $^ && $(RANLIB) $@

%$(OBJSUFFIX):%.c
	$(COMPILE.c.cmdline)

%$(OBJSUFFIX):%.cpp
	$(COMPILE.cpp.cmdline)

# Directives


# Variable definitions
LIB_NAME = opus
TARGET = $(LIBPREFIX)$(LIB_NAME)$(LIBSUFFIX)

SRCS_C = $(SILK_SOURCES) $(CELT_SOURCES) $(OPUS_SOURCES)

OBJS := $(patsubst %.c,%$(OBJSUFFIX),$(SRCS_C))

OPUSDEMO_SRCS_C = src/opus_demo.c
OPUSDEMO_OBJS := $(patsubst %.c,%$(OBJSUFFIX),$(OPUSDEMO_SRCS_C))

OPUSCOMPARE_SRCS_C = src/opus_compare.c
OPUSCOMPARE_OBJS := $(patsubst %.c,%$(OBJSUFFIX),$(OPUSCOMPARE_SRCS_C))

# Rules
default: all

all: $(TARGET) lib opus_demo opus_compare

lib: $(TARGET)

$(TARGET): $(OBJS)
	$(ARCHIVE.cmdline)

opus_demo$(EXESUFFIX): $(OPUSDEMO_OBJS) $(TARGET)
	$(LINK.o.cmdline)
	
opus_compare$(EXESUFFIX): $(OPUSCOMPARE_OBJS)
	$(LINK.o.cmdline)

clean:
	rm -f opus_demo$(EXESUFFIX) opus_compare$(EXESUFFIX) $(TARGET) $(OBJS) $(OPUSDEMO_OBJS)
