## Process this file with automake to produce Makefile.in. -*-Makefile-*-

# $Id: Makefile.am,v 1.11 2004/02/18 06:59:40 jm Exp $

# Disable automatic dependency tracking if using other tools than gcc and gmake
#AUTOMAKE_OPTIONS = no-dependencies

INCLUDES = -I$(top_srcdir)/libcelt -I$(top_builddir)/ @OGG_CFLAGS@

EXTRA_DIST = getopt_win.h getopt.c getopt1.c wave_out.c wave_out.h skeleton.h

include_HEADERS = 
noinst_HEADERS = wav_io.h

bin_PROGRAMS = celtenc celtdec

celtenc_SOURCES = celtenc.c wav_io.c skeleton.c
celtenc_LDADD = $(top_builddir)/libcelt/libcelt@LIBCELT_SUFFIX@.la $(OGG_LIBS)

celtdec_SOURCES = celtdec.c wav_io.c
celtdec_LDADD = $(top_builddir)/libcelt/libcelt@LIBCELT_SUFFIX@.la $(OGG_LIBS)
