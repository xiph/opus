# Disable automatic dependency tracking if using other tools than gcc and gmake
#AUTOMAKE_OPTIONS = no-dependencies


#EXTRA_DIST=

#INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include -I$(top_builddir) @OGG_CFLAGS@

lib_LTLIBRARIES = libcelt.la

# Sources for compilation in the library
libcelt_la_SOURCES = mdct.c

#noinst_HEADERS =

libcelt_la_LDFLAGS = -version-info @CELT_LT_CURRENT@:@CELT_LT_REVISION@:@CELT_LT_AGE@

noinst_HEADERS = os_support.h arch.h mdct.h

noinst_PROGRAMS = testcelt
testcelt_SOURCES = testcelt.c
testcelt_LDADD = $(top_builddir)/libcelt/libcelt.la
