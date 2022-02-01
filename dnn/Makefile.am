ACLOCAL_AMFLAGS = -I m4

AM_CFLAGS = -I$(top_srcdir)/include $(DEPS_CFLAGS)

dist_doc_DATA = COPYING AUTHORS README README.md

include_HEADERS = include/lpcnet.h

lib_LTLIBRARIES = liblpcnet.la
noinst_HEADERS = arch.h  \
		 common.h  \
		 freq.h  \
		 _kiss_fft_guts.h  \
		 kiss_fft.h  \
		 kiss99.h  \
		 lpcnet_private.h \
		 opus_types.h  \
		 nnet_data.h  \
		 plc_data.h  \
		 nnet.h  \
		 pitch.h  \
		 tansig_table.h \
		 vec.h \
		 vec_avx.h \
		 vec_neon.h

liblpcnet_la_SOURCES = \
	common.c \
	kiss99.c \
	lpcnet.c \
	lpcnet_dec.c \
	lpcnet_enc.c \
	nnet.c \
	nnet_data.c \
	plc_data.c \
	ceps_codebooks.c \
	pitch.c \
	freq.c \
	kiss_fft.c \
	lpcnet_plc.c

liblpcnet_la_LIBADD = $(DEPS_LIBS) $(lrintf_lib) $(LIBM)
liblpcnet_la_LDFLAGS = -no-undefined \
 -version-info @OP_LT_CURRENT@:@OP_LT_REVISION@:@OP_LT_AGE@

noinst_PROGRAMS = lpcnet_demo dump_data

lpcnet_demo_SOURCES = lpcnet_demo.c
lpcnet_demo_LDADD = liblpcnet.la


#DUMP_SOURCES = freq.c kiss_fft.c pitch.c lpcnet_dec.c lpcnet_enc.c ceps_codebooks.c common.c
#DUMP_OBJ = $(DUMP_SOURCES:.c=.lo)
#dump_data_SOURCES = dump_data.c
#dump_data_LDADD = $(DUMP_OBJ) $(LIBM)

dump_data_SOURCES = common.c dump_data.c freq.c kiss_fft.c pitch.c lpcnet_dec.c lpcnet_enc.c ceps_codebooks.c
dump_data_LDADD = $(LIBM)
dump_data_CFLAGS = $(AM_CFLAGS)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = lpcnet.pc

debug:
	$(MAKE) CFLAGS="${CFLAGS} -O0 -ggdb -DOP_ENABLE_ASSERTIONS" all

EXTRA_DIST = \
 lpcnet.pc.in \
 lpcnet-uninstalled.pc.in \
 doc/Doxyfile.in \
 doc/Makefile

# Targets to build and install just the library without the docs
liblpcnet install-liblpcnet: NO_DOXYGEN = 1

lpcnet: all
install-lpcnet: install

# Or just the docs
docs: doc/doxygen-build.stamp

install-docs:
	@if [ -z "$(NO_DOXYGEN)" ]; then \
	  ( cd doc && \
	    echo "Installing documentation in $(DESTDIR)$(docdir)"; \
	    $(INSTALL) -d $(DESTDIR)$(docdir)/html/search; \
	    for f in `find html -type f \! -name "installdox"` ; do \
	      $(INSTALL_DATA) $$f $(DESTDIR)$(docdir)/$$f;     \
	    done ) \
	fi

doc/doxygen-build.stamp: doc/Doxyfile \
                         $(top_srcdir)/include/*.h
	@[ -n "$(NO_DOXYGEN)" ] || ( cd doc && doxygen && touch $(@F) )


if HAVE_DOXYGEN

# Or everything (by default)
all-local: docs

install-data-local: install-docs

clean-local:
	$(RM) -r doc/html
	$(RM) -r doc/latex
	$(RM) doc/doxygen-build.stamp

uninstall-local:
	$(RM) -r $(DESTDIR)$(docdir)/html

endif

# We check this every time make is run, with configure.ac being touched to
# trigger an update of the build system files if update_version changes the
# current PACKAGE_VERSION (or if package_version was modified manually by a
# user with either AUTO_UPDATE=no or no update_version script present - the
# latter being the normal case for tarball releases).
#
# We can't just add the package_version file to CONFIGURE_DEPENDENCIES since
# simply running autoconf will not actually regenerate configure for us when
# the content of that file changes (due to autoconf dependency checking not
# knowing about that without us creating yet another file for it to include).
#
# The MAKECMDGOALS check is a gnu-make'ism, but will degrade 'gracefully' for
# makes that don't support it.  The only loss of functionality is not forcing
# an update of package_version for `make dist` if AUTO_UPDATE=no, but that is
# unlikely to be a real problem for any real user.
$(top_srcdir)/configure.ac: force
	@case "$(MAKECMDGOALS)" in \
	    dist-hook)                             exit 0       ;; \
	    dist-* | dist | distcheck | distclean) _arg=release ;; \
	esac; \
	if ! $(top_srcdir)/update_version $$_arg 2> /dev/null; then \
	    if [ ! -e $(top_srcdir)/package_version ]; then \
		echo 'PACKAGE_VERSION="unknown"' > $(top_srcdir)/package_version; \
	    fi; \
	    . $(top_srcdir)/package_version || exit 1; \
	    [ "$(PACKAGE_VERSION)" != "$$PACKAGE_VERSION" ] || exit 0; \
	fi; \
	touch $@

force:

# Create a minimal package_version file when make dist is run.
dist-hook:
	echo 'PACKAGE_VERSION="$(PACKAGE_VERSION)"' > $(top_distdir)/package_version


.PHONY: lpcnet install-lpcnet docs install-docs
