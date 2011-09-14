## Process this file with automake to produce Makefile.in

DOCINPUTS = $(top_srcdir)/src/opus.h \
            $(top_srcdir)/src/opus_multistream.h \
            $(top_srcdir)/celt/opus_defines.h \
            $(top_srcdir)/celt/opus_types.h

doc_DATA = doxygen-build.stamp

EXTRA_DIST = Doxyfile.in

if HAVE_DOXYGEN
doxygen-build.stamp: Doxyfile $(DOCINPUTS)
	doxygen
	touch $@
else
doxygen-build.stamp: Doxyfile $(DOCINPUTS)
	@echo "*** Warning: Doxygen not found; API documentation will not be built."
	touch $@
endif

# delete doxygen subdirs
clean-local:
	-rm -rf html
	-rm -rf latex
	-rm -rf man
	-rm doxygen-build.stamp
