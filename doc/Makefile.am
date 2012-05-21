## Process this file with automake to produce Makefile.in

DOCINPUTS = $(top_srcdir)/include/opus.h \
            $(top_srcdir)/include/opus_multistream.h \
            $(top_srcdir)/include/opus_defines.h \
            $(top_srcdir)/include/opus_types.h

EXTRA_DIST = Doxyfile.in

all-local: doxygen-build.stamp


if HAVE_DOXYGEN
doxygen-build.stamp: Doxyfile $(DOCINPUTS)
	doxygen
	touch $@

install-data-local:
	for f in `find html -type f \! -name "installdox"`; do	\
		$(INSTALL_DATA) -D $$f $(DESTDIR)$(docdir)/$$f;	\
	done

	$(INSTALL) -d $(DESTDIR)$(mandir)/man3
	cd man && find man3 -type f \
		  -exec $(INSTALL_DATA) \{} $(DESTDIR)$(mandir)/man3 \;
else
doxygen-build.stamp: Doxyfile $(DOCINPUTS)
	@echo "*** Warning: Doxygen not found; API documentation will not be built."
	touch $@
endif


# delete doxygen subdirs
clean-local:
	$(RM) -r html
	$(RM) -r latex
	$(RM) -r man
	$(RM) doxygen-build.stamp

uninstall-local:
	$(RM) -r $(DESTDIR)$(docdir)/html
	$(RM) $(DESTDIR)$(mandir)/man3/opus_*.3 $(DESTDIR)$(mandir)/man3/opus.h.3
