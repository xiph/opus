## Process this file with automake to produce Makefile.in. -*-Makefile-*-

# To disable automatic dependency tracking if using other tools than
# gcc and gmake, add the option 'no-dependencies'
AUTOMAKE_OPTIONS = 1.6

#Fools KDevelop into including all files
SUBDIRS = libcelt tests @tools@

DIST_SUBDIRS = libcelt tests tools

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = celt.pc 

EXTRA_DIST = celt.pc.in Doxyfile Doxyfile.devel

rpm: dist
	rpmbuild -ta ${PACKAGE}-${VERSION}.tar.gz
