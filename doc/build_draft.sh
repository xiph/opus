#!/bin/sh

#Stop on errors
set -e
#Set the CWD to the location of this script
[ -n "${0%/*}" ] && cd "${0%/*}"

toplevel=".."
destdir="opus_source"

echo packaging source code
rm -rf "${destdir}"
mkdir "${destdir}"
mkdir "${destdir}/src"
mkdir "${destdir}/silk"
mkdir "${destdir}/silk/float"
mkdir "${destdir}/silk/fixed"
mkdir "${destdir}/celt"
for f in `cat "${toplevel}"/opus_sources.mk "${toplevel}"/celt_sources.mk \
 "${toplevel}"/silk_sources.mk "${toplevel}"/opus_headers.txt \
 "${toplevel}"/celt_headers.txt "${toplevel}"/silk_headers.txt \
 | grep '\.[ch]' | sed -e 's/^.*=//' -e 's/\\\\//'` ; do
  cp -a "${toplevel}/${f}" "${destdir}/${f}"
done
cp -a "${toplevel}"/src/test_opus.c "${destdir}"/src/
cp -a "${toplevel}"/src/opus_compare.c "${destdir}"/src/
cp -a "${toplevel}"/celt/test_opus_custom.c "${destdir}"/celt/
cp -a "${toplevel}"/celt/opus_custom.h "${destdir}"/celt/
cp -a "${toplevel}"/Makefile.draft "${destdir}"/Makefile
cp -a "${toplevel}"/opus_sources.mk "${destdir}"/
cp -a "${toplevel}"/celt_sources.mk "${destdir}"/
cp -a "${toplevel}"/silk_sources.mk "${destdir}"/
cp -a "${toplevel}"/README.draft "${destdir}"/README
cp -a "${toplevel}"/COPYING "${destdir}"/COPYING

tar czf opus_source.tar.gz "${destdir}"
echo building base64 version
cat opus_source.tar.gz| base64 | tr -d '\n' | fold -w 64 | sed 's/^/###/' > opus_source.base64

#echo '<figure>' > opus_compare_escaped.c
#echo '<artwork>' >> opus_compare_escaped.c
#echo '<![CDATA[' >> opus_compare_escaped.c
#cat opus_compare.c >> opus_compare_escaped.c
#echo ']]>' >> opus_compare_escaped.c
#echo '</artwork>' >> opus_compare_escaped.c
#echo '</figure>' >> opus_compare_escaped.c

echo running xml2rfc
xml2rfc draft-ietf-codec-opus.xml draft-ietf-codec-opus.html &
xml2rfc draft-ietf-codec-opus.xml
wait
