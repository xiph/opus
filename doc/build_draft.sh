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
for f in `cat "${toplevel}"/opus_sources.mk "${toplevel}"/celt_sources.mk \
 "${toplevel}"/silk_sources.mk "${toplevel}"/opus_headers.txt \
 "${toplevel}"/celt_headers.txt "${toplevel}"/silk_headers.txt \
 | grep '\.[ch]' | sed -e 's/^.*=//' -e 's/\\\\//'` ; do
  cp -a "${toplevel}/${f}" "${destdir}"
done
cp -a "${toplevel}"/Makefile.draft "${destdir}"/Makefile
cp -a "${toplevel}"/opus_sources.mk "${destdir}"/
cp -a "${toplevel}"/celt_sources.mk "${destdir}"/
cp -a "${toplevel}"/silk_sources.mk "${destdir}"/
cp -a "${toplevel}"/README.draft "${destdir}"/README
cp -a "${toplevel}"/COPYING "${destdir}"/COPYING

tar czf opus_source.tar.gz "${destdir}"
echo building base64 version
cat opus_source.tar.gz| base64 -w 66 | sed 's/^/###/' > opus_source.base64

echo '<figure>' > opus_compare_escaped.m
echo '<artwork>' >> opus_compare_escaped.m
echo '<![CDATA[' >> opus_compare_escaped.m
cat opus_compare.m >> opus_compare_escaped.m
echo ']]>' >> opus_compare_escaped.m
echo '</artwork>' >> opus_compare_escaped.m
echo '</figure>' >> opus_compare_escaped.m

echo running xml2rfc
xml2rfc draft-ietf-codec-opus.xml
xml2rfc draft-ietf-codec-opus.xml draft-ietf-codec-opus.html
