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
mkdir "${destdir}/include"
for f in `cat "${toplevel}"/opus_sources.mk "${toplevel}"/celt_sources.mk \
 "${toplevel}"/silk_sources.mk "${toplevel}"/opus_headers.mk \
 "${toplevel}"/celt_headers.mk "${toplevel}"/silk_headers.mk \
 | grep '\.[ch]' | sed -e 's/^.*=//' -e 's/\\\\//'` ; do
  cp -a "${toplevel}/${f}" "${destdir}/${f}"
done
cp -a "${toplevel}"/src/opus_demo.c "${destdir}"/src/
cp -a "${toplevel}"/src/opus_compare.c "${destdir}"/src/
cp -a "${toplevel}"/celt/opus_custom_demo.c "${destdir}"/celt/
cp -a "${toplevel}"/Makefile.draft "${destdir}"/Makefile
cp -a "${toplevel}"/opus_sources.mk "${destdir}"/
cp -a "${toplevel}"/celt_sources.mk "${destdir}"/
cp -a "${toplevel}"/silk_sources.mk "${destdir}"/
cp -a "${toplevel}"/README.draft "${destdir}"/README
cp -a "${toplevel}"/COPYING "${destdir}"/COPYING
cp -a "${toplevel}"/tests/run_vectors.sh "${destdir}"/

GZIP=-9 tar --owner=root --group=root --format=v7 -czf opus_source.tar.gz "${destdir}"
echo building base64 version
cat opus_source.tar.gz| base64 | tr -d '\n' | fold -w 64 | \
 sed -e 's/^/\<spanx style="vbare"\>###/' -e 's/$/\<\/spanx\>\<vspace\/\>/' > \
 opus_source.base64


#echo '<figure>' > opus_compare_escaped.c
#echo '<artwork>' >> opus_compare_escaped.c
#echo '<![CDATA[' >> opus_compare_escaped.c
#cat opus_compare.c >> opus_compare_escaped.c
#echo ']]>' >> opus_compare_escaped.c
#echo '</artwork>' >> opus_compare_escaped.c
#echo '</figure>' >> opus_compare_escaped.c

if [[ ! -d ../opus_testvectors ]] ; then
  echo "Downloading test vectors..."
  wget 'http://opus-codec.org/testvectors/opus_testvectors.tar.gz'
  tar -C .. -xvzf opus_testvectors.tar.gz
fi
echo '<figure>' > testvectors_sha1
echo '<artwork>' >> testvectors_sha1
echo '<![CDATA[' >> testvectors_sha1
(cd ../opus_testvectors; sha1sum *.bit *.dec) >> testvectors_sha1
#cd opus_testvectors
#sha1sum *.bit *.dec >> ../testvectors_sha1
#cd ..
echo ']]>' >> testvectors_sha1
echo '</artwork>' >> testvectors_sha1
echo '</figure>' >> testvectors_sha1

echo running xml2rfc
xml2rfc draft-ietf-codec-opus.xml draft-ietf-codec-opus.html &
xml2rfc draft-ietf-codec-opus.xml
wait
