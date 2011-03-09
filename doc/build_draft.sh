#!/bin/sh

echo packaging source code
rm -rf opus_source
cat opus_sources.mk celt_sources.mk silk_sources.mk opus_headers.txt celt_headers.txt silk_headers.txt | grep '\.[ch]' | sed -e 's/^.*=//' -e 's/\\//' > all_files.txt
cat opus_extra.txt >> all_files.txt
tar czf tmp_draft.tar.gz `cat all_files.txt`

mkdir opus_source
cd opus_source
tar xzf ../tmp_draft.tar.gz
cp ../Makefile.draft Makefile
cp ../opus_sources.mk .
cp ../celt_sources.mk .
cp ../silk_sources.mk .

cd ..
tar czf opus_source.tar.gz opus_source
echo building base64 version
cat opus_source.tar.gz| base64 -w 66 | sed 's/^/###/' > doc/opus_source.base64

cd doc
echo '<figure>' > opus_compare_escaped.m
echo '<artwork>' >> opus_compare_escaped.m
echo '<![CDATA[' >> opus_compare_escaped.m
cat opus_compare.m >> opus_compare_escaped.m
echo ']]>' >> opus_compare_escaped.m
echo '</artwork>' >> opus_compare_escaped.m
echo '</figure>' >> opus_compare_escaped.m

echo running xml2rfc
xml2rfc draft-ietf-codec-opus.xml
