#!/bin/sh

mkdir -p xml_source

for i in `ls source/ | grep '\.[ch]$'` Makefile
do

echo "<section anchor=\"$i\" title=\"$i\">" > xml_source/$i
echo '<t>' >> xml_source/$i
echo '<figure><artwork><![CDATA[' >> xml_source/$i

cat source/$i >> xml_source/$i

echo ']]></artwork></figure>' >> xml_source/$i
echo '</t>' >> xml_source/$i
echo '</section>' >> xml_source/$i

done