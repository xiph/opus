#!/bin/sh

mkdir -p source

for i in `ls ../../libcelt | grep '\.[ch]$'`
do

#cat ../../libcelt/$i | sed -e 's/\&/\&amp;/g' -e 's/</\&lt;/g' -e 's/^/<t>/' -e 's/$/<\/t>/' > source/$i

echo "<section anchor=\"$i\" title=\"$i\">" > source/$i
echo '<t>' >> source/$i
echo '<figure><artwork><![CDATA[' >> source/$i


#cat ../../libcelt/$i >> source/$i
indent --no-tabs -l72 --format-all-comments ../../libcelt/$i -o tata.c
cat tata.c >> source/$i


echo ']]></artwork></figure>' >> source/$i
echo '</t>' >> source/$i
echo '</section>' >> source/$i

done
