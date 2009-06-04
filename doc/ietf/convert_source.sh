#!/bin/sh

mkdir -p source

for i in `ls ../../libcelt | grep '\.[ch]$'`
do

#echo "<section anchor=\"$i\" title=\"$i\">" > source/$i
#echo '<t>' >> source/$i
#echo '<figure><artwork><![CDATA[' >> source/$i

echo '#include "substitutions.h"' > tata.c
echo 'SOURCE_CODE_BEGIN' >> tata.c
cat ../../libcelt/$i | sed 's/^#/\/\/PREPROCESS_REMOVE#/' >> tata.c
gcc -C -E -nostdinc tata.c | grep -v '^#' | sed 's/\/\/PREPROCESS_REMOVE//' | perl -ne 'if ($begin) {print $_} if (/SOURCE_CODE_BEGIN/) {$begin=1}' > tata2.c
indent --no-tabs -l72 --format-all-comments tata2.c -o tata.c
cat tata.c > source/$i



#indent --no-tabs -l72 --format-all-comments ../../libcelt/$i -o tata.c
#cat tata.c >> source/$i


#echo ']]></artwork></figure>' >> source/$i
#echo '</t>' >> source/$i
#echo '</section>' >> source/$i

done

cat arch.h > source/arch.h
cat celt_types.h > source/celt_types.h
cat config.h > source/config.h
rm source/mfrng*.c
rm source/dump_modes*
rm source/header*
rm source/fixed*

