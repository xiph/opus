#!/bin/sh

mkdir -p source

for i in `ls ../../libcelt | grep '\.[ch]$'`
do

#echo "<section anchor=\"$i\" title=\"$i\">" > source/$i
#echo '<t>' >> source/$i
#echo '<figure><artwork><![CDATA[' >> source/$i

echo '#include "substitutions.h"' > tata.c
echo 'SOURCE_CODE_BEGIN' >> tata.c

if echo $i | grep '\.h' > /dev/null; then
	cat ../../libcelt/$i | sed -e 's/=\*/= \*/' -e 's/=-/= -/' -e 's/\t/    /g' -e 's/^#/\/\/PREPROCESS_REMOVE#/' >> tata.c
else
	cat ../../libcelt/$i | sed -e 's/=\*/= \*/' -e 's/=-/= -/' -e 's/\t/    /g' -e 's/^#include/\/\/PREPROCESS_REMOVE#include/' | sed 's/^#define/\/\/PREPROCESS_REMOVE#define/'>> tata.c
fi

#cat ../../libcelt/$i | sed 's/^#/\/\/PREPROCESS_REMOVE#/' >> tata.c
#cat ../../libcelt/$i | sed 's/^#include/\/\/PREPROCESS_REMOVE#include/' | sed 's/^#define/\/\/PREPROCESS_REMOVE#define/'>> tata.c
gcc -DHAVE_CONFIG_H -C -E -nostdinc tata.c | grep -v '^#' | sed 's/\/\/PREPROCESS_REMOVE//' | perl -ne 'if ($begin) {print $_} if (/SOURCE_CODE_BEGIN/) {$begin=1}' > tata2.c

#cat ../../libcelt/$i >> tata.c
#gcc -C -E -nostdinc tata.c -fdirectives-only | perl -ne 'if ($begin) {print $_} if (/SOURCE_CODE_BEGIN/) {$begin=1}' > tata2.c

indent -nsc -ncdb -original -sob -i2 -bl -bli0 --no-tabs -l69 --format-all-comments tata2.c -o tata.c
cat tata.c | grep -v 'include.*float_cast' | ./wrap_lines > source/$i
#cat tata.c  > source/$i



#indent --no-tabs -l72 --format-all-comments ../../libcelt/$i -o tata.c
#cat tata.c >> source/$i


#echo ']]></artwork></figure>' >> source/$i
#echo '</t>' >> source/$i
#echo '</section>' >> source/$i

done

cp arch.h source/arch.h
cp celt_types.h source/celt_types.h
cp config.h source/config.h
cp Makefile.ietf source/Makefile

rm source/mfrng*.c
rm source/dump_modes*
rm source/header*
rm source/fixed*

