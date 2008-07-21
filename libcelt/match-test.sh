#!/bin/sh

#outfile=`mktemp`
#if [ $? != 0 ]; then
#   echo "count not create temp output file"
#   exit 0
#fi
outfile=/dev/null

if [ -f mono_test_file.sw ]; then
   echo -n "mono test... "
   ./testcelt 44100 1 256 32 mono_test_file.sw $outfile
   if [ $? != 0 ]; then
      exit 1
   fi
else
   echo "no mono test file"
fi

if [ -f stereo_test_file.sw ]; then
   echo -n "stereo test... "
   ./testcelt 44100 2 256 92 stereo_test_file.sw $outfile
   if [ $? != 0 ]; then
      exit 1
   fi
else
   echo "no stereo test file"
fi

exit 0
