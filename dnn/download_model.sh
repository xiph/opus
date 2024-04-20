#!/bin/sh
set -e

model=opus_data-$1.tar.gz

if [ ! -f $model ]; then
        echo "Downloading latest model"
        wget https://media.xiph.org/opus/models/$model
fi

SHA256=$(command -v sha256sum)
if [ "$?" != "0" ]
then
   echo "Could not find sha256 sum. Skipping verification. Please verify manually that sha256 hash of ${model} matches ${1}."
else
   echo "Validating checksum"
   checksum=$1
   checksum2=$(sha256sum $model | awk '{print $1}')
   if [ "$checksum" != "$checksum2" ]
   then
      echo "checksums don't match, aborting"
      exit 1
   else
      echo "checksums match"
   fi

fi



tar xvomf $model
