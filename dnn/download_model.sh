#!/bin/sh
set -e

model=opus_data-$1.tar.gz

if [ ! -f $model ]; then
   echo "Downloading latest model"
   if command -v wget >/dev/null; then
      wget -O $model https://media.xiph.org/opus/models/$model
   else
      # if wget is not available use curl
      curl -o $model https://media.xiph.org/opus/models/$model
   fi
fi

if command -v sha256sum >/dev/null; then
   SHA256SUM=sha256sum
elif command -v shasum >/dev/null; then
   SHA256SUM="shasum -a 256"
else
   SHA256SUM=
fi

if [ -n "$SHA256SUM" ]; then
   echo "Validating checksum"
   checksum="$1"
   checksum2=$($SHA256SUM $model | awk '{print $1}')
   if [ "$checksum" != "$checksum2" ]
   then
      echo "Aborting due to mismatching checksums. This could be caused by a corrupted download of $model."
      echo "Consider deleting local copy of $model and running this script again."
      exit 1
   else
      echo "Checksums match"
   fi
else
   echo "Could not find sha256sum or shasum; skipping verification. Please verify manually that sha256 hash of ${model} matches ${1}."
fi

tar xvzomf $model
