#!/bin/sh
set -e

model=lpcnet_data-$1.tar.gz

if [ ! -f $model ]; then
        echo "Downloading latest model"
        wget https://media.xiph.org/lpcnet/data/$model
fi
tar xvof $model
touch src/nnet_data.[ch]
touch src/plc_data.[ch]
mv src/*.[ch] .
