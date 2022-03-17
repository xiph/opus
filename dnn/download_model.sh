#!/bin/sh
set -e

model=$1.tar.gz

if [ ! -f $model ]; then
        echo "Downloading latest model"
        wget https://media.xiph.org/lpcnet/data/plc_challenge/$model
fi
tar xvf $model
touch src/nnet_data.[ch]
touch src/plc_data.[ch]
mv src/*.[ch] .
