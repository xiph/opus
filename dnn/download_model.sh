#!/bin/sh
set -e

model=opus_data-$1.tar.gz

if [ ! -f $model ]; then
        echo "Downloading latest model"
        wget https://media.xiph.org/opus/models/$model
fi
tar xvomf $model
