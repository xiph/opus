#!/bin/sh

cd datasets

#parallel -j +2 'unzip -n {}' ::: *.zip

find . -name "*.wav" | parallel -k -j 20 'sox --no-dither {} -t sw -r 16000 -c 1 -' > ../all_speech.sw
