#!/bin/sh

rm logs_mono.txt
rm logs_stereo.txt

if [ "$#" -ne "3" ]; then
    echo "usage: run_vectors.sh <exec path> <vector path> <rate>"
    exit 1
fi

CMD_PATH=$1
VECTOR_PATH=$2
RATE=$3

OPUS_DEMO=$CMD_PATH/opus_demo
OPUS_COMPARE=$CMD_PATH/opus_compare

if [ -d $VECTOR_PATH ]; then
    echo Test vectors found in $VECTOR_PATH
else
    echo No test vectors found
    #Don't make the test fail here because the test vectors
    #will be distributed separately
    exit 0
fi

if [ -x $OPUS_DEMO ]; then
    echo Decoding with $OPUS_DEMO
else
    echo ERROR: Decoder not found: $OPUS_DEMO
    exit 1
fi

echo "=============="
echo Testing mono
echo "=============="
echo

for file in 01 02 03 04 05 06 07 08 09 10 11 12
do
    if [ -e $VECTOR_PATH/testvector$file.bit ]; then
        echo Testing testvector$file
    else
        echo Bitstream file not found: testvector$file.bit
    fi
    if $OPUS_DEMO -d $RATE 1 $VECTOR_PATH/testvector$file.bit tmp.out >> logs_mono.txt 2>&1; then
        echo successfully decoded
    else
        echo ERROR: decoding failed
        exit 1
    fi
    $OPUS_COMPARE -r $RATE $VECTOR_PATH/testvector$file.dec tmp.out >> logs_mono.txt 2>&1
    float_ret=$?
    if [ "$float_ret" -eq "0" ]; then
        echo output matches reference
    else
        echo ERROR: output does not match reference
        exit 1
    fi
    echo
done

echo "=============="
echo Testing stereo
echo "=============="
echo

for file in 01 02 03 04 05 06 07 08 09 10 11 12
do
    if [ -e $VECTOR_PATH/testvector$file.bit ]; then
        echo Testing testvector$file
    else
        echo Bitstream file not found: testvector$file
    fi
    if $OPUS_DEMO -d $RATE 2 $VECTOR_PATH/testvector$file.bit tmp.out >> logs_stereo.txt 2>&1; then
        echo successfully decoded
    else
        echo ERROR: decoding failed
        exit 1
    fi
    $OPUS_COMPARE -s -r $RATE $VECTOR_PATH/testvector$file.dec tmp.out >> logs_stereo.txt 2>&1
    float_ret=$?
    if [ "$float_ret" -eq "0" ]; then
        echo output matches reference
    else
        echo ERROR: output does not match reference
        exit 1
    fi
    echo
done



echo All tests have passed successfully
grep quality logs_mono.txt | awk '{sum+=$4}END{print "Average mono quality is", sum/NR, "%"}'
grep quality logs_stereo.txt | awk '{sum+=$4}END{print "Average stereo quality is", sum/NR, "%"}'
