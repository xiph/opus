#!/bin/sh

if [ "$#" -ne "2" ]; then
    echo "usage: run_vectors.sh <exec path> <vector path>"
    exit 1
fi

CMD_PATH=$1
VECTOR_PATH=$2

OPUS_DEMO=$CMD_PATH/opus_demo
OPUS_COMPARE=$CMD_PATH/opus_compare

if [ -d $VECTOR_PATH ]; then
    echo Test vectors found in $VECTOR_PATH
else
    echo No test vectors found
    #Don't make the test fail here because the test vectors will be 
    #distributed separateyl
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

for file in test1_mono test2_mono test3_mono test4_mono test5_mono
do
    if [ -e $VECTOR_PATH/$file.bit ]; then
        echo Testing $file
    else 
        echo Bitstream file not found: $file
    fi
    if $OPUS_DEMO -d 48000 1 $VECTOR_PATH/$file.bit tmp.out > /dev/null 2>&1; then
        echo successfully decoded
    else
        echo ERROR: decoding failed
        exit 1
    fi
    $OPUS_COMPARE $VECTOR_PATH/$file.float tmp.out > /dev/null 2>&1
    float_ret=$?
    $OPUS_COMPARE $VECTOR_PATH/$file.fixed tmp.out > /dev/null 2>&1
    fixed_ret=$?
    if [ "$float_ret" -eq "0" -o "$fixed_ret" -eq "0" ]; then
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

for file in test1_stereo test2_stereo test3_stereo test4_stereo
do
    if [ -e $VECTOR_PATH/$file.bit ]; then
        echo Testing $file
    else 
        echo Bitstream file not found: $file
    fi
    if $OPUS_DEMO -d 48000 2 $VECTOR_PATH/$file.bit tmp.out > /dev/null 2>&1; then
        echo successfully decoded
    else
        echo ERROR: decoding failed
        exit 1
    fi
    $OPUS_COMPARE -s $VECTOR_PATH/$file.float tmp.out > /dev/null 2>&1
    float_ret=$?
    $OPUS_COMPARE -s $VECTOR_PATH/$file.fixed tmp.out > /dev/null 2>&1
    fixed_ret=$?
    if [ "$float_ret" -eq "0" -o "$fixed_ret" -eq "0" ]; then
        echo output matches reference
    else
        echo ERROR: output does not match reference
        exit 1
    fi
    echo
done



echo All tests have passed successfully
