#!/bin/sh

if [ "$#" -ne "2" ]; then
    echo "usage: dred_vectors.sh <exec path> <vector path>"
    exit 1
fi

CMD_PATH=$1
VECTOR_PATH=$2

: ${FARGAN_DEMO:=$CMD_PATH/fargan_demo}
: ${OPUS_DEMO:=$CMD_PATH/opus_demo}
: ${DRED_COMPARE:=$CMD_PATH/dred_compare}

if [ -d "$VECTOR_PATH" ]; then
    echo "Test vectors found in $VECTOR_PATH"
else
    echo "No test vectors found"
    #Don't make the test fail here because the test vectors
    #will be distributed separately
    exit 0
fi

if [ ! -x "$DRED_COMPARE" ]; then
    echo "ERROR: Compare program not found: $DRED_COMPARE"
    exit 1
fi

if [ -x "$FARGAN_DEMO" ]; then
    echo "Decoding with $FARGAN_DEMO"
else
    echo "ERROR: Decoder not found: $FARGAN_DEMO"
    exit 1
fi

echo "=============="
echo "Testing DRED decoding"
echo "=============="
echo

for i in 1 2 3 4 5 6 7 8
do
    if [ -e "$VECTOR_PATH/vector${i}_dred.bit" ]; then
        echo "Testing vector${i}_dred.bit"
    else
        echo "Bitstream file not found: vector${i}_dred.bit"
    fi
    if "$FARGAN_DEMO" -dred-decoding "$VECTOR_PATH/vector${i}_dred.bit" tmp.f32 >> logs_dred_decode.txt 2>&1; then
        echo "successfully decoded"
    else
        echo "ERROR: decoding failed"
        exit 1
    fi
    "$DRED_COMPARE" -features -thresholds .5 .15 .02 "$VECTOR_PATH/vector${i}_dred_dec.f32" tmp.f32 >> logs_dred_decode.txt 2>&1
    float_ret=$?
    if [ "$float_ret" -eq "0" ]; then
        echo "output matches reference"
    else
        echo "ERROR: DRED decoder output does not match reference"
        exit 1
    fi
    echo
done

for i in 1 2 3 4 5 6 7 8
do
    if [ -e "$VECTOR_PATH/vector${i}_features.f32" ]; then
        echo "Testing vector${i}_features.f32"
    else
        echo "Bitstream file not found: vector${i}_features.f32"
    fi
    if "$FARGAN_DEMO" -fargan-synthesis "$VECTOR_PATH/vector${i}_features.f32" tmp.sw >> logs_dred_synthesis.txt 2>&1; then
        echo "successfully decoded"
    else
        echo "ERROR: decoding failed"
        exit 1
    fi
    "$DRED_COMPARE" -audio -thresholds 0.25 1.0 0.15 "$VECTOR_PATH/vector${i}_orig.sw" tmp.sw >> logs_dred_synthesis.txt 2>&1
    float_ret=$?
    if [ "$float_ret" -eq "0" ] ; then
        echo "output matches reference"
    else
        echo "ERROR: vocoder output does not match reference"
        exit 1
    fi
    echo
done

for i in 1 2 3 4 5 6 7 8
do
    if [ -e "$VECTOR_PATH/vector${i}_opus.bit" ]; then
        echo "Testing vector${i}_opus.bit"
    else
        echo "Bitstream file not found: vector${i}_opus.bit"
    fi
    if "$OPUS_DEMO" -d 16000 1 "$VECTOR_PATH/vector${i}_opus.bit" tmp.sw >> logs_dred_opus.txt 2>&1; then
        echo "successfully decoded"
    else
        echo "ERROR: decoding failed"
        exit 1
    fi
    "$DRED_COMPARE" -audio -thresholds 1.0 3.0 0.25 "$VECTOR_PATH/vector${i}_orig.sw" tmp.sw >> logs_dred_opus.txt 2>&1
    float_ret=$?
    if [ "$float_ret" -eq "0" ] ; then
        echo "output matches reference"
    else
        echo "WARNING: encoder output is not close enough to reference. This could be a bug, but it does not prevent compliance"
        echo
        echo "Conformance tests passed"
        exit 1
    fi
    echo
done

echo
echo "Conformance tests passed"
