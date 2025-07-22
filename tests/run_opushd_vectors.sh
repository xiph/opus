#!/bin/sh

# Copyright (c) 2011-2012 Jean-Marc Valin
#
#  This file is extracted from RFC6716. Please see that RFC for additional
#  information.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  - Redistributions of source code must retain the above copyright
#  notice, this list of conditions and the following disclaimer.
#
#  - Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
#  - Neither the name of Internet Society, IETF or IETF Trust, nor the
#  names of specific contributors, may be used to endorse or promote
#  products derived from this software without specific prior written
#  permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
#  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

rm -f logs_mono.txt logs_mono2.txt
rm -f logs_stereo.txt logs_stereo2.txt

if [ "$#" -ne "2" ]; then
    echo "usage: run_vectors.sh <exec path> <vector path>"
    exit 1
fi

CMD_PATH=$1
VECTOR_PATH=$2
RATE=96000

: ${OPUS_DEMO:=$CMD_PATH/opus_demo}
: ${QEXT_COMPARE:=$CMD_PATH/qext_compare}

if [ -d "$VECTOR_PATH" ]; then
    echo "Test vectors found in $VECTOR_PATH"
else
    echo "No test vectors found"
    #Don't make the test fail here because the test vectors
    #will be distributed separately
    exit 0
fi

if [ ! -x "$QEXT_COMPARE" ]; then
    echo "ERROR: Compare program not found: $QEXT_COMPARE"
    exit 1
fi

if [ -x "$OPUS_DEMO" ]; then
    echo "Decoding with $OPUS_DEMO"
else
    echo "ERROR: Decoder not found: $OPUS_DEMO"
    exit 1
fi

echo "============================"
echo "Testing original testvectors"
echo "============================"
echo


for file in 01 02 03 04 05 06 07 08 09 10 11 12
do
    if [ -e "$VECTOR_PATH/testvector$file.bit" ]; then
        echo "Testing testvector$file"
    else
        echo "Bitstream file not found: testvector$file"
    fi
    if "$OPUS_DEMO" -d "$RATE" 2 -ignore_extensions -f32 "$VECTOR_PATH/testvector$file.bit" tmp.out >> logs_stereo.txt 2>&1; then
        echo "successfully decoded"
    else
        echo "ERROR: decoding failed"
        exit 1
    fi
    "$QEXT_COMPARE" -s -r "$RATE" -f32 -thresholds 0.05 .1 .1 "$VECTOR_PATH/testvector${file}_96k.f32" tmp.out >> logs_stereo.txt 2>&1
    if [ "$?" -eq "0" ]; then
        echo "output matches reference"
    else
        echo "ERROR: output does not match reference"
        exit 1
    fi
    echo
done


echo "==========================="
echo "Testing Opus HD testvectors"
echo "==========================="
echo
for file in 01 02 03 04 05 06
do
    if [ -e "$VECTOR_PATH/testvector$file.bit" ]; then
        echo "Testing testvector$file"
    else
        echo "Bitstream file not found: testvector$file"
    fi
    if "$OPUS_DEMO" -d "$RATE" 2 -f32 "$VECTOR_PATH/qext_vector$file.bit" tmp.out >> logs_qext.txt 2>&1; then
        echo "successfully decoded"
    else
        echo "ERROR: decoding failed"
        exit 1
    fi
    "$QEXT_COMPARE" -s -r "$RATE" -f32 -thresholds 0.05 .1 .1 "$VECTOR_PATH/qext_vector${file}dec.f32" tmp.out >> logs_qext.txt 2>&1
    if [ "$?" -eq "0" ]; then
        echo "output matches reference"
    else
        echo "ERROR: output does not match reference"
        exit 1
    fi
    echo
done

echo "=================================="
echo "Testing Opus HD fuzzng testvectors"
echo "=================================="
echo
for file in 01 02 03 04 05 06
do
    if [ -e "$VECTOR_PATH/testvector$file.bit" ]; then
        echo "Testing testvector$file"
    else
        echo "Bitstream file not found: testvector$file"
    fi
    if "$OPUS_DEMO" -d "$RATE" 2 -f32 "$VECTOR_PATH/qext_vector${file}fuzz.bit" tmp.out >> logs_qextfuzz.txt 2>&1; then
        echo "successfully decoded"
    else
        echo "ERROR: decoding failed"
        exit 1
    fi
    "$QEXT_COMPARE" -s -r "$RATE" -f32 -thresholds 0.1 .5 1 "$VECTOR_PATH/qext_vector${file}fuzzdec.f32" tmp.out >> logs_qextfuzz.txt 2>&1
    if [ "$?" -eq "0" ]; then
        echo "output matches reference"
    else
        echo "ERROR: output does not match reference"
        exit 1
    fi
    echo
done



echo "All tests have passed successfully"
