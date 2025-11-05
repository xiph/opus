#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

dnn/download_model.sh "a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e"

echo "Updating build configuration files, please wait...."

autoreconf -isf
