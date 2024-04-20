#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

dnn/download_model.sh "8f34305a299183509d22c7ba66790f67916a0fc56028ebd4c8f7b938458f2801"

echo "Updating build configuration files, please wait...."

autoreconf -isf
