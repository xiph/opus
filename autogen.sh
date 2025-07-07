#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

dnn/download_model.sh "8a07d57c4fce6fb30f23b3e0d264004e04f1d7b421f5392ef61543d021a439af"

echo "Updating build configuration files, please wait...."

autoreconf -isf
