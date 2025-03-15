#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

dnn/download_model.sh "4ec556dd87e63c17c4a805c40685ef3fe1fad7c8b26b123f2ede553b50158cb1"

echo "Updating build configuration files, please wait...."

autoreconf -isf
