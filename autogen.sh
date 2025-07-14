#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

dnn/download_model.sh "6985a6e7c1b1ea6a186d436c70a33dfec2256dca4c6249e9f176b3e2bfe99180"

echo "Updating build configuration files, please wait...."

autoreconf -isf
