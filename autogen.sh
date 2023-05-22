#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

git submodule update --init
(cd lpcnet; ./download_model.sh 399be7c)

echo "Updating build configuration files, please wait...."

autoreconf -isf
