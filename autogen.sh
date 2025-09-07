#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

dnn/download_model.sh "0af99db0c52f470799698431f66cbc83d245dea45c24d6015a958b3f22a20029"

echo "Updating build configuration files, please wait...."

autoreconf -isf
