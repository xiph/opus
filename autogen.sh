#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

dnn/download_model.sh "4ed9445b96698bad25d852e912b41495ddfa30c8dbc8a55f9cde5826ed793453"

echo "Updating build configuration files, please wait...."

autoreconf -isf
