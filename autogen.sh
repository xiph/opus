#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

dnn/download_model.sh "a86f0a9db852691d4335608733ec8384a407e585801ab9e4b490e0be297ac382"

echo "Updating build configuration files, please wait...."

autoreconf -isf
