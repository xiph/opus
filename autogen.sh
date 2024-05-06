#!/bin/sh
# Copyright (c) 2010-2015 Xiph.Org Foundation and contributors.
# Use of this source code is governed by a BSD-style license that can be
# found in the COPYING file.

# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

<<<<<<< HEAD
dnn/download_model.sh "66d438011afadd36428794dedf39e7c44843f03f708046f53ce605b8533629ca"
=======
dnn/download_model.sh "160753e983198f29f1aae67c54caa0e30bd90f1ce916a52f15bdad2df8e35e58"
>>>>>>> eb39efaa (Fixing lossgen to work with USE_WEIGHTS_FILE)

echo "Updating build configuration files, please wait...."

autoreconf -isf
