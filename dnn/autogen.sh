#!/bin/sh
# Run this to set up the build system: configure, makefiles, etc.
set -e

srcdir=`dirname $0`
test -n "$srcdir" && cd "$srcdir"

#SHA1 of the first commit compatible with the current model
commit=7d8d216

if [ ! -f lpcnet_data-$commit.tar.gz ]; then
	echo "Downloading latest model"
	wget https://media.xiph.org/lpcnet/data/lpcnet_data-$commit.tar.gz
fi
tar xvf lpcnet_data-$commit.tar.gz

echo "Updating build configuration files for lpcnet, please wait...."

autoreconf -isf
