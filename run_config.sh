#!/bin/bash

PATH="$PATH:/usr/local/bin:/opt/homebrew/bin"

DIR="$1"
CC="$2"
CFLAGS="$3"
OS="$4"
ARCH="$5"
CCACHE="$6"
NE10LIB="$7"
NE10INC="$8"
ANDROID_TOOLCHAIN_PREFIX="$9"
LDFLAGS="${10}"

CONFIG_OPTS="--disable-doc --disable-extra-programs --disable-shared --enable-intrinsics"

if [ "x$OS" = "xios" -o "x$OS" = "xandroid" ]; then
  # CAR-485: Use NE10 on both iOS & Android, but the benefit doesn't seem huge
  CONFIG_OPTS="$CONFIG_OPTS --with-NE10-includes=$NE10INC --with-NE10-libraries=$NE10LIB"

  if [ "x$OS" = "xios" ]; then
    # run-time CPU support doesn't work on iOS, so we disable (--disable-rtcd)
    CONFIG_OPTS="$CONFIG_OPTS --disable-rtcd"
  fi
fi

if [ "x$OS" != "xios" ]; then
  # Enable fixed point for non-iOS platforms
  CONFIG_OPTS="$CONFIG_OPTS --enable-fixed-point"
fi

echo "CC: ${CC}"
echo "CFLAGS: ${CFLAGS}"
echo "LDLAGS: ${LDFLAGS}"
echo "OS: ${OS}"
echo "ARCH: ${ARCH}"

if [ "$OS" = "android" ] ; then
  if [ "x$ARCH" = "xx86" ] ; then
    CONFIG_OPTS="${CONFIG_OPTS} --host=i686-linux-android"
  elif [ "x$ARCH" = "xx86_64" ]; then
    CONFIG_OPTS="${CONFIG_OPTS} --host=x86_64-linux-android"
  elif [ "x$ARCH" = "xarmv7" ]; then
    CONFIG_OPTS="${CONFIG_OPTS} --host=arm-linux-androideabi"
  elif [ "x$ARCH" = "xarm64" ]; then
    CONFIG_OPTS="${CONFIG_OPTS} --host=aarch64-linux-android"
  else
    echo "ERROR: Unsupported arch: ${ARCH}"
    exit 1
  fi

  export STRIP=${ANDROID_TOOLCHAIN_PREFIX}strip
  export AR=${ANDROID_TOOLCHAIN_PREFIX}ar
  export LD=${ANDROID_TOOLCHAIN_PREFIX}ld
  export RANLIB=${ANDROID_TOOLCHAIN_PREFIX}ranlib
  export NM=${ANDROID_TOOLCHAIN_PREFIX}nm

  export CPP="${CC} -E"
  export CPPFLAGS=${CFLAGS}

elif [ "x$OS" = "xosx" ] ; then
  OS=darwin12
  CFLAGS="${CFLAGS} -arch x86_64"
  LDFLAGS="${LDFLAGS} -arch x86_64"
elif [ "x$OS" = "xios" ] ; then
  OS=darwin
  if [ "x$ARCH" = "xi386" -o "x$ARCH" = "xx86_64" ] ; then
    SDK="iphonesimulator"
    OS=darwin12
    CONFIG_OPTS="$CONFIG_OPTS --host=x86_64"
  else
    SDK="iphoneos"
    CONFIG_OPTS="${CONFIG_OPTS} --host=arm-apple-darwin"
  fi
  CC="xcrun -sdk ${SDK} clang -arch ${ARCH}"
  if [ "x$CCACHE" != "x" ]; then
    CC="$CCACHE $CC"
  fi
fi

export CC
export CFLAGS
export LDFLAGS

# DEVOPS-1227: Using log file so Xcode doesn't parse output and incorrectly
# think it's an error
echo "Running autogen, log file: ${PWD}/autogen.log"
${DIR}/autogen.sh > ./autogen.log 2>&1

echo "Running configure"
${DIR}/configure ${CONFIG_OPTS}

rm -f ${DIR}/test-driver
