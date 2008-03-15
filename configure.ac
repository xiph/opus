dnl Process this file with autoconf to produce a configure script. -*-m4-*-

AC_INIT(libcelt/arch.h)

AM_CONFIG_HEADER([config.h])

CELT_MAJOR_VERSION=0
CELT_MINOR_VERSION=3
CELT_MICRO_VERSION=0
CELT_EXTRA_VERSION=
CELT_VERSION=$CELT_MAJOR_VERSION.$CELT_MINOR_VERSION.$CELT_MICRO_VERSION$CELT_EXTRA_VERSION

CELT_LT_CURRENT=0
CELT_LT_REVISION=0
CELT_LT_AGE=0

AC_SUBST(CELT_LT_CURRENT)
AC_SUBST(CELT_LT_REVISION)
AC_SUBST(CELT_LT_AGE)

# For automake.
VERSION=$CELT_VERSION
PACKAGE=celt

AC_SUBST(CELT_VERSION)

AM_INIT_AUTOMAKE($PACKAGE, $VERSION, no-define)
AM_MAINTAINER_MODE

AC_CANONICAL_HOST
AM_PROG_LIBTOOL

AC_C_BIGENDIAN
AC_C_CONST
AC_C_INLINE
AC_C_RESTRICT


AC_MSG_CHECKING(for C99 variable-size arrays)
AC_TRY_COMPILE( , [
int foo=10;
int array[foo];
],
[has_var_arrays=yes;AC_DEFINE([VAR_ARRAYS], [], [Use C99 variable-size arrays])
],
has_var_arrays=no
)
AC_MSG_RESULT($has_var_arrays)

AC_CHECK_HEADERS([alloca.h getopt.h])
AC_MSG_CHECKING(for alloca)
AC_TRY_COMPILE( [#include <alloca.h>], [
int foo=10;
int *array = alloca(foo);
],
[
has_alloca=yes;
if test x$has_var_arrays = "xno" ; then
AC_DEFINE([USE_ALLOCA], [], [Make use of alloca])
fi
],
has_alloca=no
)
AC_MSG_RESULT($has_alloca)


AC_CHECK_HEADERS(sys/soundcard.h sys/audioio.h)

XIPH_PATH_OGG([tools="tools"], [tools=""])
AC_SUBST(tools)

AC_CHECK_LIB(m, sin)

# Check for getopt_long; if not found, use included source.
AC_CHECK_FUNCS([getopt_long],,
[# FreeBSD has a gnugetopt library.
  AC_CHECK_LIB([gnugetopt],[getopt_long],
[AC_DEFINE([HAVE_GETOPT_LONG])],
[# Use the GNU replacement.
AC_LIBOBJ(getopt)
AC_LIBOBJ(getopt1)])])

AC_CHECK_LIB(winmm, main)

AC_DEFINE_UNQUOTED(CELT_VERSION, "${CELT_VERSION}", [Complete version string])
AC_DEFINE_UNQUOTED(CELT_MAJOR_VERSION, ${CELT_MAJOR_VERSION}, [Version major])
AC_DEFINE_UNQUOTED(CELT_MINOR_VERSION, ${CELT_MINOR_VERSION}, [Version minor])
AC_DEFINE_UNQUOTED(CELT_MICRO_VERSION, ${CELT_MICRO_VERSION}, [Version micro])
AC_DEFINE_UNQUOTED(CELT_EXTRA_VERSION, "${CELT_EXTRA_VERSION}", [Version extra])

AC_ARG_ENABLE(fixed-point, [  --enable-fixed-point    Compile as fixed-point],
[if test "$enableval" = yes; then
  AC_DEFINE([FIXED_POINT], , [Compile as fixed-point])
  AC_DEFINE([DOUBLE_PRECISION], , [Compile as fixed-point])
  AC_DEFINE([MIXED_PRECISION], , [Compile as fixed-point])
else
  AC_DEFINE([FLOATING_POINT], , [Compile as floating-point])
fi],
AC_DEFINE([FLOATING_POINT], , [Compile as floating-point]))

AC_ARG_ENABLE(fixed-point-debug, [  --enable-fixed-point-debug  Debug fixed-point implementation],
[if test "$enableval" = yes; then
  AC_DEFINE([FIXED_DEBUG], , [Debug fixed-point implementation])
fi])

AC_ARG_ENABLE(static-modes, [  --enable-static-modes],
[if test "$enableval" = yes; then
  AC_DEFINE([STATIC_MODES], , [Static modes])
fi])

AC_ARG_ENABLE(assertions, [  --enable-assertions],
[if test "$enableval" = yes; then
  AC_DEFINE([ENABLE_ASSERTIONS], , [Assertions])
fi])

if test $ac_cv_c_compiler_gnu = yes ; then
        CFLAGS="$CFLAGS -fvisibility=hidden -W -Wstrict-prototypes -Wall -Wextra -Wcast-align -Wnested-externs -Wshadow -Wno-parentheses -Wno-unused-parameter -Wno-sign-compare"
        #CFLAGS="$CFLAGS -fvisibility=hidden -W -Wstrict-prototypes -Wmissing-prototypes -Wall -Waggregate-return -Wcast-align -Wcast-qual -Wnested-externs -Wshadow -Wno-parentheses"
fi

AC_CHECK_SIZEOF(short)
AC_CHECK_SIZEOF(int)
AC_CHECK_SIZEOF(long)
AC_CHECK_SIZEOF(long long)

if test x$has_char16 = "xyes" ; then
        case 1 in
                $ac_cv_sizeof_short) SIZE16="short";;
                $ac_cv_sizeof_int) SIZE16="int";;
        esac
else
        case 2 in
                $ac_cv_sizeof_short) SIZE16="short";;
                $ac_cv_sizeof_int) SIZE16="int";;
        esac
fi

if test x$has_char16 = "xyes" ; then
        case 2 in
                $ac_cv_sizeof_int) SIZE32="int";;
                $ac_cv_sizeof_long) SIZE32="long";;
                $ac_cv_sizeof_short) SIZE32="short";;
        esac
else
        case 4 in
                $ac_cv_sizeof_int) SIZE32="int";;
                $ac_cv_sizeof_long) SIZE32="long";;
                $ac_cv_sizeof_short) SIZE32="short";;
        esac
fi

AC_SUBST(SIZE16)
AC_SUBST(SIZE32)

AC_OUTPUT([Makefile libcelt/Makefile tests/Makefile celt.pc tools/Makefile])

if test "x$tools" = "x"; then 
echo "**IMPORTANT**"
echo "You don't seem to have the development package for libogg (libogg-devel) installed. Only the library will be built (no encoder/decoder executable)"
echo "You can download libogg from http://www.vorbis.com/download.psp"
fi

echo "Type \"make; make install\" to compile and install";
