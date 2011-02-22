#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.59])
AC_INIT(interface/SKP_Silk_SDK_API.h)
AM_INIT_AUTOMAKE(libSKP_SILK_SDK,1.0.2)

# Checks for programs.
AC_PROG_CC
AC_PROG_LIBTOOL

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([float.h stdint.h stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_LIB([m],[pow])

ac_enable_fixed="no";
AC_ARG_ENABLE(fixed-point, [  --enable-fixed-point    compile as fixed-point],
[if test "$enableval" = yes; then
ac_enable_fixed="yes";
AC_DEFINE([FIXED_POINT], [1], [Compile as fixed-point])
else
ac_enable_fixed="no";
dnl AC_DEFINE([FIXED_POINT], [0], [Compile as floating-point])
fi],
dnl AC_DEFINE([FIXED_POINT], [0], [Compile as floating-point])
)

AM_CONDITIONAL([FIXED_POINT], [test x$ac_enable_fixed = xyes])


AC_CONFIG_FILES([Makefile])
AC_OUTPUT
