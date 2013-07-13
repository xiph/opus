dnl Process this file with autoconf to produce a configure script. -*-m4-*-

dnl The package_version file will be automatically synced to the git revision
dnl by the update_version script when configured in the repository, but will
dnl remain constant in tarball releases unless it is manually edited.
m4_define([CURRENT_VERSION],
          m4_esyscmd([ ./update_version 2>/dev/null || true
                       if test -e package_version; then
                           . ./package_version
                           printf "$PACKAGE_VERSION"
                       else
                           printf "unknown"
                       fi ]))

AC_INIT([opus],[CURRENT_VERSION],[opus@xiph.org])

AC_CONFIG_SRCDIR(src/opus_encoder.c)
AC_CONFIG_MACRO_DIR([m4])

dnl enable silent rules on automake 1.11 and later
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# For libtool.
dnl Please update these for releases.
OPUS_LT_CURRENT=4
OPUS_LT_REVISION=0
OPUS_LT_AGE=4

AC_SUBST(OPUS_LT_CURRENT)
AC_SUBST(OPUS_LT_REVISION)
AC_SUBST(OPUS_LT_AGE)

AM_INIT_AUTOMAKE([no-define])
AM_MAINTAINER_MODE([enable])

AC_CANONICAL_HOST
AC_MINGW32
AM_PROG_LIBTOOL
AM_PROG_CC_C_O

AC_PROG_CC_C99
AC_C_CONST
AC_C_INLINE

AC_DEFINE([OPUS_BUILD], [], [This is a build of OPUS])

#Use a hacked up version of autoconf's AC_C_RESTRICT because it's not
#strong enough a test to detect old buggy versions of GCC (e.g. 2.95.3)
#Note: Both this and the test for variable-size arrays below are also
#      done by AC_PROG_CC_C99, but not thoroughly enough apparently.
AC_CACHE_CHECK([for C/C++ restrict keyword], ac_cv_c_restrict,
  [ac_cv_c_restrict=no
   # The order here caters to the fact that C++ does not require restrict.
   for ac_kw in __restrict __restrict__ _Restrict restrict; do
     AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
      [[typedef int * int_ptr;
	int foo (int_ptr $ac_kw ip, int * $ac_kw baz[]) {
	return ip[0];
       }]],
      [[int s[1];
	int * $ac_kw t = s;
	t[0] = 0;
	return foo(t, (void *)0)]])],
      [ac_cv_c_restrict=$ac_kw])
     test "$ac_cv_c_restrict" != no && break
   done
  ])

AH_VERBATIM([restrict],
[/* Define to the equivalent of the C99 'restrict' keyword, or to
   nothing if this is not supported.  Do not define if restrict is
   supported directly.  */
#undef restrict
/* Work around a bug in Sun C++: it does not support _Restrict or
   __restrict__, even though the corresponding Sun C compiler ends up with
   "#define restrict _Restrict" or "#define restrict __restrict__" in the
   previous line.  Perhaps some future version of Sun C++ will work with
   restrict; if so, hopefully it defines __RESTRICT like Sun C does.  */
#if defined __SUNPRO_CC && !defined __RESTRICT
# define _Restrict
# define __restrict__
#endif])

case $ac_cv_c_restrict in
   restrict) ;;
   no) AC_DEFINE([restrict], []) ;;
   *)  AC_DEFINE_UNQUOTED([restrict], [$ac_cv_c_restrict]) ;;
esac

AC_MSG_CHECKING(for C99 variable-size arrays)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],
                   [[static int x; char a[++x]; a[sizeof a - 1] = 0; int N; return a[0];]])],
    [ has_var_arrays=yes
      use_alloca="no (using var arrays)"
      AC_DEFINE([VAR_ARRAYS], [1], [Use C99 variable-size arrays])
    ],[
      has_var_arrays=no
    ])
AC_MSG_RESULT([$has_var_arrays])

AS_IF([test "$has_var_arrays" = "no"],
  [
   AC_CHECK_HEADERS([alloca.h])
   AC_MSG_CHECKING(for alloca)
   AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <alloca.h>]],
                                      [[int foo=10; int *array = alloca(foo);]])],
     [ use_alloca=yes;
       AC_DEFINE([USE_ALLOCA], [], [Make use of alloca])
     ],[
       use_alloca=no
     ])
   AC_MSG_RESULT([$use_alloca])
  ])

LT_LIB_M

AC_ARG_ENABLE([fixed-point],
    [AS_HELP_STRING([--enable-fixed-point],
                    [compile without floating point (for machines without a fast enough FPU)])],,
    [enable_fixed_point=no])

AS_IF([test "$enable_fixed_point" = "yes"],[
  enable_float="no"
  AC_DEFINE([FIXED_POINT], [1], [Compile as fixed-point (for machines without a fast enough FPU)])
  PC_BUILD="fixed-point"
],[
  enable_float="yes";
  PC_BUILD="floating-point"
])

AM_CONDITIONAL([FIXED_POINT], [test "$enable_fixed_point" = "yes"])

AC_ARG_ENABLE([fixed-point-debug],
    [AS_HELP_STRING([--enable-fixed-point-debug], [debug fixed-point implementation])],,
    [enable_fixed_point_debug=no])

AS_IF([test "$enable_fixed_point_debug" = "yes"],[
  AC_DEFINE([FIXED_DEBUG], [1], [Debug fixed-point implementation])
])

AC_ARG_ENABLE([custom-modes],
    [AS_HELP_STRING([--enable-custom-modes], [enable non-Opus modes, e.g. 44.1 kHz & 2^n frames])],,
    [enable_custom_modes=no])

AS_IF([test "$enable_custom_modes" = "yes"],[
  AC_DEFINE([CUSTOM_MODES], [1], [Custom modes])
  PC_BUILD="$PC_BUILD, custom modes"
])

AM_CONDITIONAL([CUSTOM_MODES], [test "$enable_custom_modes" = "yes"])

has_float_approx=no
#case "$host_cpu" in
#i[[3456]]86 | x86_64 | powerpc64 | powerpc32 | ia64)
#  has_float_approx=yes
#  ;;
#esac    

AC_ARG_ENABLE([float-approx],
    [AS_HELP_STRING([--enable-float-approx], [enable fast approximations for floating point])],
    [if test "$enable_float_approx" = "yes"; then
       AC_WARN([Floating point approximations are not supported on all platforms.])
     fi
    ],
    [enable_float_approx=$has_float_approx])

AS_IF([test "$enable_float_approx" = "yes"],[
  AC_DEFINE([FLOAT_APPROX], [1], [Float approximations])
])

AC_ARG_ENABLE([asm],
    [AS_HELP_STRING([--disable-asm], [Disable assembly optimizations])],,
    [enable_asm=yes])

rtcd_support=no
cpu_arm=no

AS_IF([test "$enable_asm" = "yes"],[
    asm_optimization="no asm for your platform, please send patches"
    case $host_cpu in
    arm*)
        cpu_arm=yes
        AS_GCC_INLINE_ASSEMBLY([asm_optimization="ARM"],
            [asm_optimization="disabled"])
        if test "$asm_optimization" = "ARM" ; then
            rtcd_support=yes
            AC_DEFINE([ARMv4_ASM], 1, [Use generic ARMv4 asm optimizations])
            AS_ASM_ARM_EDSP([ARMv5E_ASM=1],[ARMv5E_ASM=0])
            if test "$ARMv5E_ASM" = "1" ; then
                AC_DEFINE([ARMv5E_ASM], [1], [Use ARMv5E asm optimizations])
                asm_optimization="$asm_optimization (EDSP)"
            fi
            AS_ASM_ARM_MEDIA([ARMv6_ASM=1],[ARMv6_ASM=0])
            if test "$ARMv6_ASM" = "1" ; then
                AC_DEFINE([ARMv6_ASM], [1], [Use ARMv6 asm optimizations])
                asm_optimization="$asm_optimization (Media)"
            fi
            AS_ASM_ARM_NEON([ARM_HAVE_NEON=1],[ARM_HAVE_NEON=0])
            if test "$ARM_HAVE_NEON" = "1" ; then
              AC_DEFINE([ARM_HAVE_NEON], 1, [Use ARM NEON optimizations])
              asm_optimization="$asm_optimization (NEON)"
            fi
        fi
        ;;
    esac
],[
    asm_optimization="disabled"
])

AM_CONDITIONAL([CPU_ARM], [test "$cpu_arm" = "yes"])

AC_ARG_ENABLE([rtcd],
    [AS_HELP_STRING([--disable-rtcd], [Disable run-time CPU capabilities detection])],,
    [enable_rtcd=yes])

AS_IF([test "$enable_rtcd" = "yes"],[
    AS_IF([test "$rtcd_support" = "yes"],[
        AC_DEFINE([OPUS_HAVE_RTCD], [1], [Use run-time CPU capabilities detection])
    ],[
        rtcd_support="no rtcd for your platform, please send patches"
    ])
],[
    rtcd_support="no"
])

AC_ARG_ENABLE([assertions],
    [AS_HELP_STRING([--enable-assertions],[enable additional software error checking])],,
    [enable_assertions=no])

AS_IF([test "$enable_assertions" = "yes"], [
  AC_DEFINE([ENABLE_ASSERTIONS], [1], [Assertions])
])

AC_ARG_ENABLE([fuzzing],
    [AS_HELP_STRING([--enable-fuzzing],[causes the encoder to make random decisions])],,
    [enable_fuzzing=no])

AS_IF([test "$enable_fuzzing" = "yes"], [
  AC_DEFINE([FUZZING], [1], [Fuzzing])
])

AC_ARG_ENABLE([doc],
    [AS_HELP_STRING([--disable-doc], [Do not build API documentation])],,
    [enable_doc=yes])

AS_IF([test "$enable_doc" = "yes"], [
  AC_CHECK_PROG(HAVE_DOXYGEN, [doxygen], [yes], [no])
],[
  HAVE_DOXYGEN=no
])

AM_CONDITIONAL([HAVE_DOXYGEN], [test "$HAVE_DOXYGEN" = "yes"])

AC_ARG_ENABLE([extra-programs],
    [AS_HELP_STRING([--disable-extra-programs], [Do not build extra programs (demo and tests)])],,
    [enable_extra_programs=yes])

AM_CONDITIONAL([EXTRA_PROGRAMS], [test "$enable_extra_programs" = "yes"])


saved_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -fvisibility=hidden"
AC_MSG_CHECKING([if ${CC} supports -fvisibility=hidden])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[char foo;]])],
    [ AC_MSG_RESULT([yes]) ],
    [ AC_MSG_RESULT([no])
      CFLAGS="$saved_CFLAGS"
    ])

CFLAGS="$CFLAGS -W"

warn_CFLAGS="-Wall -Wextra -Wcast-align -Wnested-externs -Wshadow -Wstrict-prototypes"
saved_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS $warn_CFLAGS"
AC_MSG_CHECKING([if ${CC} supports ${warn_CFLAGS}])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[char foo;]])],
    [ AC_MSG_RESULT([yes]) ],
    [ AC_MSG_RESULT([no])
      CFLAGS="$saved_CFLAGS"
    ])

saved_LIBS="$LIBS"
LIBS="$LIBS $LIBM"
AC_CHECK_FUNCS([lrintf])
AC_CHECK_FUNCS([lrint])
LIBS="$saved_LIBS"

AC_CHECK_FUNCS([__malloc_hook])

AC_SUBST([PC_BUILD])


AC_CONFIG_FILES([Makefile opus.pc opus-uninstalled.pc
                 doc/Makefile doc/Doxyfile])
AC_CONFIG_HEADERS([config.h])

AC_OUTPUT

AC_MSG_NOTICE([
------------------------------------------------------------------------
  $PACKAGE_NAME $PACKAGE_VERSION:  Automatic configuration OK.

    Compiler support:

      C99 var arrays: ................ ${has_var_arrays}
      C99 lrintf: .................... ${ac_cv_func_lrintf}
      Use alloca: .................... ${use_alloca}
    
    General configuration:
    
      Floating point support: ........ ${enable_float}
      Fast float approximations: ..... ${enable_float_approx}
      Fixed point debugging: ......... ${enable_fixed_point_debug}
      Assembly optimization: ......... ${asm_optimization}
      Run-time CPU detection: ........ ${rtcd_support}
      Custom modes: .................. ${enable_custom_modes}
      Assertion checking: ............ ${enable_assertions}
      Fuzzing: ....................... ${enable_fuzzing}

      API documentation: ............. ${enable_doc}
      Extra programs: ................ ${enable_extra_programs}
------------------------------------------------------------------------

 Type "make; make install" to compile and install
 Type "make check" to run the test suite
])

