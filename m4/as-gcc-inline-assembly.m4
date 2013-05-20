dnl as-gcc-inline-assembly.m4 0.1.0

dnl autostars m4 macro for detection of gcc inline assembly

dnl David Schleef <ds@schleef.org>

dnl $Id$

dnl AS_COMPILER_FLAG(ACTION-IF-ACCEPTED, [ACTION-IF-NOT-ACCEPTED])
dnl Tries to compile with the given CFLAGS.
dnl Runs ACTION-IF-ACCEPTED if the compiler can compile with the flags,
dnl and ACTION-IF-NOT-ACCEPTED otherwise.

AC_DEFUN([AS_GCC_INLINE_ASSEMBLY],
[
  AC_MSG_CHECKING([if compiler supports gcc-style inline assembly])

  AC_TRY_COMPILE([], [
#ifdef __GNUC_MINOR__
#if (__GNUC__ * 1000 + __GNUC_MINOR__) < 3004
#error GCC before 3.4 has critical bugs compiling inline assembly
#endif
#endif
__asm__ (""::) ], [flag_ok=yes], [flag_ok=no])

  if test "X$flag_ok" = Xyes ; then
    $1
    true
  else
    $2
    true
  fi
  AC_MSG_RESULT([$flag_ok])
])

AC_DEFUN([AC_TRY_ASSEMBLE],
[ac_c_ext=$ac_ext
 ac_ext=${ac_s_ext-s}
 cat > conftest.$ac_ext <<EOF
 .file "configure"
[$1]
EOF
if AC_TRY_EVAL(ac_compile); then
  ac_ext=$ac_c_ext
  ifelse([$2], , :, [  $2
  rm -rf conftest*])
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
  ac_ext=$ac_c_ext
ifelse([$3], , , [  rm -rf conftest*
  $3
])dnl
fi
rm -rf conftest*])


AC_DEFUN([AS_ASM_ARM_NEON],
[
  AC_MSG_CHECKING([if assembler supports NEON instructions on ARM])

  AC_TRY_ASSEMBLE([vorr d0,d0,d0], [flag_ok=yes], [flag_ok=no])

  if test "X$flag_ok" = Xyes ; then
    $1
    true
  else
    $2
    true
  fi
  AC_MSG_RESULT([$flag_ok])
])


AC_DEFUN([AS_ASM_ARM_MEDIA],
[
  AC_MSG_CHECKING([if assembler supports ARMv6 media instructions on ARM])

  AC_TRY_ASSEMBLE([shadd8 r3,r3,r3], [flag_ok=yes], [flag_ok=no])

  if test "X$flag_ok" = Xyes ; then
    $1
    true
  else
    $2
    true
  fi
  AC_MSG_RESULT([$flag_ok])
])


AC_DEFUN([AS_ASM_ARM_EDSP],
[
  AC_MSG_CHECKING([if assembler supports EDSP instructions on ARM])

  AC_TRY_ASSEMBLE([qadd r3,r3,r3], [flag_ok=yes], [flag_ok=no])

  if test "X$flag_ok" = Xyes ; then
    $1
    true
  else
    $2
    true
  fi
  AC_MSG_RESULT([$flag_ok])
])
