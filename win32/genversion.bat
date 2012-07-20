@echo off

for /f %%v in ('git describe --tags --match "v*"') do set version=%%v

set version_out=#define %2 "%version%"

echo %version_out% > %1_temp

echo n | comp %1_temp %1 > NUL 2> NUL

if not errorlevel 1 goto exit

copy /y %1_temp %1

:exit

del %1_temp
