@echo off
REM Run this to set up the build system: configure, makefiles, etc.

setlocal enabledelayedexpansion

REM Parse the real autogen.sh script for version
for /F "tokens=4 delims= " %%A in ('findstr "download_model.sh" autogen.sh') do (
    set "model=%%A"
)
REM Remove trailing ")" character from the model variable
set "model=%model:~0,-1%"

cd lpcnet
call download_model.bat %model%
cd ..

echo Updating build configuration files, please wait....
