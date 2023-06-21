@echo off
set model=lpcnet_data-%1.tar.gz

if not exist %model% (
    echo Downloading latest model
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://media.xiph.org/lpcnet/data/%model%', '%model%')"
)

tar -xvzf %model%

