#!/bin/sh

#Stop on errors
set -e
#Set the CWD to the location of this script
[ -n "${0%/*}" ] && cd "${0%/*}"

echo running xml2rfc
xml2rfc draft-terriberry-oggopus.xml draft-terriberry-oggopus.html &
xml2rfc draft-terriberry-oggopus.xml
wait
