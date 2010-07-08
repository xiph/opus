#!/bin/sh

./convert_source.sh

./ietf_source.sh

#codec draft
xml2rfc draft-valin-celt-codec.xml draft-valin-celt-codec.html

xml2rfc draft-valin-celt-codec.xml draft-valin-celt-codec.txt

#RTP draft
xml2rfc draft-valin-celt-rtp-profile.xml draft-valin-celt-rtp-profile.html

xml2rfc draft-valin-celt-rtp-profile.xml draft-valin-celt-rtp-profile.txt
