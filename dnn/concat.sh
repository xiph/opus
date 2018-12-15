# Place in 16k-LP7 from TSPSpeech.iso and run to concatenate wave files
# into one headerless training file
for i in */*.wav
do
sox $i -r 16000 -c 1 -t sw -
done > input.s16
