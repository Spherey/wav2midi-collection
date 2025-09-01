
# this converter is the highest quality wav2midi converter thus far.

# HOW IT WORKS

## the "filterbank-zerocross" converter converts .wav files to .mid by using a filterbank algorithm to:
 1. separate an audio file into 128 bands
 before
 2. using the zero crossings of each band (they're all waveforms)
 to
 3. determine the length of various notes
 and
 4. write them to the output midi file
