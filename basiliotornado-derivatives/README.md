these scripts are a derivative from https://github.com/Basiliotornado/Dynamic-Midi-Converter

as of 8/31/2025, there are two scripts in this folder. the one called "NOTreduced" has twice the precision/note density than the other




this script uses two kinds of spectrograms:

 -a regular spectrogram made using a regular window function
 
 -a spectrogram made using an odd-symmetric gaussian window function




odd-symmetric spectrograms create frequency responses of two side-lobes, where the sharp dip in the center is where the instantaneous frequency (IF) is.

if you subtract an odd symmetric spectrogram from a regular spectrogram (where there is usually one wide main lobe), it creates a "spike" lobe where the IF is.

therefore, it circumvents spectral imprecision in both axes (time and frequency)
