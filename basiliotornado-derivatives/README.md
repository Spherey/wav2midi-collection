these scripts are a derivative from https://github.com/Basiliotornado/Dynamic-Midi-Converter

as of 8/31/2025, there are two scripts in this folder. the one called "NOTreduced" has twice the precision/note density than the other


# HOW IT WORKS

## this script uses two kinds of spectrograms:

 -a regular spectrogram made using a regular window function
 
 -a spectrogram made using an odd-symmetric gaussian window function

### odd-symmetric spectrograms create frequency responses of two side-lobes, where the sharp dip in the center is where the instantaneous frequency (IF) is.

### if you subtract an odd symmetric spectrogram from a regular spectrogram (where there is usually one wide main lobe), it creates a thin "spike" lobe where the IF is. (when i say subtraction in this context, think of it like subtracting two images using layer modes in an image editor program)

### therefore, it circumvents spectral imprecision in both axes (time and frequency)

for each key in the output midi file, all notes are split/chopped such that the length of each note is either 1x or a multiple of a frequency's cycle length. (if all the notes were of equal length, the whole midi would sound "buzzy" and monotone, like in this video for instance https://www.youtube.com/watch?v=5InaVQ3JWLs)


# TIPS FOR USE ELSEWHERE

 -messing with the script ppq values and midi ppq values are not recommended

 -to import into onlinesequencer.net (and this applies for any other W2M converter), you must use "tweakNotes(n => n.volume = Math.pow(n.volume, 1.5) * 0.5);" in the console while selecting all of the notes to fix any velocity mapping issues
