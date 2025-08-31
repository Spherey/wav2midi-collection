from scipy.fft import fft
import numpy as np
import mido
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.io.wavfile import read
import sys
from scipy.signal import get_window, resample

# Combined correction factor (13200/17939 ≈ 0.7358)
CORRECTION_FACTOR = 0.75

def frequency_from_key(key):
    return 2 ** ((key - 69) / 12) * 440

def flat_top_window(size):
    return get_window('blackmanharris', int(size), fftbins=True)

def calculate_phase(samples, target_freq, samplerate):
    cycle_samples = int(round(samplerate / target_freq))
    if cycle_samples < 1 or cycle_samples > len(samples):
        return 0.0, 0.0
    
    win = flat_top_window(cycle_samples)
    windowed = samples[:cycle_samples] * win
    
    fft_result = fft(windowed)
    magnitude = np.abs(fft_result[1]) / np.sum(win)
    phase = np.angle(fft_result[1])
    
    return magnitude, phase

def process_note(args):
    note, padded_L, padded_R, samplerate, max_padding = args
    freq = frequency_from_key(note)
    if freq < 20:
        return []
    
    cycle_samples = int(round(samplerate / freq))
    if cycle_samples < 1:
        return []
    
    note_events = []
    n = max_padding
    
    while n <= len(padded_L) - cycle_samples - max_padding:
        window_L = padded_L[n:n+cycle_samples]
        window_R = padded_R[n:n+cycle_samples]
        
        mag_L, phase_L = calculate_phase(window_L, freq, samplerate)
        mag_R, phase_R = calculate_phase(window_R, freq, samplerate)
        
        # Phase alignment
        shift = -((phase_L + phase_R) / 2) * cycle_samples / (2 * np.pi)
        adjusted_n = n + int(round(shift))
        adjusted_n = max(max_padding, min(adjusted_n, len(padded_L) - cycle_samples - max_padding))
        
        # Get final magnitudes
        mag_L, _ = calculate_phase(padded_L[adjusted_n:adjusted_n+cycle_samples], freq, samplerate)
        mag_R, _ = calculate_phase(padded_R[adjusted_n:adjusted_n+cycle_samples], freq, samplerate)
        
        # Apply correction factor to timing
        start_time = (adjusted_n - max_padding) / samplerate / CORRECTION_FACTOR
        duration = cycle_samples / samplerate / CORRECTION_FACTOR
        
        if mag_L > 0:
            note_events.append(('L', note, start_time, duration, mag_L))
        if mag_R > 0:
            note_events.append(('R', note, start_time, duration, mag_R))
        
        n += cycle_samples
    
    return note_events

def main():
    parser = argparse.ArgumentParser(description='Phase-Aligned Audio to MIDI')
    parser.add_argument("i", type=str, help="Input audio file")
    parser.add_argument("-o", type=str, default="output.mid", help="Output MIDI file")
    parser.add_argument("--ppqn", type=int, default=30000, help="Pulses per quarter note")
    parser.add_argument("--bpm", type=int, default=120, help="Tempo in BPM")
    parser.add_argument("--threads", type=int, default=cpu_count(), help="Processing threads")
    args = parser.parse_args()

    # Load and resample audio
    try:
        if args.i.lower().endswith('.wav'):
            orig_rate, data = read(args.i)
        else:
            from subprocess import run
            cmd = ['ffmpeg', '-i', args.i, '-ar', '48000', '-ac', '2', '-f', 's16le', '-']
            result = run(cmd, capture_output=True)
            data = np.frombuffer(result.stdout, dtype=np.int16).reshape(-1, 2)
            orig_rate = 48000
        
        # Apply correction factor through resampling
        new_length = int(len(data) * CORRECTION_FACTOR)
        if data.ndim == 1:
            data = resample(data, new_length)
        else:
            data = resample(data, new_length, axis=0)
        samplerate = int(orig_rate * CORRECTION_FACTOR)
        
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        sys.exit(1)

    # Handle mono/stereo
    if data.ndim == 1:
        dataL = dataR = data
    else:
        dataL = data[:, 0]
        dataR = data[:, 1]

    # Calculate padding
    lowest_freq = frequency_from_key(0)
    max_padding = int(2 * samplerate / max(lowest_freq, 20))

    padded_L = np.pad(dataL, (max_padding, max_padding), mode='constant')
    padded_R = np.pad(dataR, (max_padding, max_padding), mode='constant')

    # Process notes
    all_events = []
    with Pool(args.threads) as pool:
        tasks = [(note, padded_L, padded_R, samplerate, max_padding) for note in range(128)]
        for result in tqdm(pool.imap(process_note, tasks), total=128, desc="Processing notes"):
            all_events.extend(result)

    # Normalize velocities
    if not all_events:
        print("No events generated!")
        sys.exit(1)
    
    max_mag = max(event[4] for event in all_events)
    processed_events = []
    for event in all_events:
        ch, note, start, dur, mag = event
        velocity = int((mag / max_mag) ** 0.5 * 127)
        velocity = max(1, min(velocity, 127))
        processed_events.append((ch, note, start, dur, velocity))

    # Create MIDI structure
    mid = mido.MidiFile(ticks_per_beat=args.ppqn, type=1)
    main_track = mido.MidiTrack()
    left_track = mido.MidiTrack()
    right_track = mido.MidiTrack()
    mid.tracks.extend([main_track, left_track, right_track])

    main_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(args.bpm)))

    # Sort and add events
    track_events = {'L': [], 'R': []}
    for event in processed_events:
        ch, note, start, dur, vel = event
        track = 'L' if ch == 'L' else 'R'
        start_ticks = mido.second2tick(start, args.ppqn, mido.bpm2tempo(args.bpm))
        end_ticks = mido.second2tick(start + dur, args.ppqn, mido.bpm2tempo(args.bpm))
        track_events[track].append(('note_on', note, vel, start_ticks))
        track_events[track].append(('note_off', note, 0, end_ticks))

    # Build tracks
    for channel in ['L', 'R']:
        current_track = left_track if channel == 'L' else right_track
        events = sorted(track_events[channel], key=lambda x: x[3])
        last_time = 0
        for event in events:
            msg_type, note, vel, time = event
            delta = int(time - last_time)
            current_track.append(mido.Message(
                msg_type,
                note=note,
                velocity=vel,
                time=delta,
                channel=0 if channel == 'L' else 1
            ))
            last_time = time

    mid.save(args.o)
    print(f"Saved MIDI to {args.o}")

if __name__ == "__main__":
    main()
