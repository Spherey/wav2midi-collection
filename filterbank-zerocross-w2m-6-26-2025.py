import numpy as np
import mido
import argparse
import os
from scipy.signal import firwin, filtfilt
from scipy.io.wavfile import read
from tqdm import tqdm
from scipy.interpolate import interp1d

# ==================================================================
# PSYCHOACOUSTIC WEIGHTING FUNCTION (ISO 226-inspired)
# ==================================================================
def get_psychoacoustic_gain(freq):
    """Frequency-dependent gain based on human hearing sensitivity"""
    points = [
        (20, -25.0), (50, -15.0), (100, -8.0), (200, -3.0),
        (500, -0.5), (1000, 0.0), (2000, 1.5), (4000, 4.0),
        (6000, 3.0), (8000, 2.0), (10000, 0.0), (15000, -3.0)
    ]
    freqs, gains = zip(*points)
    interpolator = interp1d(
        freqs, 
        gains, 
        kind='cubic',
        bounds_error=False,
        fill_value=(gains[0], gains[-1])
    )
    
    gain_db = interpolator(freq)
    return 10 ** (gain_db / 20)

class FrequencyBand:
    def __init__(self, midi_note, sr):
        self.note = midi_note
        self.freq = 440 * (2 ** ((midi_note - 69) / 12))
        self.sr = sr
        self.filter = self.create_filter()
        
    def create_filter(self):
        numtaps = 2049
        cutoff = [self.freq * 0.95, self.freq * 1.05]
        return firwin(numtaps, cutoff, fs=self.sr, pass_zero=False)

class AudioAnalyzer:
    def __init__(self, file_path, target_sr=44100, stereo=False):
        self.stereo = stereo
        self.sr, self.audio = read(file_path)
        
        if self.audio.ndim > 1:
            if stereo:
                # Separate left/right channels
                self.audio_left = self.audio[:, 0].astype(np.float32)
                self.audio_right = self.audio[:, 1].astype(np.float32)
                # Normalize each channel separately
                self.audio_left /= np.max(np.abs(self.audio_left))
                self.audio_right /= np.max(np.abs(self.audio_right))
            else:
                # Mono conversion
                self.audio = self.audio.mean(axis=1).astype(np.float32)
                self.audio /= np.max(np.abs(self.audio))
        else:
            self.audio = self.audio.astype(np.float32)
            self.audio /= np.max(np.abs(self.audio))
        
        if self.sr != target_sr:
            if stereo:
                self.resample(target_sr, stereo=True)
            else:
                self.resample(target_sr)
            self.sr = target_sr
            
    def resample(self, target_sr, stereo=False):
        ratio = target_sr / self.sr
        if stereo:
            # Resample left channel
            old_length = len(self.audio_left)
            new_length = int(old_length * ratio)
            x_old = np.arange(old_length)
            x_new = np.linspace(0, old_length-1, new_length)
            self.audio_left = np.interp(x_new, x_old, self.audio_left)
            
            # Resample right channel
            old_length = len(self.audio_right)
            new_length = int(old_length * ratio)
            x_old = np.arange(old_length)
            x_new = np.linspace(0, old_length-1, new_length)
            self.audio_right = np.interp(x_new, x_old, self.audio_right)
        else:
            # Resample mono audio
            old_length = len(self.audio)
            new_length = int(old_length * ratio)
            x_old = np.arange(old_length)
            x_new = np.linspace(0, old_length-1, new_length)
            self.audio = np.interp(x_new, x_old, self.audio)

class MidiConverter:
    def __init__(self, ppqn=22050, bpm=120*(88200/1920), stereo=False):
        self.ppqn = ppqn
        self.bpm = bpm
        self.ticks_per_sec = 44100
        self.stereo = stereo
        
    def convert_events(self, all_events):
        # Create tracks for each channel (1 for mono, 2 for stereo)
        tracks = {1: mido.MidiTrack(), 2: mido.MidiTrack()} if self.stereo else {1: mido.MidiTrack()}
        
        # Add panning to each track
        if self.stereo:
            # Set left pan for channel 1 track
            tracks[1].append(mido.Message(
                'control_change', channel=1, control=10, value=0
            ))
            # Set right pan for channel 2 track
            tracks[2].append(mido.Message(
                'control_change', channel=2, control=10, value=127
            ))
        else:
            # Center pan for mono track
            tracks[1].append(mido.Message(
                'control_change', channel=1, control=10, value=64
            ))
        
        sorted_events = sorted(all_events, key=lambda x: x['time'])
        last_times = {channel: 0 for channel in tracks.keys()}
        
        for event in tqdm(sorted_events, desc="Placing notes"):
            channel = event['channel']
            track = tracks[channel]
            ticks = int(event['time'] * self.ticks_per_sec)
            delta = ticks - last_times[channel]
            
            track.append(mido.Message(
                event['type'],
                note=event['note'],
                velocity=event.get('velocity', 64),
                time=delta,
                channel=channel
            ))
            last_times[channel] = ticks
        
        # Create master track for tempo
        master_track = mido.MidiTrack()
        return [master_track] + list(tracks.values())

def analyze_band(args):
    band, audio, min_duration_sec = args
    try:
        filtered = filtfilt(band.filter, [1.0], audio)
    except ValueError:
        return []
    
    # Pre-calculate psychoacoustic gain for this band
    band_gain = get_psychoacoustic_gain(band.freq)
    
    crossings = []
    state = 0
    HYSTERESIS = 0.0001
    
    for i in tqdm(range(len(filtered)), desc=f"Note {band.note:03d}", leave=False):
        current = filtered[i]
        if state == 0 and current > HYSTERESIS:
            state = 1
            crossings.append(i)
        elif state == 1 and current < -HYSTERESIS:
            state = 0
            crossings.append(i)
    
    if len(crossings) < 2:
        return []

    # Process oscillation cycles into note events
    note_events = []
    for i in range(2, len(crossings), 2):
        start = crossings[i-2]
        end = crossings[i]
        
        if end - start < 2:
            continue
            
        segment = filtered[start:end]
        
        # Apply psychoacoustic weighting before velocity calculation
        weighted_segment = segment * band_gain
        
        # Calculate velocity from weighted amplitude
        peak_amplitude = np.max(np.abs(weighted_segment))
        velocity = int(np.clip(peak_amplitude ** 0.5 * 127, 1, 127))
        
        start_time = start / band.sr
        end_time = end / band.sr

        wavelength = 1.0 / band.freq
        max_duration = 1.5 * wavelength
        actual_duration = end_time - start_time

        if actual_duration > max_duration:
            end_time = start_time + wavelength

        note_events.append({
            'start_time': start_time,
            'end_time': end_time,
            'velocity': velocity,
            'duration': end_time - start_time
        })
    
    # Merge short notes if min duration specified
    if min_duration_sec > 0:
        merged_events = []
        current_merge = []
        current_velocities = []
        
        for note in note_events:
            # Always add current note to potential merge group
            current_merge.append(note)
            current_velocities.append(note['velocity'])
            
            # Calculate total duration of current merge group
            total_duration = current_merge[-1]['end_time'] - current_merge[0]['start_time']
            
            # If we've reached the minimum duration, merge the group
            if total_duration >= min_duration_sec:
                avg_velocity = int(np.mean(current_velocities))
                
                merged_events.append({
                    'start_time': current_merge[0]['start_time'],
                    'end_time': current_merge[-1]['end_time'],
                    'velocity': avg_velocity
                })
                
                # Reset merge group
                current_merge = []
                current_velocities = []
        
        # Add any remaining notes in the merge group
        if current_merge:
            avg_velocity = int(np.mean(current_velocities))
            merged_events.append({
                'start_time': current_merge[0]['start_time'],
                'end_time': current_merge[-1]['end_time'],
                'velocity': avg_velocity
            })
        
        # Replace original notes with merged ones
        note_events = merged_events
    
    # Convert to MIDI events
    events = []
    for note in note_events:
        events.append({
            'type': 'note_on', 
            'note': band.note, 
            'time': note['start_time'], 
            'velocity': note['velocity']
        })
        events.append({
            'type': 'note_off', 
            'note': band.note, 
            'time': note['end_time'], 
            'velocity': 0
        })
    
    return events

def colorize_midi(input_path, output_path, num_tracks=31, stereo=False):
    mid = mido.MidiFile(input_path)
    # Create structure: tracks[velocity_bin][channel] = list of events
    tracks = {bin_idx: {1: [], 2: []} for bin_idx in range(num_tracks)}
    active_notes = {}
    control_messages = []  # Collect all control messages

    # First pass: collect all events
    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            
            # Collect control messages (including panning)
            if msg.type in ['control_change', 'set_tempo']:
                control_messages.append((current_time, msg))
                
            # Handle note events
            elif msg.type == 'note_on' and msg.velocity > 0:
                track_index = min(msg.velocity // (128 // num_tracks), num_tracks - 1)
                channel = msg.channel
                tracks[track_index][channel].append(('on', msg.note, current_time, msg.velocity))
                active_notes[(msg.note, channel)] = (track_index, current_time)
                
            elif msg.type in ['note_off', 'note_on'] and (msg.velocity == 0 or msg.type == 'note_off'):
                channel = msg.channel
                note_key = (msg.note, channel)
                if note_key in active_notes:
                    track_index, start_time = active_notes[note_key]
                    tracks[track_index][channel].append(('off', msg.note, current_time, 0))
                    del active_notes[note_key]

    # Create new MIDI structure
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    
    # Master track for global settings
    master_track = mido.MidiTrack()
    
    # Add control messages to master track (sorted by time)
    prev_time = 0
    for abs_time, msg in sorted(control_messages, key=lambda x: x[0]):
        delta = abs_time - prev_time
        msg.time = delta
        master_track.append(msg)
        prev_time = abs_time
    
    new_mid.tracks.append(master_track)

    # Process note tracks with panning
    for vel_bin in range(num_tracks):
        for channel in [1, 2]:
            track_events = tracks[vel_bin][channel]
            if not track_events:
                continue
                
            sorted_track = sorted(track_events, key=lambda x: x[2])
            midi_track = mido.MidiTrack()
            
            # Add panning to this specific track
            if stereo:
                pan_value = 0 if channel == 1 else 127
                midi_track.append(mido.Message(
                    'control_change', channel=channel, control=10, value=pan_value, time=0
                ))
            
            prev_time = 0
            for event in sorted_track:
                delta = event[2] - prev_time
                if event[0] == 'on':
                    msg = mido.Message('note_on', note=event[1], velocity=event[3], 
                                      channel=channel, time=delta)
                else:
                    msg = mido.Message('note_off', note=event[1], velocity=0, 
                                      channel=channel, time=delta)
                midi_track.append(msg)
                prev_time = event[2]
                
            new_mid.tracks.append(midi_track)

    new_mid.save(output_path)

def main(file_path, output_path):
    # STEREO PROMPT (FIRST PROMPT)
    stereo_enabled = input("\nEnable stereo output? (y/n): ").strip().lower() == 'y'
    
    audio_analyzer = AudioAnalyzer(file_path, stereo=stereo_enabled)
    converter = MidiConverter(stereo=stereo_enabled)
    
    # Ask about short note limiting
    use_min_duration = input("\nImpose minimum note length? (y/n): ").strip().lower() == 'y'
    min_duration_sec = 0
    if use_min_duration:
        min_duration_ms = float(input("Minimum note duration (milliseconds): "))
        min_duration_sec = min_duration_ms / 1000.0
    
    bands = [FrequencyBand(n, audio_analyzer.sr) for n in tqdm(range(128), desc="Creating bands")]
    
    all_events = []
    
    # Process audio based on stereo setting
    if stereo_enabled:
        # Process left channel (channel 1)
        for band in tqdm(bands, desc="Analyzing bands (left)"):
            events = analyze_band((band, audio_analyzer.audio_left, min_duration_sec))
            for event in events:
                event['channel'] = 1  # Left channel = channel 1
            all_events.extend(events)
        
        # Process right channel (channel 2)
        for band in tqdm(bands, desc="Analyzing bands (right)"):
            events = analyze_band((band, audio_analyzer.audio_right, min_duration_sec))
            for event in events:
                event['channel'] = 2  # Right channel = channel 2
            all_events.extend(events)
    else:
        # Mono processing (channel 1)
        for band in tqdm(bands, desc="Analyzing bands"):
            events = analyze_band((band, audio_analyzer.audio, min_duration_sec))
            for event in events:
                event['channel'] = 1  # All on channel 1
            all_events.extend(events)
    
    midi = mido.MidiFile(type=1)
    midi.tracks = converter.convert_events(all_events)
    midi.tracks[0].append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(converter.bpm)))
    
    base_path = os.path.splitext(output_path)[0]
    intermediate_path = f"{base_path}_intermediate.mid"
    colored_path = f"{base_path}-colored.mid"
    
    midi.save(intermediate_path)
    
    # Colorization prompt
    colorize = input("\nConvert velocities to colored tracks? (y/n): ").strip().lower() == 'y'
    
    if colorize:
        colorize_midi(intermediate_path, colored_path, stereo=stereo_enabled)
        os.remove(intermediate_path)
        print(f"\nSuccessfully saved colored MIDI to {colored_path}")
    else:
        os.rename(intermediate_path, output_path)
        print(f"\nMIDI saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Professional Audio-to-MIDI Converter with Psychoacoustic Weighting')
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('output', help='Output MIDI file')
    args = parser.parse_args()
    
    main(args.input, args.output)