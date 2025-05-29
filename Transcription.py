import scipy.signal
import tkinter as tk
import os
from tkinter import filedialog, Text, messagebox
from music21 import stream, note, chord, metadata, environment, clef
import librosa
import numpy as np
import threading
import subprocess
import librosa.display
import matplotlib.pyplot as plt
import crepe
from scipy.signal import butter, filtfilt, find_peaks

# Global references for Tkinter objects
root = None
text = None
sheet_stream = None

def safe_insert(msg):
    """
    Helper function that inserts text into the Tkinter Text widget
    only if it exists (i.e., when running the UI).
    """
    if text is not None:
        text.insert(tk.END, msg)
        text.update_idletasks()

def convert_mp3_to_wav(mp3_file_path):
    wav_file_path = mp3_file_path.replace(".mp3", ".wav")
    try:
        subprocess.run(["ffmpeg", "-i", mp3_file_path, wav_file_path], check=True)
        safe_insert(f"Successfully converted {mp3_file_path} to WAV format.\n")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to convert MP3 to WAV: {e}")
        return None
    return wav_file_path

def bandpass_filter(data, lowcut, highcut, sr, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def map_duration_to_standard(duration):
    """Map raw duration to closest standard musical duration."""
    standard_durations = [0.25, 0.5, 1.0, 2.0, 4.0]
    return min(standard_durations, key=lambda x: abs(x - duration))

def detect_pitch_peaks(segment, sr, tolerance=0.03, peak_threshold_ratio=0.3):
    """
    Perform FFT-based peak detection on an audio segment.
    Returns a list of candidate fundamental frequencies (in Hz) after filtering out likely harmonics.
    """
    if len(segment) == 0:
        return []
    # Apply a Hann window to reduce spectral leakage
    window = np.hanning(len(segment))
    segment_win = segment * window

    # Compute FFT (only positive frequencies)
    fft_vals = np.abs(np.fft.rfft(segment_win))
    fft_freqs = np.fft.rfftfreq(len(segment), 1/sr)

    # Set a threshold based on a fraction of the maximum FFT magnitude
    max_val = np.max(fft_vals)
    if max_val == 0:
        return []
    threshold = max_val * peak_threshold_ratio

    # Find peaks that are above the threshold and have a minimum prominence
    peaks, properties = find_peaks(fft_vals, height=threshold, prominence=threshold/2)
    candidate_freqs = fft_freqs[peaks]

    # Filter candidates to lie within the piano frequency range
    candidate_freqs = [f for f in candidate_freqs if 27.5 <= f <= 4186.01]

    # Remove harmonically related peaks
    fundamentals = []
    for f in sorted(candidate_freqs):
        is_harmonic = False
        for fundamental in fundamentals:
            n = round(f / fundamental)
            if n > 1:
                if abs(f - n * fundamental) / (n * fundamental) < tolerance:
                    is_harmonic = True
                    break
        if not is_harmonic:
            fundamentals.append(f)
    return fundamentals

def extract_notes(y, sr, show_debug_plots=True):
    """
    Extract notes/chords from the audio signal y at sample rate sr.
    If show_debug_plots=False, skip the matplotlib plt.show() calls.
    """
    # Bandpass filter
    y_filtered = bandpass_filter(y, 27.5, 4186, sr)

    # Show filtered waveform only if show_debug_plots is True
    if show_debug_plots:
        plt.figure()
        plt.plot(y_filtered, label='Filtered Waveform')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title('Filtered Audio Waveform')
        plt.legend()
        plt.show()

    # Use Crepe
    time_arr, frequency_arr, confidence_arr, activation_arr = crepe.predict(y_filtered, sr, viterbi=True)

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    if show_debug_plots:
        plt.figure()
        plt.plot(onset_env, label='Onset Strength Envelope')
        plt.vlines(onset_times, 0, np.max(onset_env), color='r', alpha=0.9, linestyle='--', label='Onsets')
        plt.xlabel('Time (s)')
        plt.ylabel('Onset Strength')
        plt.title('Onset Detection')
        plt.legend()
        plt.show()

    notes = []

    # Process each onset window (except the last one)
    for i in range(len(onset_times) - 1):
        onset_time = onset_times[i]
        next_onset_time = onset_times[i + 1]
        start_sample = int(onset_time * sr)
        end_sample = int(next_onset_time * sr)
        segment = y_filtered[start_sample:end_sample]

        # Use FFT–based analysis to detect multiple simultaneous pitches
        chord_freqs = detect_pitch_peaks(segment, sr)

        # Map duration to a standard value
        duration = next_onset_time - onset_time
        duration = map_duration_to_standard(duration)

        if len(chord_freqs) == 1:
            midi_pitch = round(69 + 12 * np.log2(chord_freqs[0] / 440.0))
            note_obj = note.Note()
            note_obj.pitch.midi = midi_pitch
            note_obj.quarterLength = duration
            note_obj.offset = onset_time
            notes.append(note_obj)
        elif len(chord_freqs) > 1:
            chord_notes = []
            for f in chord_freqs:
                n = note.Note()
                n.pitch.midi = round(69 + 12 * np.log2(f / 440.0))
                chord_notes.append(n)
            chord_obj = chord.Chord(chord_notes)
            chord_obj.quarterLength = duration
            chord_obj.offset = onset_time
            notes.append(chord_obj)
        else:
            mask = (time_arr >= onset_time) & (time_arr < next_onset_time)
            detected_frequencies = np.array(frequency_arr)[mask]
            if len(detected_frequencies) > 0:
                fundamental_freq = np.median(detected_frequencies)
                if 27.5 <= fundamental_freq <= 4186.01:
                    midi_pitch = round(69 + 12 * np.log2(fundamental_freq / 440.0))
                    note_obj = note.Note()
                    note_obj.pitch.midi = midi_pitch
                    note_obj.quarterLength = duration
                    note_obj.offset = onset_time
                    notes.append(note_obj)

    # Handle the last onset window explicitly
    if len(onset_times) > 0:
        onset_time = onset_times[-1]
        start_sample = int(onset_time * sr)
        segment = y_filtered[start_sample:]
        chord_freqs = detect_pitch_peaks(segment, sr)
        duration = 0.5  # default duration for the final note/chord
        duration = map_duration_to_standard(duration)
        if len(chord_freqs) == 1:
            midi_pitch = round(69 + 12 * np.log2(chord_freqs[0] / 440.0))
            note_obj = note.Note()
            note_obj.pitch.midi = midi_pitch
            note_obj.quarterLength = duration
            note_obj.offset = onset_time
            notes.append(note_obj)
        elif len(chord_freqs) > 1:
            chord_notes = []
            for f in chord_freqs:
                n = note.Note()
                n.pitch.midi = round(69 + 12 * np.log2(f / 440.0))
                chord_notes.append(n)
            chord_obj = chord.Chord(chord_notes)
            chord_obj.quarterLength = duration
            chord_obj.offset = onset_time
            notes.append(chord_obj)
        else:
            mask = (time_arr >= onset_time)
            detected_frequencies = np.array(frequency_arr)[mask]
            if len(detected_frequencies) > 0:
                fundamental_freq = np.median(detected_frequencies)
                if 27.5 <= fundamental_freq <= 4186.01:
                    midi_pitch = round(69 + 12 * np.log2(fundamental_freq / 440.0))
                    note_obj = note.Note()
                    note_obj.pitch.midi = midi_pitch
                    note_obj.quarterLength = duration
                    note_obj.offset = onset_time
                    notes.append(note_obj)
    return notes

def read_audio(file_path, show_debug_plots=True):
    safe_insert(f"Reading audio file: {file_path}\n")
    if file_path.lower().endswith(".mp3"):
        file_path = convert_mp3_to_wav(file_path)
        if not file_path:
            return []
    try:
        y, sr = librosa.load(file_path, sr=44100)
        if y is None or len(y) == 0:
            raise ValueError("Loaded audio file is empty or invalid.")
        safe_insert(f"Audio file loaded successfully. Sample rate: {sr}, Length: {len(y)} samples.\n")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load audio file: {e}")
        print(f"Error in librosa.load: {e}")
        return []

    safe_insert("Extracting notes from the audio file...\n")

    notes = extract_notes(y, sr, show_debug_plots=show_debug_plots)
    safe_insert(f"Extracted {len(notes)} notes/chords from audio.\n")

    for n in notes:
        if isinstance(n, note.Note):
            print(f"Note: {n.name}, Pitch: {n.pitch.frequency:.2f} Hz, Duration: {n.quarterLength}, Offset: {n.offset}")
        elif isinstance(n, chord.Chord):
            chord_notes = ", ".join(f"{nn.name} ({nn.pitch.frequency:.2f} Hz)" for nn in n.notes)
            print(f"Chord: [{chord_notes}], Duration: {n.quarterLength}, Offset: {n.offset}")
    return notes

def open_file():
    global sheet_stream
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        text.delete(1.0, tk.END)
        text.insert(tk.END, f"Selected file: {file_path}\n")
        text.update_idletasks()
        threading.Thread(target=process_audio_file, args=(file_path,)).start()

def process_audio_file(file_path):
    global sheet_stream
    notes = read_audio(file_path)
    if notes:
        # Sort notes by their offset (timestamp)
        notes.sort(key=lambda x: x.offset)

        # Create separate streams for treble and bass clefs
        treble_stream = stream.Part()
        treble_stream.append(clef.TrebleClef())

        bass_stream = stream.Part()
        bass_stream.append(clef.BassClef())

        current_offset_treble = 0
        current_offset_bass = 0

        for n in notes:
            if isinstance(n, note.Note):
                clef_decision_pitch = n.pitch.midi
            elif isinstance(n, chord.Chord):
                clef_decision_pitch = np.median([nn.pitch.midi for nn in n.notes])
            else:
                clef_decision_pitch = 60  # fallback

            # Use the same rule as for single notes: if the pitch is C4 (MIDI 60) or above, use treble
            if clef_decision_pitch >= 60:
                if n.offset > current_offset_treble:
                    rest_duration = map_duration_to_standard(n.offset - current_offset_treble)
                    rest = note.Rest()
                    rest.quarterLength = rest_duration
                    treble_stream.append(rest)
                treble_stream.append(n)
                current_offset_treble = n.offset + n.quarterLength
            else:
                if n.offset > current_offset_bass:
                    rest_duration = map_duration_to_standard(n.offset - current_offset_bass)
                    rest = note.Rest()
                    rest.quarterLength = rest_duration
                    bass_stream.append(rest)
                bass_stream.append(n)
                current_offset_bass = n.offset + n.quarterLength

        # Combine both staves into a single score
        piano_score = stream.Score()
        piano_score.insert(0, treble_stream)
        piano_score.insert(0, bass_stream)

        # Set sheet_stream and update
        sheet_stream = piano_score
        display_notes(notes)

        safe_insert(f"Stream created with {len(notes)} notes/chords.\n")
        safe_insert("Audio processing completed successfully.\n")
        print(f"Sheet stream successfully created and populated with {len(notes)} notes/chords.")
    else:
        safe_insert("No notes extracted from audio. Check the file format or content.\n")

def display_notes(notes):
    for note_obj in notes:
        if isinstance(note_obj, note.Note):
            safe_insert(f'Note: {note_obj.name} {note_obj.quarterLength}, Offset: {note_obj.offset}\n')
        elif isinstance(note_obj, chord.Chord):
            safe_insert(f'Chord: {". ".join(n.name for n in note_obj.notes)} {note_obj.quarterLength}, Offset: {note_obj.offset}\n')

def export_to_pdf():
    global sheet_stream
    if sheet_stream:
        print("Exporting to PDF...")
        try:
            sheet_stream.metadata = metadata.Metadata()
            sheet_stream.metadata.title = "Converted Audio Composition"
            sheet_stream.metadata.composer = "Composer Name"
            sheet_stream.write(fmt='lily.pdf', fp='C:\\FYP\\lily.pdf')
            safe_insert("lily.pdf written to disk successfully!\n")
            messagebox.showinfo("Success", "lily.pdf written to disk successfully!")
            print("PDF Export successful!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export to PDF: {e}")
            print(f"Error during PDF export: {e}")
    else:
        messagebox.showerror("Error", "No audio file processed. Please load an audio file first.")
        safe_insert("Export to PDF failed: Sheet stream is None or empty.\n")
        print("Export to PDF failed: Sheet stream is None or empty.")

def transcribe_audio_file(input_file, output_pdf="transcription.pdf", show_debug_plots=False):
    """
    Processes an audio file and exports the transcription as a PDF.
    If a file with a double .pdf suffix exists, returns that filename.
    This function bypasses the Tkinter UI.
    """
    notes = read_audio(input_file, show_debug_plots=show_debug_plots)
    if not notes:
        raise ValueError("No notes extracted from audio file.")
    notes.sort(key=lambda x: x.offset)

    # Create streams for treble and bass
    treble_stream = stream.Part()
    treble_stream.append(clef.TrebleClef())
    bass_stream = stream.Part()
    bass_stream.append(clef.BassClef())

    current_offset_treble = 0
    current_offset_bass = 0

    for n in notes:
        if isinstance(n, note.Note):
            clef_decision_pitch = n.pitch.midi
        elif isinstance(n, chord.Chord):
            clef_decision_pitch = np.median([nn.pitch.midi for nn in n.notes])
        else:
            clef_decision_pitch = 60
        if clef_decision_pitch >= 60:
            if n.offset > current_offset_treble:
                rest_duration = map_duration_to_standard(n.offset - current_offset_treble)
                rest = note.Rest()
                rest.quarterLength = rest_duration
                treble_stream.append(rest)
            treble_stream.append(n)
            current_offset_treble = n.offset + n.quarterLength
        else:
            if n.offset > current_offset_bass:
                rest_duration = map_duration_to_standard(n.offset - current_offset_bass)
                rest = note.Rest()
                rest.quarterLength = rest_duration
                bass_stream.append(rest)
            bass_stream.append(n)
            current_offset_bass = n.offset + n.quarterLength

    piano_score = stream.Score()
    piano_score.insert(0, treble_stream)
    piano_score.insert(0, bass_stream)

    piano_score.metadata = metadata.Metadata()
    piano_score.metadata.title = "Converted Audio Composition"
    piano_score.metadata.composer = "Composer Name"

    # short script to deal with music 21 corrupt file issue
    piano_score.write(fmt="lily.pdf", fp=output_pdf)
    
    if output_pdf.lower().endswith(".pdf"):
        candidate = output_pdf + ".pdf"
        if os.path.exists(candidate):
            return candidate
    return output_pdf


# Only create the Tkinter UI if run directly, not if imported.
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Music Notation Software")

    open_button = tk.Button(root, text="Open Audio File", command=open_file)
    open_button.pack()

    export_button = tk.Button(root, text="Export to PDF", command=export_to_pdf)
    export_button.pack()

    text = Text(root)
    text.pack()

    sheet_stream = None

    root.mainloop()
