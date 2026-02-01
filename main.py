#!/usr/bin/env python3
"""
system_note_keyer_daemon.py - Fixed for repeated notes + noise rejection
"""

import subprocess
import numpy as np
import math
import time
from collections import deque
import shutil
import argparse

# ----------------- Config / tuning -----------------
SAMPLE_RATE = 44100
FRAME_SIZE = 1024 
FMIN, FMAX = 50.0, 2000.0

# INCREASED THRESHOLDS to stop random inputs
ENERGY_THRESHOLD = 0.005  # Ignore low-level background hum
CONFIDENCE_THRESHOLD = 0.5 # Autocorrelation must be very clear
TOLERANCE_CENTS = 40      # How close to the exact note you must be

SMOOTH_WINDOW = 4       
STABLE_FRAMES_FOR_ON = 3
STABLE_FRAMES_FOR_OFF = 2 

MAX_NOTES = 7
NOTE_C4, NOTE_C5, NOTE_G4, NOTE_E4 = 60, 72, 67, 64
TARGET_NOTES = {NOTE_C4, NOTE_C5, NOTE_G4, NOTE_E4}

MIDI_TO_KEY = {
    NOTE_C4: "Down",
    NOTE_C5: "Up",
    NOTE_G4: "Right",
    NOTE_E4: "Left",
}

KEY_PRESS_DELAY = 0.25
# ---------------------------------------------------

def get_monitor_source():
    try:
        info = subprocess.check_output(["pactl", "info"], text=True)
        for line in info.splitlines():
            if line.startswith("Default Sink:"):
                return line.split()[-1] + ".monitor"
    except: pass
    return "auto_null.monitor" # Fallback

def freq_to_midi_float(freq):
    return 69.0 + 12.0 * math.log2(freq / 440.0)

def get_closest_target(freq):
    """Returns the MIDI note only if it's one of our targets within tolerance."""
    if freq <= 0: return 0
    midi_float = freq_to_midi_float(freq)
    midi_int = int(round(midi_float))
    
    # Check if it's a target note
    if midi_int in TARGET_NOTES:
        # Check if it's 'in tune' enough to be real
        cents_off = abs(midi_float - midi_int) * 100
        if cents_off < TOLERANCE_CENTS:
            return midi_int
    return 0

def detect_pitch_autocorr(frame):
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < ENERGY_THRESHOLD:
        return 0.0, 0.0

    frame = frame - np.mean(frame)
    windowed = frame * np.hanning(len(frame))
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[len(corr)//2:]

    min_lag = int(SAMPLE_RATE / FMAX)
    max_lag = int(SAMPLE_RATE / FMIN)
    max_lag = min(max_lag, len(corr) - 1)

    peak_idx = np.argmax(corr[min_lag:max_lag]) + min_lag
    confidence = corr[peak_idx] / (corr[0] + 1e-12)
    
    if confidence < CONFIDENCE_THRESHOLD:
        return 0.0, confidence

    return SAMPLE_RATE / peak_idx, confidence

# ----- Key sending -----
try:
    from pynput.keyboard import Controller, Key
    KBD = Controller()
    PYNPUT_AVAILABLE = True
except:
    PYNPUT_AVAILABLE = False

XDOTOOL_AVAILABLE = shutil.which("xdotool") is not None

def send_key_name(keyname):
    if PYNPUT_AVAILABLE:
        key_map = {"Left": Key.left, "Right": Key.right, "Up": Key.up, "Down": Key.down}
        if keyname in key_map:
            KBD.press(key_map[keyname])
            time.sleep(0.03)
            KBD.release(key_map[keyname])
    elif XDOTOOL_AVAILABLE:
        subprocess.run(["xdotool", "key", keyname])

def action_send_keys(sequence):
    if not sequence: return
    print(f"--- SENDING SEQUENCE: {sequence} ---")
    for midi in sequence:
        keyname = MIDI_TO_KEY.get(midi)
        if keyname:
            send_key_name(keyname)
            time.sleep(KEY_PRESS_DELAY)

def run_listener():
    monitor = get_monitor_source()
    print(f"Listening on {monitor}...")

    parec = subprocess.Popen(
        ["parec", "-d", monitor, "--format=s16le", "--rate", str(SAMPLE_RATE), "--channels", "1"],
        stdout=subprocess.PIPE, bufsize=0
    )

    smooth_buf = deque(maxlen=SMOOTH_WINDOW)
    collected = []
    active_note = 0
    stable_on_count = 0
    stable_off_count = 0

    try:
        while True:
            raw = parec.stdout.read(FRAME_SIZE * 2)
            if not raw: break

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            freq, conf = detect_pitch_autocorr(audio)
            
            # Immediately filter for our 4 target notes
            current_midi = get_closest_target(freq)
            smooth_buf.append(current_midi)
            
            # Get the most common note in our window (majority rules)
            counts = np.bincount(smooth_buf) if any(smooth_buf) else [0]
            smoothed = np.argmax(counts)

            if active_note == 0:
                # Waiting for a note to start
                if smoothed != 0:
                    stable_on_count += 1
                    if stable_on_count >= STABLE_FRAMES_FOR_ON:
                        active_note = smoothed
                        collected.append(active_note)
                        print(f"Note ON: {active_note}")
                        stable_on_count = 0
                        if len(collected) >= MAX_NOTES:
                            action_send_keys(collected)
                            collected.clear()
                else:
                    stable_on_count = 0
                    # If silent for a while, flush the buffer
                    if collected:
                        stable_off_count += 1
                        if stable_off_count > 12: # ~0.3 seconds
                            action_send_keys(collected)
                            collected.clear()
                            stable_off_count = 0
            else:
                # Waiting for note to end (silence or change)
                if smoothed != active_note:
                    stable_off_count += 1
                    if stable_off_count >= STABLE_FRAMES_FOR_OFF:
                        print(f"Note OFF: {active_note}")
                        active_note = 0
                        stable_off_count = 0
                else:
                    stable_off_count = 0

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        parec.terminate()

if __name__ == "__main__":
    run_listener()
