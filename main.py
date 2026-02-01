#!/usr/bin/env python3
"""
system_note_keyer_daemon.py - Fixed for repeated notes + noise rejection
"""

import sys
import queue
import subprocess
import numpy as np
import math
import time
from collections import deque
import shutil
import argparse

from pynput.keyboard import Controller, Key

# ----------------- Config / tuning -----------------
SAMPLE_RATE = 44100
FRAME_SIZE = 1024 
FMIN, FMAX = 50.0, 2000.0

ENERGY_THRESHOLD = 0.005  
CONFIDENCE_THRESHOLD = 0.5 
TOLERANCE_CENTS = 40      

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

def get_wasapi_loopback_device():
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    wasapi_index = next(
        i for i, h in enumerate(hostapis) if h["name"] == "Windows WASAPI"
    )

    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if (
            d["hostapi"] == wasapi_index
            and d["max_output_channels"] > 0
        ):
            return i

    raise RuntimeError("No WASAPI output device found")


def get_audio_source():
    if sys.platform.startswith("linux"):
        try:
            info = subprocess.check_output(["pactl", "info"], text=True)
            for line in info.splitlines():
                if line.startswith("Default Sink:"):
                    return ("pulse", line.split()[-1] + ".monitor")
        except:
            pass
        return ("pulse", None)
    else:
        return ("sounddevice", None)

def freq_to_midi_float(freq):
    return 69.0 + 12.0 * math.log2(freq / 440.0)

def get_closest_target(freq):
    """Returns the MIDI note only if it's one of our targets within tolerance."""
    if freq <= 0: return 0
    midi_float = freq_to_midi_float(freq)
    midi_int = int(round(midi_float))
    
    if midi_int in TARGET_NOTES:
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
KBD = Controller()
KEY_MAP = {
    "Left": Key.left,
    "Right": Key.right,
    "Up": Key.up,
    "Down": Key.down,
}

def send_key_name(keyname):
    key = KEY_MAP.get(keyname)
    if not key:
        return
    KBD.press(key)
    time.sleep(0.03)
    KBD.release(key)

def action_send_keys(sequence):
    if not sequence: return
    print(f"--- SENDING SEQUENCE: {sequence} ---")
    for midi in sequence:
        keyname = MIDI_TO_KEY.get(midi)
        if keyname:
            send_key_name(keyname)
            time.sleep(KEY_PRESS_DELAY)




def run_listener():
    backend, monitor = get_audio_source()
    print(f"Audio backend: {backend}")

    smooth_buf = deque(maxlen=SMOOTH_WINDOW)
    collected = []
    active_note = 0
    stable_on_count = 0
    stable_off_count = 0

    if backend == "pulse":
        print(f"Listening on {monitor}...")
        parec = subprocess.Popen(
            ["parec", "-d", monitor, "--format=s16le",
             "--rate", str(SAMPLE_RATE), "--channels", "1"],
            stdout=subprocess.PIPE, bufsize=0
        )

        def audio_frames():
            while True:
                raw = parec.stdout.read(FRAME_SIZE * 2)
                if not raw:
                    break
                yield np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    else:

        import sounddevice as sd

        q = queue.Queue()
        def callback(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        device_index = get_wasapi_loopback_device()
        print(f"Using WASAPI loopback device #{device_index}")

        stream = sd.InputStream(
            device=device_index,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=FRAME_SIZE,
            callback=callback,
            extra_settings=sd.WasapiSettings(loopback=True),
        )
        stream.start()

        def audio_frames():
            while True:
                yield q.get()

    try:
        for audio in audio_frames():
            freq, conf = detect_pitch_autocorr(audio)
            current_midi = get_closest_target(freq)
            smooth_buf.append(current_midi)

            counts = np.bincount(smooth_buf) if any(smooth_buf) else [0]
            smoothed = np.argmax(counts)

            if active_note == 0:
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
                    if collected:
                        stable_off_count += 1
                        if stable_off_count > 12:
                            action_send_keys(collected)
                            collected.clear()
                            stable_off_count = 0
            else:
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
        if backend == "pulse":
            parec.terminate()
        else:
            stream.stop()
            stream.close()


if __name__ == "__main__":
    run_listener()
