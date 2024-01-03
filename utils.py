import argparse
import time
import math
import array
import os
import sys
import random
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy.signal import spectrogram
import pygame
import pywt
import librosa
import multiprocessing

parser = argparse.ArgumentParser(description="Audio helper.")
functions = ['draw', 'record', 'send' ]
parser.add_argument('function', choices=functions, help="")
parser.add_argument('file', help="audio(wav) filepath (read/write)")
parser.add_argument('--frequency', help="frequency for genernating(Q1)")
parser.add_argument('--sampling_rate', help="samping rate for generating(Q1) and recording(Q3)")
parser.add_argument('--duration', help="duration for genernating(Q1) and recording(Q3)")
args = parser.parse_args()

PREAMBLE_CODE="01010101"
POSTAMBLE_CODE="0011"
STEPS_TABLE = [1, 2, 3, 5] 

PREAMBLE_TEMPLATE = {}
for step in STEPS_TABLE:
  preamble = []
  for s in PREAMBLE_CODE:
    for i in range(step):
      preamble.append(float(ord(s) - ord("0")))
  PREAMBLE_TEMPLATE[step] = preamble

PAYLOAD_SIZE = 8
MSG_SIZE = len(PREAMBLE_CODE) + len(POSTAMBLE_CODE) + PAYLOAD_SIZE
CHANNELS = 2 
SINGLE_DURATION = 0.5
CHORD_DIFF = 7
POWER_THRESHOLD = 1e-5
F_SAMPLE_RATE = 44100
MAX_SAMPLE_MSG = int(MSG_SIZE * SINGLE_DURATION * F_SAMPLE_RATE)
FRAMES_PER_BUFFER = 4096
CHUNK = 4096
  
def generate_beep(duration, frequency, sample_rate = F_SAMPLE_RATE, _dtype = np.float32):
  t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
  carrier_wave = np.sin(2 * np.pi * frequency * t)
  return np.array(carrier_wave, dtype=_dtype)

def find_nearest_patch(frequency):
  n = 12 * math.log2(frequency / 440.0) + 49
  return round(n)

def piano_patch(n):
  return (2.0 ** ( (n - 49.0) / 12.0 )) * 440

def piano_major_scale(n):
  m = 40 + (n // 7) * 12 + (n % 7) * 2
  if n % 7 >= 3:
    m -= 1
  return piano_patch(m)

def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order=4):
  nyquist = 0.5 * sample_rate
  low = lowcut / nyquist
  high = highcut / nyquist
  b, a = scipy.signal.butter(order, [low, high], btype='band', analog=False)
  filtered_data = scipy.signal.filtfilt(b, a, data)
  return filtered_data

F_MAP = [ piano_major_scale(i * CHORD_DIFF) for i in range(CHANNELS) ]

# compute difference between two template.
# YYY: for now it's O(n x len(template)), can be optimized later.
def compute_diff(signal, template):
    diff = []
    for i in range(len(signal) - len(template) + 1):
      difference = np.sum( np.abs( np.array(signal[i: i+len(template)]) - template ) )
      diff.append(difference)
    return diff

def detect_preamble(audio_data, preamble_sequence):
    correlation_result = compute_diff(audio_data, preamble_sequence)
    max_index = -1
    for i in range(len(correlation_result)):
      if correlation_result[i] == 0:
        max_index = i
        break
    if max_index == -1:
      return None
    return max_index

def decode(signal):
    signal = decode_once(signal)
    return signal

# select target frequency band
def select_frequency(fft_result, fft_freqs, power_threshold):
  stop = False
  ans = None
  cnt = 0
  for i in range(len(fft_result) // 2):
    while fft_freqs[i] > F_MAP[cnt] + 30:
      cnt += 1
      if cnt == CHANNELS:
        stop = True
        break
    if stop:
      break
    if fft_freqs[i] > F_MAP[cnt] - 30:
      v = { 's': np.abs(fft_result[i]), 'f': fft_freqs[i], 'chord': cnt}
      if ans == None or ans['s'] < v['s']:
        ans = v
  if ans is not None and ans['s'] > power_threshold:
    return ans['chord'] 
  else:
    return 1024

def save_signal(signal):
  with wave.open('denoise_xb.wav', 'wb') as wf:
    p = pyaudio.PyAudio()
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
    wf.setframerate(F_SAMPLE_RATE)
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=F_SAMPLE_RATE, input=True, output=True)
    print('Recording...')
    count = 0
    for _ in range(0, F_SAMPLE_RATE // CHUNK * 100):
      if _ * (CHUNK + 1) > len(signal):
        print("break")
        break
      print("write")
      vv = signal[_ * CHUNK : (_ + 1)* (CHUNK)]
      v = np.array(vv, dtype=np.float32)
      print(len(v), len(signal[_ * CHUNK : _ * (CHUNK + 1)]))
      wf.writeframes(v.tobytes())
      count+=1
    print('Done', count)
    wf.close()
    stream.close()
    p.terminate()
 

def decode_once(signal):
    if len(signal) < MAX_SAMPLE_MSG: 
      return signal

    # Apply the band-pass filter
    signal = butter_bandpass_filter(signal, F_MAP[0] // 2, F_MAP[-1] + 1000, F_SAMPLE_RATE, 5)

    # Apply spectral subtraction for denoising
    # D = librosa.amplitude_to_db(librosa.stft(signal), ref=np.max)
    # D2 = librosa.decompose.nn_filter(D, aggregate=np.median, metric='cosine')
    # Invert the denoised spectrogram to get the denoised signal
    # signal = librosa.griffinlim(librosa.db_to_amplitude(D2))

    # 小波变换
    # wavelet = 'db1'  
    # level = 4        
    # coeffs = pywt.wavedec(signal, wavelet, level=level)
    # threshold = 0.00005  
    # coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    # signal = pywt.waverec(coeffs_thresholded, wavelet)

    # save denoised signal.
    # save_signal(signal)

    offset = -1
    # YYY： find a proper start of the signal
    
    denoised = False
    for small_step in STEPS_TABLE:
      small_step_len = SINGLE_DURATION / small_step
      step = int(small_step_len * F_SAMPLE_RATE)

      sliced = []
      power_threshold = POWER_THRESHOLD * step
      for i in range(0, len(signal) - step,  step ):
        fft_result = np.fft.fft(signal[i : i + step])
        fft_freqs = np.fft.fftfreq(len(fft_result), 1/F_SAMPLE_RATE)
        possible_bit = select_frequency(fft_result, fft_freqs, power_threshold)
        sliced.append(possible_bit)

      preamble = PREAMBLE_TEMPLATE[small_step]
      payload_index = detect_preamble(sliced, preamble)
      if payload_index is None:
        if not denoised:
          denoised = True
          signal = librosa.effects.trim(signal, top_db=80)[0]
        continue
      else:
        step = int(SINGLE_DURATION * F_SAMPLE_RATE)
        offset = int(payload_index * small_step_len * F_SAMPLE_RATE)
        break

    if offset == -1:
      return signal[-MAX_SAMPLE_MSG:]

    print("good:", sliced[:MSG_SIZE])

    if offset + MAX_SAMPLE_MSG > len(signal):
      return signal[offset:]

    sliced = []
    power_threshold = POWER_THRESHOLD * step 
    for i in range(offset, len(signal) - step,  step ):
      fft_result = np.fft.fft(signal[i : i + step])
      fft_freqs = np.fft.fftfreq(len(fft_result), 1/F_SAMPLE_RATE)
      possible_bit = select_frequency(fft_result, fft_freqs, power_threshold)
      sliced.append(possible_bit)

    # TODO: check preamble the second time.
    print("* sliced from preamble:", sliced)
    msg = sliced[:MSG_SIZE]
    print("* payload:", msg[ len(PREAMBLE_CODE): len(PREAMBLE_CODE) + PAYLOAD_SIZE ] )
    # TODO: check postamble.
    return signal[offset + MAX_SAMPLE_MSG: ]


def draw_audio(path):
    raw = wave.open(path) 
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="float32")
    # Load the WAV file
    decode_once(signal) 

def generate_bit(index, stream, wf = None, duration = SINGLE_DURATION):
  audio = np.sum( [ generate_beep(duration, F_MAP[_index] , F_SAMPLE_RATE) for _index in index ], axis=0)
  sbytes = np.array(audio, dtype=np.float32)
  stream.write(sbytes.tobytes())
  if wf:
    wf.writeframes(sbytes.tobytes())

def send_text(payload ="0110"):
  while len(payload) % PAYLOAD_SIZE:
    payload += "0"
  p = pyaudio.PyAudio()
  stream = p.open(rate=F_SAMPLE_RATE, channels=1, format=pyaudio.paFloat32, output=True)
  # wf is used of testing, can be removed later.
  wf = wave.open('output.wav', 'wb')
  wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
  wf.setframerate(F_SAMPLE_RATE)
  wf.setnchannels(1)
  chunks = [ payload[i:i+PAYLOAD_SIZE] for i in range(0, len(payload), PAYLOAD_SIZE)]
  for chunk in chunks:
    send_text_two_bit(chunk, stream, wf)
    print("chunk:", chunk)
  stream.close()
  p.terminate()
  wf.close()

def send_text_two_bit(payload, stream, wf = None):
  seq = PREAMBLE_CODE + payload + POSTAMBLE_CODE
  for s in seq:
    generate_bit([0 if s == "0" else 1], stream, wf)

def keep_decoding(frame_queue):
  signal = []
  count = 0
  while True:
    count = 0
    while True:
      count += 1
      f = frame_queue.get()
      signal = np.append(signal, np.frombuffer(f, dtype=np.float32))
      if len(signal) / F_SAMPLE_RATE > 5:
        break
    if len(signal) > MAX_SAMPLE_MSG:
      signal = decode(signal)
      print("decode.", len(signal), signal[0])

def keep_recording(record_length = 100, debug = True):
  frame_queue = multiprocessing.Queue(maxsize=4096)
  process = multiprocessing.Process(target=keep_decoding, args=(frame_queue, ))
  process.start()

  print("keep recording...")
  with wave.open('record.wav', 'wb') as wf:
    p = pyaudio.PyAudio()
    if debug:
      wf.setnchannels(1)
      wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
      wf.setframerate(F_SAMPLE_RATE)
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=F_SAMPLE_RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
    print('Recording...')
    for _ in range(0, F_SAMPLE_RATE // CHUNK * record_length):
        v = stream.read(CHUNK)
        if debug:
          wf.writeframes(np.frombuffer(v, dtype=np.float32).tobytes())
        frame_queue.put(v)
    print('Done')
    stream.close()
    p.terminate()

if __name__ == "__main__":
  if args.function == "draw":
    draw_audio(args.file)
  elif args.function == "record":
    keep_recording()
  elif args.function == "send":
    send_text(args.file)
  else:
    print("i don't understand.")