import argparse
import datetime
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
import pywt
import pygame


def generate_beep(duration, frequency, sample_rate, _dtype = np.float32):
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

def save_signal(signal, sampling_rate, fn, CHUNK=4096):
  # print(type(signal), type(signal[0]))
  # s = np.iinfo(np.int16).max * signal
  wavfile.write(fn, sampling_rate, signal)
  # print(sampling_rate)
  # print("len:", len(signal), np.max(signal), np.min(signal))
  # with wavfile.open(fn, 'w', sample_rate = sampling_rate, num_channels=1, bits_per_sample=32, fmt=wavfile.chunk.WavFormat.IEEE_FLOAT ) as wf:
    # wf.write_float([signal])
  return
 



  with wave.open(fn, 'wb') as wf:
    p = pyaudio.PyAudio()
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
    wf.setframerate(sampling_rate)
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sampling_rate, input=True, output=True)
    print('Recording...')
    count = 0
    for _ in range(0, sampling_rate // CHUNK * 100):
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
 
def draw_fft_result(signals, base_results = None, sample_rate = 44100):
  fft_result = np.abs(np.fft.fft(signals))
  # if base_results is not None:
    # fft_result = fft_result - base_results
  fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
  fn = datetime.datetime.now().strftime("%m%d-%H-%M-%S.%f.png")
  plt.figure(1)
  plt.plot(fft_freqs, fft_result)
  plt.plot([0], [300])
  plt.savefig(fn)
  plt.close()
  # possible_bit = self.select_frequency(fft_result, fft_freqs, power_threshold)
  # sliced.append(possible_bit)
  