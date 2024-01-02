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
functions = ['generate', 'draw', 'record', 'preamble', 'send', 'receive' ]
parser.add_argument('function', choices=functions, help="choose from generate(Q1), draw(Q2), record(Q3), okk_modulate(Q4), okk_demodulate(Q4), pulse_modulate(2-Q1), pulse_demodulate(2-Q1), bpsk_modulate(2-Q2), and bpsk_modulate(2-Q2). Several testers are introduced to measure methods.")
parser.add_argument('file', help="audio(wav) filepath (read/write)")
parser.add_argument('--frequency', help="frequency for genernating(Q1)")
parser.add_argument('--phase', help="initial phase for genernating(Q1)(rad)")
parser.add_argument('--sampling_rate', help="samping rate for generating(Q1) and recording(Q3)")
parser.add_argument('--duration', help="duration for genernating(Q1) and recording(Q3)")
parser.add_argument('--support_m4a', help="use ffmpeg and pydub to convert m4a file towards wav file for drawing(Q2)", default=False)
parser.add_argument('--noise_db', help="db for noise(2-Q3)")
parser.add_argument('--with_zero', help="zero(Q4-dft-1)")
args = parser.parse_args()

if bool(args.support_m4a):
  print("Enable support for m4a file...")
  from pydub import AudioSegment
np

PREAMBLE_CODE="01010101"
POSTAMBLE_CODE="0011"
PAYLOAD_SIZE = 8
MSG_SIZE = len(PREAMBLE_CODE) + len(POSTAMBLE_CODE) + PAYLOAD_SIZE
CHANNELS = 2 
SINGLE_DURATION = 0.2
CHORD_DIFF = 7
POWER_THRESHOLD = 1e-6
MAX_SAMPLE_MSG = int(MSG_SIZE * SINGLE_DURATION * 44100)
  

def generate_beep(duration, frequency, sample_rate = 44100, _dtype = np.float32):
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
EXTEND_F_MAP = [ F_MAP[0] - 300 ]
EXTEND_F_MAP.extend( F_MAP )
EXTEND_F_MAP.append( F_MAP[-1] + 300 )
                # piano_major_scale(i * 3) for i in range(-1, CHANNELS + 1, 1) ]

def find_nearest_major_scale(frequency):
  diff = np.abs( EXTEND_F_MAP - frequency ) 
  ret = np.argmin(diff) - 1
  if ret == -1 or ret == CHANNELS:
    return 999
  return ret

def generate_audio(output_file = "1.wav", sampling_rate = 44100, frequency = 400, initial_phase = 0, duration = 5):
  t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
  signal = np.sin(2 * np.pi * frequency * t + initial_phase) 
  signal = (signal * 32767).astype(np.int16) # covert to np.int16
  wavfile.write(output_file, int(sampling_rate), signal)
  print(f"Audio genertaed. Saved to {output_file}")

def record_audio(wav_file="recorded_audio.wav", duration=5, sample_rate=44100, channels=1):
    audio = pyaudio.PyAudio()
    format = pyaudio.paInt16
    frames_per_buffer = 1024
    stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=frames_per_buffer)
    print("Start recording...")
    frames = []
    for _ in range(0, int(sample_rate / frames_per_buffer * duration)):
        data = stream.read(frames_per_buffer)
        frames.append(data)
    print("Stop recording...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(wav_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    print(f"Audio recorded. Saved to {wav_file}")

def pulse_demodulation(wav_file = "pulse.wav", sample_rate = 48000, zero_duration = 0.020, one_duration = 0.030, pulse_duration = 0.010):
    with wave.open(wav_file, 'rb') as wf:
        frames = wf.readframes(-1)
    samples = np.frombuffer(frames, dtype=np.int16)
    samples = np.abs(scipy.signal.hilbert(samples))
    demodulated_signal = []
    amplitude_threhold = 0.5 * 32767
    interval_threhold =  sample_rate * 1 * ((one_duration + pulse_duration) + (zero_duration + pulse_duration)) / 2 # compute the interval threhold for 1 and 0
    start = 0
    for i in range(1, len(samples)):
        # look for the pulse start.
        if samples[i] > amplitude_threhold and samples[i-1] < amplitude_threhold:
          if i - start > interval_threhold:
            demodulated_signal.append(1)
          else:
            demodulated_signal.append(0)
          start = i
    return demodulated_signal

def cross_correlation(signal, template):
    # Perform cross-correlation between the signal and the template
    # correlation_result = np.correlate(signal, template, mode='full')
    # return correlation_result
    diff = []
    for i in range(len(signal) - len(template) + 1):
      difference = np.sum( np.abs( np.array(signal[i: i+len(template)]) - template ) )
      diff.append(difference)
    return diff

def detect_preamble(audio_data, preamble_sequence):
    # Perform cross-correlation to detect preamble
    correlation_result = cross_correlation(audio_data, preamble_sequence)
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


def decode_once(signal):
    if len(signal) < MAX_SAMPLE_MSG: 
      return signal

    # Set the cutoff frequencies for the band-pass filter
    lowcut = 100  # Adjust as needed
    highcut = 5000  # Adjust as needed
    
    # Apply the band-pass filter
    signal = butter_bandpass_filter(signal, lowcut, highcut, 44100, 5)
    # Apply spectral subtraction for denoising
    # D = librosa.amplitude_to_db(librosa.stft(signal), ref=np.max)
    # D2 = librosa.decompose.nn_filter(D, aggregate=np.median, metric='cosine')
    # Invert the denoised spectrogram to get the denoised signal
    # signal = librosa.griffinlim(librosa.db_to_amplitude(D2))
    # wavelet = 'db1'  # 选择小波基函数
    # level = 4        # 分解的层数
    # coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # # 阈值处理
    # threshold = 0.00005  # 阈值，需要根据实际情况调整
    # coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    
    # # 小波逆变换
    # signal = pywt.waverec(coeffs_thresholded, wavelet)

    # with wave.open('denoise_xb.wav', 'wb') as wf:
    #   p = pyaudio.PyAudio()
    #   wf.setnchannels(1)
    #   wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
    #   wf.setframerate(44100)
    #   stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, output=True)
    #   print('Recording...')
    #   count = 0
    #   for _ in range(0, 44100 // CHUNK * 100):
    #     if _ * (CHUNK + 1) > len(signal):
    #       print("break")
    #       break
    #     print("write")
    #     vv = signal[_ * CHUNK : (_ + 1)* (CHUNK)]
    #     v = np.array(vv, dtype=np.float32)
    #     print(len(v), len(signal[_ * CHUNK : _ * (CHUNK + 1)]))
    #     wf.writeframes(v.tobytes())
    #     count+=1
    #   print('Done', count)
    #   wf.close()
    #   stream.close()
    #   p.terminate()
    
    f_rate = 44100
    time = np.linspace( 0, len(signal) / f_rate, num = len(signal)) 
    offset = -1
    steps_table = [1, 1, 2, 3, 5]
    denoised = False
    for small_step in steps_table:
      SMALL_STEP = small_step
      SMALL_STEP_LEN = SINGLE_DURATION / SMALL_STEP
      
      step = int(SMALL_STEP_LEN * 44100)
      sliced = []
      candidates = []
      power_seq = []
      t2 = []
      v2 = []
      count = 0
      for i in range(0, len(signal) - step,  step ):
        fft_result = np.fft.fft(signal[i : i + step])
        fft_freqs = np.fft.fftfreq(len(fft_result), 1/44100)
        # 找到主频率
        main_frequency = np.abs(fft_freqs[np.argmax(np.abs(fft_result))])
        power = np.max(np.abs(fft_result)) 
        possible_bit = 999
        if power < POWER_THRESHOLD:
          main_frequency = 0.0
          power = 0.0
        else:
          possible_bit = find_nearest_major_scale(main_frequency)
        candidates.append(count)
        power_seq.append(power)
        sliced.append(possible_bit)
        count+=1

      preamble = []
      for s in PREAMBLE_CODE:
        for i in range(SMALL_STEP):
          preamble.append(0.0 if s == "0" else 1.0)

      payload_index = detect_preamble(sliced, preamble)
      if payload_index is None:
        if not denoised:
          denoised = True
          # signal = librosa.effects.trim(signal, top_db=40)[0]
          # print("too much moise: let us denoise.")
        # plt.figure(1)
        # plt.title("Sound Wave")
        # plt.xlabel("Time")
        # plt.plot(time, signal)
        # # plt.plot(t2, v2, color='red')
        # plt.show()
        # print(SMALL_STEP, "no preamble", preamble, "\n", sliced)
        continue
      else:
        step = int(SINGLE_DURATION * 44100)
        offset = int(payload_index * SMALL_STEP_LEN * 44100)
        break
    if offset == -1:
      return signal[-MAX_SAMPLE_MSG:]
    print("good:", sliced)
    if offset + MAX_SAMPLE_MSG > len(signal):
      return signal[offset:]

    sliced = []
    for i in range(offset, len(signal) - step,  step ):
      fft_result = np.fft.fft(signal[i : i + step])
      fft_freqs = np.fft.fftfreq(len(fft_result), 1/44100)
      # 找到主频率
      main_frequency = np.abs(fft_freqs[np.argmax(np.abs(fft_result))])
      power = np.max(np.abs(fft_result)) 
      possible_bit = 999
      if power < POWER_THRESHOLD:
        main_frequency = 0.0
        power = 0.0
      else:
        possible_bit = find_nearest_major_scale(main_frequency)
      candidates.append(count)
      power_seq.append(power)
      t2.append(time[i])
      v2.append(power)
      sliced.append(possible_bit)
      count+=1
    # * check preamble twice.
    print("sliced:", sliced)
    msg = sliced[:MSG_SIZE]
    print("payload:", msg[ len(PREAMBLE_CODE): len(PREAMBLE_CODE) + PAYLOAD_SIZE ] )

    # msg = sliced[:MSG_SIZE]

    return signal[offset + MAX_SAMPLE_MSG: ]


def draw_audio(path):
    if bool(args.support_m4a) and path.endswith(".m4a"):
      m4a_file = AudioSegment.from_file(path, format="m4a")
      m4a_file.export(path + ".wav", format="wav")
      raw = wave.open(path + ".wav")
    else:
      raw = wave.open(path) 
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="float32")
    # Load the WAV file
    decode_once(signal) 


def generate_bit(index, stream, wf = None, duration = SINGLE_DURATION):
  audio = np.sum( [ generate_beep(duration, F_MAP[_index] , 44100) for _index in index ], axis=0)
  sbytes = np.array(audio, dtype=np.float32)
  stream.write(sbytes.tobytes())
  if wf:
    wf.writeframes(sbytes.tobytes())

def send_text(payload ="0110"):
  while len(payload) % PAYLOAD_SIZE:
    payload += "0"
  p = pyaudio.PyAudio()
  stream = p.open(rate=44100, channels=1, format=pyaudio.paFloat32, output=True)
  # wf is used of testing, can be removed later.
  wf = wave.open('output.wav', 'wb')
  wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
  wf.setframerate(44100)
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
      if len(signal) / 44100 > 5:
        break
    if len(signal) > MAX_SAMPLE_MSG:
      signal = decode(signal)
      print("decode.", len(signal), signal[0])

def keep_recording(record_length = 100):
  frame_queue = multiprocessing.Queue(maxsize=4096)
  process = multiprocessing.Process(target=keep_decoding, args=(frame_queue, ))
  process.start()

  print("keep recording...")
  with wave.open('record.wav', 'wb') as wf:
    p = pyaudio.PyAudio()
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
    wf.setframerate(44100)
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=4096)
    print('Recording...')
    CHUNK = 4096
    frame = []
    for _ in range(0, 44100 // CHUNK * record_length):
        v = stream.read(CHUNK)
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