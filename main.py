import argparse
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

parser = argparse.ArgumentParser(description="Audio helper.")
functions = ['generate', 'draw', 'record', 'ook_modulate', 'ook_demodulate', 'pulse_modulate', 'pulse_demodulate', 'bpsk_modulate', 'bpsk_demodulate', 'q2_test1', 'q2_test2', 'qpsk_modulate', 'qpsk_demodulate', 'q3_test1', 'q3_test2', 'q3_dft', 'dft', 'cc', 'short_dft', 'preamble' ]
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


def ook_modulation(wav_file = "ook.wav", input_signal = "010011101100101", amplitude=1, sample_rate=48000, frequency=20000, duration=0.025):
  input_signal = "010011101100101"
  t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
  signal = np.array([amplitude if bit == '1' else 0 for bit in input_signal])
  carrier_wave = np.sin(2 * np.pi * frequency * t)
  modulated_signal = np.zeros(0)
  for bit in signal:
      modulated_signal = np.append(modulated_signal, bit * carrier_wave)
  modulated_signal = np.int16(modulated_signal * 32767)
  wavfile.write(wav_file, 48000, modulated_signal)
  print(f"Modulation over. Saved to {wav_file}")

def ook_demodulation(wav_file = "ook.wav", threshold = 40, sample_rate = 48000, symbol_duration = 0.025):
    samples_per_symbol = int(sample_rate * (symbol_duration))
    with wave.open(wav_file, 'rb') as wf:
        frames = wf.readframes(-1)
    samples = np.frombuffer(frames, dtype=np.int16)
    demodulated_signal = []
    for i in range(0, len(samples), samples_per_symbol):
        symbol_samples = np.abs(samples[i:i + samples_per_symbol])
        average_amplitude = np.mean(symbol_samples)
        if average_amplitude >= threshold:
            demodulated_signal.append(1)
        else:
            demodulated_signal.append(0)
    print(f"Demodulation over.")
    return demodulated_signal
  
def pulse_modulation(wav_file = "pulse.wav", input_signal = "010011101100101", amplitude=1, sample_rate=48000, frequency=20000, pulse_duration=0.010, zero_duration=0.020, one_duration=0.030, noise_db=None):
  pulse_t = np.linspace(0, pulse_duration, int(sample_rate * pulse_duration), endpoint=False)
  zero_t = np.linspace(0, zero_duration, int(sample_rate * zero_duration), endpoint=False) * 0.0
  one_t = np.linspace(0, one_duration, int(sample_rate * one_duration), endpoint=False) * 0.0
  pulse_wave = np.sin(2 * np.pi * frequency * pulse_t) * amplitude
  modulated_signal = np.zeros(0)
  for bit in input_signal:
    modulated_signal = np.append(modulated_signal, pulse_wave) # the pulse
    if bit == "1":
      modulated_signal = np.append(modulated_signal, one_t) # the one's interval
    else:
      modulated_signal = np.append(modulated_signal, zero_t) # the zero's interval
  modulated_signal = np.append(modulated_signal, pulse_wave) # end the signal
  if noise_db != None:
    # add AWGN
    a = np.var(modulated_signal) # compute signal's power
    noise_power = a / (10 ** (noise_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(modulated_signal))
    modulated_signal = modulated_signal + noise
  modulated_signal = np.int16(modulated_signal * 32767)
  wavfile.write(wav_file, sample_rate, modulated_signal)
  print(f"Modulation over. Saved to {wav_file}")

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

def bpsk_modulation(wav_file = "bpsk.wav", input_signal = "0100111011001010", amplitude=1, sample_rate=48000, frequency=20000, symbol_duration=0.025, noise_db=None):
  n_t = np.linspace(0, symbol_duration, int(sample_rate * symbol_duration), endpoint=False)
  # compute base wave
  base_wave = np.sin(2 * np.pi * frequency * n_t) * amplitude
  modulated_signal = np.zeros(0)
  for bit in input_signal:
    pa = 1 - 2 * (1 if bit == "1" else 0) # reverse if the bit is one
    modulated_signal = np.append(modulated_signal, base_wave * pa)
  if noise_db != None:
    # add AWGN
    a = np.var(modulated_signal) # compute signal power
    noise_power = a / (10 ** (noise_db / 10)) # compute noise power
    noise = np.random.normal(0, np.sqrt(noise_power), len(modulated_signal))
    modulated_signal = modulated_signal + noise
  modulated_signal = np.int16(modulated_signal * 32767)
  wavfile.write(wav_file, sample_rate, modulated_signal)
  print(f"Modulation over. Saved to {wav_file}")
  return modulated_signal

def bpsk_demodulation(wav_file = "pulse.wav", sample_rate = 48000, symbol_duration = 0.025, frequency = 20000, amplitude = 1):
    with wave.open(wav_file, 'rb') as wf:
        frames = wf.readframes(-1)
    samples = np.frombuffer(frames, dtype=np.int16) / 32767
    # compute metadata
    samp_per = np.int16(symbol_duration * sample_rate)
    num_symb = np.int16(np.floor(len(samples) / (symbol_duration * sample_rate)))
    # compute base wave
    n_t = np.linspace(0, symbol_duration, int(sample_rate * symbol_duration), endpoint=False)
    t = np.tile(n_t, num_symb + 1)[:len(samples)]
    base_wave = np.sin(2 * np.pi * frequency * t) * amplitude 
    # multiply the basewave with sample
    r = base_wave * samples
    # lowoass
    cutoff = 50 / (0.5 * sample_rate)
    b, a = scipy.signal.butter(2, cutoff, btype='low', analog=False)
    filtered_r = scipy.signal.lfilter(b, a, r)
    # demodulate
    demodulated_signal = []
    for i in range(0, num_symb):
      if np.mean(filtered_r[i * samp_per: (i + 1) * samp_per - 1]) < 0:
          demodulated_signal.append(1)
      else:
          demodulated_signal.append(0)
    return demodulated_signal
    
def qpsk_modulation(wav_file = "qpsk.wav", input_signal = "0100111011001010", amplitude=1, sample_rate=48000, frequency=20000, symbol_duration=0.025, noise_db=None):
  n_t = np.linspace(0, symbol_duration, int(sample_rate * symbol_duration), endpoint=False)

  if len(input_signal) % 2:
    input_signal.append('0')

  # compute base wave I, Q
  base_wave_i = np.sin(2 * np.pi * frequency * n_t) * amplitude
  base_wave_q = np.cos(2 * np.pi * frequency * n_t) * amplitude

  modulated_signal = np.zeros(0)

  def bitwrap(c):
    return 0 if c == '0' else 1

  for i in range(len(input_signal)//2):
    fi = (1 - 2 * bitwrap(input_signal[i * 2])) * np.sqrt(2) / 2;
    fq = (1 - 2 * bitwrap(input_signal[i * 2 + 1])) * np.sqrt(2) / 2;
    # compute I, Q
    modulated_signal = np.append(modulated_signal, base_wave_i * fi + base_wave_q * fq)

  if noise_db != None:
    # add AWGN
    a = np.var(modulated_signal) # compute signal power
    noise_power = a / (10 ** (noise_db / 10)) # compute noise power
    noise = np.random.normal(0, np.sqrt(noise_power), len(modulated_signal))
    modulated_signal = modulated_signal + noise

  modulated_signal = np.int16(modulated_signal * 32767)
  wavfile.write(wav_file, sample_rate, modulated_signal)
  print(f"Modulation over. Saved to {wav_file}")

def qpsk_demodulation(wav_file = "qpsk.wav", sample_rate = 48000, symbol_duration = 0.025, frequency = 20000, amplitude = 1):
    with wave.open(wav_file, 'rb') as wf:
        frames = wf.readframes(-1)

    samples = np.frombuffer(frames, dtype=np.int16) / 32767

    # compute metadata
    samp_per = np.int16(symbol_duration * sample_rate)
    num_symb = np.int16(np.floor(len(samples) / (symbol_duration * sample_rate)))

    # compute base wave I, Q
    n_t = np.linspace(0, symbol_duration, int(sample_rate * symbol_duration), endpoint=False)
    t = np.tile(n_t, num_symb + 1)[:len(samples)]
    base_wave_i = np.sin(2 * np.pi * frequency * t) * amplitude 
    base_wave_q = np.cos(2 * np.pi * frequency * t) * amplitude 

    # multiply the basewave with sample
    # lowoass
    cutoff = 50 / (0.5 * sample_rate)
    b, a = scipy.signal.butter(2, cutoff, btype='low', analog=False)

    r_i = base_wave_i * samples
    filtered_r_i = scipy.signal.lfilter(b, a, r_i)

    r_q = base_wave_q * samples
    filtered_r_q = scipy.signal.lfilter(b, a, r_q)
    # demodulate

    demodulated_signal_i = []
    for i in range(0, num_symb):
      if np.mean(filtered_r_i[i * samp_per: (i + 1) * samp_per - 1]) < 0:
          demodulated_signal_i.append(1)
      else:
          demodulated_signal_i.append(0)

    demodulated_signal_q = []
    for i in range(0, num_symb):
      if np.mean(filtered_r_q[i * samp_per: (i + 1) * samp_per - 1]) < 0:
          demodulated_signal_q.append(1)
      else:
          demodulated_signal_q.append(0)
    
    # combine the result of I Q
    demodulated_signal = []
    for i in range(0, num_symb):
      demodulated_signal.append(demodulated_signal_i[i])
      demodulated_signal.append(demodulated_signal_q[i])
    
    return demodulated_signal
 

def draw_audio(path):
    if bool(args.support_m4a) and path.endswith(".m4a"):
      m4a_file = AudioSegment.from_file(path, format="m4a")
      m4a_file.export(path + ".wav", format="wav")
      raw = wave.open(path + ".wav")
    else:
      raw = wave.open(path) 
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
    f_rate = raw.getframerate()
    time = np.linspace( 0, len(signal) / f_rate, num = len(signal)) 
    print(time)
    plt.figure(1)
    plt.title("Sound Wave")
    plt.xlabel("Time")
    plt.plot(time, signal)
    plt.show()

def test_module(module_function, demodule_function, module_args, demodule_args, input_len = 20, test_times = 10 ):
  sys.stdout = open(os.devnull, 'w')
  def genRandom01(_length):
    binary_string = ''.join(str(random.choice([0, 1])) for _ in range(_length))
    return binary_string
  input = [ genRandom01(input_len) for i in range(test_times) ] # generate test data
  good = 0
  for i in input:
    module_function(wav_file = "_test_module.wav", input_signal = i, **module_args)
    ret = demodule_function(wav_file = "_test_module.wav", **demodule_args)
    ret = ''.join(map(str, ret))
    shared = min(len(i), len(ret))
    for j in range(shared): # compute correct bits for i(the input) and ret(the demodulated data)
      if ret[j] == i[j]:
        good += 1
    print(good)
  sys.stdout = sys.__stdout__
  return good / (test_times * input_len)

# For Q2 test 3-a)
def q2_test1(png_file="test.png"): 
  pulse_ans = {}
  bpsk_ans = {}
  for db in range(0, 20):
    pulse_ans[db] = test_module(pulse_modulation, pulse_demodulation, {"noise_db": db}, {})
    bpsk_ans[db] = test_module(bpsk_modulation, bpsk_demodulation, {"noise_db": db}, {})
  plt.figure(1)
  plt.title("Correction Rate")
  plt.xlabel("Time")
  plt.plot(list(pulse_ans.keys()), list(pulse_ans.values()), label = "pulse")
  plt.plot(list(bpsk_ans.keys()), list(bpsk_ans.values())  , label = "bpsk")
  plt.legend()
  plt.savefig(png_file)

# For Q2 test 3-b)  
def q2_test2(png_file="test.png"): 
  bpsk_duration_ans = {}
  bpsk_duration_ans2 = {}
  for rate in range(1, 16):
    bpsk_duration_ans[25 * rate] = test_module(bpsk_modulation, bpsk_demodulation, {"noise_db": 10, "symbol_duration" : 0.025 * rate}, {"symbol_duration" : 0.025 * rate})
    bpsk_duration_ans2[25 * rate] = test_module(bpsk_modulation, bpsk_demodulation, {"noise_db": 0, "symbol_duration" : 0.025 * rate}, {"symbol_duration" : 0.025 * rate})
  print(bpsk_duration_ans)
  plt.figure(1)
  plt.title("Correction Rate")
  plt.xlabel("Duration Time(ms)")
  plt.plot(list(bpsk_duration_ans.keys()), list(bpsk_duration_ans.values()),   label = "DB=10")
  plt.plot(list(bpsk_duration_ans2.keys()), list(bpsk_duration_ans2.values()), label = "DB=0")
  plt.legend()
  plt.savefig(png_file)

# For Q3 test 2-a)
def q3_test1(png_file="test.png"): 
  pulse_ans = {}
  bpsk_ans = {}
  qpsk_ans = {}
  for db in range(-10, 20):
    pulse_ans[db] = test_module(pulse_modulation, pulse_demodulation, {"noise_db": db}, {})
    bpsk_ans[db] = test_module(bpsk_modulation, bpsk_demodulation, {"noise_db": db}, {})
    qpsk_ans[db] = test_module(qpsk_modulation, qpsk_demodulation, {"noise_db": db}, {})
  plt.figure(1)
  plt.title("Correction Rate")
  plt.xlabel("db")
  plt.plot(list(pulse_ans.keys()), list(pulse_ans.values()), label = "pulse")
  plt.plot(list(bpsk_ans.keys()), list(bpsk_ans.values())  , label = "bpsk")
  plt.plot(list(qpsk_ans.keys()), list(qpsk_ans.values())  , label = "qpsk")
  plt.legend()
  plt.savefig(png_file)

# For Q3 test 2-b)  
def q3_test2(png_file="test.png"): 
  qpsk_duration_ans = {}
  qpsk_duration_ans2 = {}
  for rate in range(1, 16):
    qpsk_duration_ans[25 * rate] = test_module(qpsk_modulation, qpsk_demodulation, {"noise_db": 10, "symbol_duration" : 0.025 * rate}, {"symbol_duration" : 0.025 * rate})
    qpsk_duration_ans2[25 * rate] = test_module(qpsk_modulation, qpsk_demodulation, {"noise_db": 0, "symbol_duration" : 0.025 * rate}, {"symbol_duration" : 0.025 * rate})
  print(qpsk_duration_ans)
  plt.figure(1)
  plt.title("Correction Rate")
  plt.xlabel("Duration Time(ms)")
  plt.plot(list(qpsk_duration_ans.keys()), list(qpsk_duration_ans.values()),   label = "DB=10")
  plt.plot(list(qpsk_duration_ans2.keys()), list(qpsk_duration_ans2.values()), label = "DB=0")
  plt.legend()
  plt.savefig(png_file)

def compute_dft(x):
    return np.fft.fft(x)
    N = len(x)
    X = []
    for k in range(N):
      v = 0.0
      for n in range(N):
        v += x[n] * np.exp( -2j * np.pi * k * n / N)
      X.append(v)
    return X

def dft(N=64, f=(lambda x: 1.0)):
  r = []
  for i in range(N):
    r.append(f(i))
  ret = compute_dft(r)
  ret = np.array(np.abs(ret) / N ) # 1 / N normalized dft
  x = []
  y = []
  for i in range(N):
    x.append(i)
    y.append(ret[i])
  return x, y
    
def q3_test_dft(file="a.png"):
  N_choices = [16, 64, 1024]
  func_choices = [lambda n: 1, lambda n: 1 - np.abs(n) / N, lambda n: np.sin(2 * np.pi * n / N) ]
  func_name = [ "y(n) = 1", "y(n) = 1 - |n| /N ", "y(n) = sin( 2*pi*n / N )" ]
  plt.figure(figsize=(12 ,8))
  index = 0
  for N in N_choices:
    for func in func_choices:
      index += 1
      plt.subplot(len(N_choices), len(func_choices), index)
      x, y = dft(N, func)
      plt.plot(x, y)
      plt.xlabel('Frequency')
      plt.ylabel('Amplitude')
      plt.title(f'N = {N}, func = {func_name[(index - 1)% 3]}')
      plt.grid(True)
  plt.tight_layout()
  plt.savefig(file)

def read_and_dft(wav_file="a.png", size_with_zero = 1):
  with wave.open(wav_file, 'rb') as wf:
      frames = wf.readframes(-1)
  samples = np.frombuffer(frames, dtype=np.int16) / 32767
  size = len(samples)
  x, y = dft(size_with_zero * size, lambda i: (samples[i] if i < size else 0.0))
  plt.plot(x, y)
  plt.xlabel('Frequency')
  plt.ylabel('Amplitude')
  # plt.title(f'N = {N}, func = {func_name[(index - 1)% 3]}')
  plt.grid(True)
  plt.savefig(wav_file + "." + str(size_with_zero) + ".png")

def short_dft(wav_file="res.wav"):
  with wave.open(wav_file, 'rb') as wf:
    frames = wf.readframes(-1)
  samples = np.frombuffer(frames, dtype=np.int16) / 32767
  for i in range(12):
    plt.subplot(3, 4, i + 1)
    window_length = (i + 1) * 10
    f, t, Sxx = spectrogram(samples, nperseg=window_length, noverlap=window_length*0.5 )
    # 绘制时频图
    plt.pcolormesh(t, f, 10*np.log10(Sxx))
    plt.title(f'window = { window_length }')
  plt.tight_layout()
  plt.savefig(wav_file + ".spec.png")

def circular_convolution(x, y):
  x = np.fft.fft(x)
  y = np.fft.fft(y)
  z = x * y
  z = np.real(np.fft.ifft(z))
  return z

def read_and_cc(mode):
  if mode == "input":
    print("input sequence size> ")
    n = int(input())
    x = []
    y = []
    print("input A[0..n) in one line:")
    x = [ float(a) for a in input().split(" ")]
    print("input B[0..n) in one line:")
    y = [ float(a) for a in input().split(" ")]
    if len(x) != n or len(y) != n:
      print("size unmatch")
      return
    z = circular_convolution(x, y)
    print("result:", z)
  else:
    g = [3,2, -2,1,0,1]
    h = [-5, -1,3, -2,4,4]
    print("g x h: ", circular_convolution(g, h))
    x = [ np.cos( np.pi * n / 2.0 ) for n in range(0, 4+1) ]
    y = [ 3.0 ** n for n in range(0, 4+1) ]
    print("x x y: ", circular_convolution(x, y))

def add_preamble(audio_data, preamble):
    preamble_data = array.array('h', preamble)
    preamble_data.extend(audio_data)
    return preamble_data

def cross_correlation(signal, template):
    # Perform cross-correlation between the signal and the template
    correlation_result = np.correlate(signal, template, mode='full')
    return correlation_result

def detect_preamble(audio_data, preamble_sequence):
    # Perform cross-correlation to detect preamble
    correlation_result = cross_correlation(audio_data, preamble_sequence)

    print(audio_data, preamble_sequence)
    print(np.abs(correlation_result))

    # Find the index where the correlation is maximum
    max_index = np.argmax(np.abs(correlation_result))

    # Extract the presumed preamble based on the maximum correlation index
    detected_preamble = audio_data[max_index + 1:]

    return detected_preamble

def with_preamble(file):
  seq = "1111000011110000"
  ret = bpsk_modulation(file, seq + "0101")
  de_ret = bpsk_demodulation(file)
  print(de_ret)
  pygame.mixer.pre_init(frequency=20000, size=-16, channels=2)
  pygame.init()
  pygame.mixer.init()
  print(pygame.mixer.get_init())
  a = np.repeat(ret.reshape(len(ret), 1), 2, axis = 1)
  print(len(ret))
  sound = pygame.sndarray.make_sound(a)
  sound.play()
  pygame.time.wait(int(sound.get_length() * 1000))
  print("stop.")
  print("length:", sound.get_length())
  de_p = detect_preamble(de_ret, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
  print(de_p)


if __name__ == "__main__":
  if args.function == "generate":
    if args.duration is None or args.sampling_rate is None or args.frequency is None or args.phase is None:
      print("Error: Arguments missing.")
      exit(1)
    generate_audio(args.file, int(args.sampling_rate), int(args.frequency), float(args.phase), float(args.duration))
  elif args.function == "draw":
    draw_audio(args.file)
  elif args.function == "record":
    if args.duration is None or args.sampling_rate is None:
      print("Error: Arguments missing.")
      exit(1)
    record_audio(args.file, float(args.duration), int(args.sampling_rate))
  elif args.function == "ook_modulate":
    ook_modulation(args.file)
  elif args.function == "ook_demodulate":
    print(ook_demodulation(args.file))
  elif args.function == "pulse_modulate":
    pulse_modulation(args.file)
  elif args.function == "pulse_demodulate":
    print(pulse_demodulation(args.file))
  elif args.function == "bpsk_modulate":
    bpsk_modulation(args.file, noise_db=int(args.noise_db) if args.noise_db else None)
  elif args.function == "bpsk_demodulate":
    print(bpsk_demodulation(args.file))
  elif args.function == "q2_test1":
    q2_test1(args.file)
  elif args.function == "q2_test2":
    q2_test2(args.file)
  elif args.function == "qpsk_modulate":
    qpsk_modulation(args.file, noise_db=int(args.noise_db) if args.noise_db else None)
  elif args.function == "qpsk_demodulate":
    print(qpsk_demodulation(args.file))
  elif args.function == "q3_test1":
    q3_test1(args.file)
  elif args.function == "q3_test2":
    q3_test2(args.file)
  elif args.function == "q3_dft":
    q3_test_dft(args.file)
  elif args.function == "dft":
    read_and_dft(args.file, size_with_zero = int(args.with_zero) if args.with_zero else 1)
  elif args.function == "cc":
    read_and_cc(args.file)
  elif args.function == "short_dft":
    short_dft(args.file)
  elif args.function == "preamble":
    with_preamble(args.file)
  else:
    print("i don't understand.")