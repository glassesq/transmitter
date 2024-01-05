from utils import *

import multiprocessing
import pyaudio
import wave
import numpy as np
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


class Player:

  CHUNK = 4096
  def __init__(self):
    self.F_SAMPLE_RATE = 44100
    self.FRAMES_PER_BUFFER = 4096
    self.SINGLE_DURATION = 0.05

    self.switch = 0
    self.count = 0

    # message layout
    # YYY: preamble code + payload + postamble
    self.PAYLOAD_SIZE = 50
    self.PREAMBLE_CODE="1112223344"
    self.POSTAMBLE_CODE="4433322111"
    self.MSG_SIZE = len(self.PREAMBLE_CODE) + len(self.POSTAMBLE_CODE) + self.PAYLOAD_SIZE

    self.MIN_SAMPLE_MSG = int(self.MSG_SIZE * self.SINGLE_DURATION * self.F_SAMPLE_RATE)
    self.MAX_SAMPLE_MSG = int(self.MSG_SIZE * self.SINGLE_DURATION * self.F_SAMPLE_RATE)

    # TODO: remove steps.
    self.STEPS_TABLE = [1, 2, 3, 5] 
    
    # for step in self.STEPS_TABLE:


    # Use piano chords as frequency map 
    # YYY: it is not the best frequency for transmitting, but the best for my ears when testing.
    self.JUMP_FREQ_AREA_PER_SYMBOL = 2 # Increase this number to get more accurate result.
    self.SYMBOL_PER_DURATION = 3 # Increase this number to transfer more info at one duration
    self.CHANNELS = 8 # Increaase this number to transfer more info at one duration.
    self.FGAP = 100 # Increase this number to get more accurate result.
    self.CHORD_DIFF = 8 # 

    preamble = []
    for s in self.PREAMBLE_CODE:
      for i in range(self.SYMBOL_PER_DURATION):
        preamble.append(float(ord(s) - ord("0")))
      # self.PREAMBLE_TEMPLATE[step] = preamble
    self.PREAMBLE_TEMPLATE = preamble

    self.SYMBOL_PER_PAYLOAD = self.PAYLOAD_SIZE * self.SYMBOL_PER_DURATION

    self.POWER_THRESHOLD = 5.0 / self.F_SAMPLE_RATE
    # self.F_MAP = [ piano_major_scale(i * self.CHORD_DIFF) for i in range(self.CHANNELS) ]

    self.BASE_FREQ = 1000
    self.GAP_RATE = 2.2 # Increase this number to get more accurate result
    self.AREA_GAP = int( self.FGAP * self.CHANNELS * self.GAP_RATE * 1.15 )
    self.F_MAP = []
    for a in range(self.JUMP_FREQ_AREA_PER_SYMBOL * self.SYMBOL_PER_DURATION):
      self.F_MAP.extend([ self.BASE_FREQ + self.AREA_GAP * a + i * (self.FGAP * self.GAP_RATE) for i in range(self.CHANNELS) ])
    print(self.F_MAP)

    self.debug = False

  def generate_bit(self, index, stream, duration, sampling_rate, wf = None):
    audio = np.sum( [ generate_beep(duration, self.F_MAP[_index] , sampling_rate) for _index in index ], axis=0)
    sbytes = np.array(audio, dtype=np.float32)
    stream.write(sbytes.tobytes())
    if wf:
      wf.writeframes(sbytes.tobytes())

  # select target frequency band
  # YYY: Given fft_result, fft_reqs and power_threshold, map literal signals towards possible bit in frequency map. If the strength of that frequency is too weak, consider it as a noise.
  def select_frequency(self, fft_result, fft_freqs, power_threshold, jump_index, symbol_index = 0) :
    stop = False
    ans = None
    chord = 0
    for i in range(len(fft_result) // 2):
      while fft_freqs[i] > self.compute_target_frequency(chord, jump_index, symbol_index) + self.FGAP:
        chord += 1
        if chord == self.CHANNELS:
          stop = True
          break
      if stop:
        break
      if fft_freqs[i] > self.compute_target_frequency(chord, jump_index, symbol_index) - self.FGAP:
        v = { 's': np.abs(fft_result[i]), 'f': fft_freqs[i], 'chord': chord}
        if ans == None or ans['s'] < v['s']:
          ans = v
    if ans is not None and ans['s'] > power_threshold:
      return ans['chord'] 
    else:
      return 1024
  
  # TODO: with a better encoding algorithm.
  def encode_payload(self, payload):
    ec = []
    for c in payload:
      if ord(c) <= ord('Z') and ord(c) >= ord('A'): # directly map.
        v = ord(c) - ord('A') + 1
        ec.extend( [ v // 8, v % 8 ] )
        # ec.append(ord(c) - ord('A') + 1)
      elif ord(c) <= ord('z') and ord(c) >= ord('a'):
        v = ord(c) - ord('a') + 27
        ec.extend( [ v // 8, v % 8 ] )
        # ec.append(ord(c) - ord('a') + 1 + 26)
      else:
        ec.append( [ 7, 7 ] )
    # TODO!
    return [ e % self.CHANNELS for e in ec ]

  # TODO: with a better encoding algorithm.
  def decode_payload(self, payload):
    ec = ""
    for i in range(0, len((payload)), 2):
      c = payload[i] * 8 + payload[i + 1]
      if c <= 26 and c >= 1: # directly map.
        ec += chr(c + ord('A') - 1)
      elif c <= 26 + 26 and c >= 26 + 1:
        ec += chr(c + ord('a') - 27)
      else:
        ec += '.'
    # TODO!
    return ec

  # Freqency index: symbol_index * self.CHANNELS * self.FREQ_AREA_PER_SYMBOL + jump_index * self.CHANNELS + chord
  def compute_target_frequency(self, chord, jump_index, symbol_index):
    return self.F_MAP[symbol_index * self.CHANNELS * self.JUMP_FREQ_AREA_PER_SYMBOL + jump_index * self.CHANNELS + chord]

  def compute_target_frequency_index(self, chord, jump_index, symbol_index):
    return symbol_index * self.CHANNELS * self.JUMP_FREQ_AREA_PER_SYMBOL + jump_index * self.CHANNELS + chord

  def send_text(self, payload ="aaaa"):
    # YYY: encode payload as a zero-one sequence

    payload = self.encode_payload(payload)
    while len(payload) % self.SYMBOL_PER_PAYLOAD:
      payload.append(0)
    
    p = pyaudio.PyAudio()
    stream = p.open(rate=self.F_SAMPLE_RATE, channels=1, format=pyaudio.paFloat32, output=True)
    # wf is used of testing, can be removed later.
    wf = wave.open('output.wav', 'wb')
    wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
    wf.setframerate(self.F_SAMPLE_RATE)
    wf.setnchannels(1)
    chunks = [ payload[i:i+self.SYMBOL_PER_PAYLOAD] for i in range(0, len(payload), self.SYMBOL_PER_PAYLOAD)]
    for chunk in chunks:
      self.send_text_n_bit(chunk, stream, wf)
      print("chunk:", self.decode_payload(chunk), chunk)
    stream.close()
    p.terminate()
    wf.close()
  
  def copy_amble_code(self, code, time):
    a = []
    for c in code:
      for t in range(time):
        a.append(ord(c) - ord('0'))
    return a

  def send_text_n_bit(self, payload, stream, wf = None, area_id = 0):
    seq = self.copy_amble_code(self.PREAMBLE_CODE, self.SYMBOL_PER_DURATION) 
    
    # [ int(c) for c in self.PREAMBLE_CODE ] * self.SYMBOL_PER_DURATION
    seq.extend(payload)
    seq.extend( self.copy_amble_code(self.POSTAMBLE_CODE, self.SYMBOL_PER_DURATION)  ) # TODO!
    inner_index = 0
    for symbol_group in range(0, len(seq) // self.SYMBOL_PER_DURATION ):
      bits = []
      jump_index = symbol_group % self.JUMP_FREQ_AREA_PER_SYMBOL
      for symbol_index in range(self.SYMBOL_PER_DURATION):
        s = seq[ symbol_group * self.SYMBOL_PER_DURATION + symbol_index ]
        bits.append( self.compute_target_frequency_index( s , jump_index, symbol_index)  )
      print(bits)
      self.generate_bit(bits, stream, self.SINGLE_DURATION, self.F_SAMPLE_RATE, wf = wf)
        
        # target_bit = seq[symbol_i]
    # for s in seq:
      # TODO: multiple char in one duration
      # target_bit = s + inner_index * self.CHANNELS
      # self.generate_bit([target_bit], stream, self.SINGLE_DURATION, self.F_SAMPLE_RATE, wf = wf)
      # inner_index = (inner_index + 1) % self.JUMP_FREQ_AREA_PER_SYMBOL
  
  def send_text_two_bit(self, payload, stream, wf = None):
    print("send text with two bit is only used for testing purpose. deprecated.")
    exit()
    seq = self.PREAMBLE_CODE + payload + self.POSTAMBLE_CODE
    for s in seq:
      self.generate_bit([0 if s == "0" else 1], stream, self.SINGLE_DURATION, self.F_SAMPLE_RATE, wf = wf)
  
  def keep_decoding(self, frame_queue):
    signal = []
    count = 0
    while True:
      count = 0
      while True:
        count += 1
        f = frame_queue.get()
        signal = np.append(signal, np.frombuffer(f, dtype=np.float32))
        if len(signal) / self.F_SAMPLE_RATE > 5:
          break
      if len(signal) > self.MAX_SAMPLE_MSG:
        signal = self.decode_once(signal)
        print("decode.", len(signal), signal[0])
  
  def decode_once(self,signal):
    if len(signal) < self.MIN_SAMPLE_MSG: 
      return signal
  
    # Apply the band-pass filter
    signal = butter_bandpass_filter(signal, self.F_MAP[0] // 2, self.F_MAP[-1] + 1000, self.F_SAMPLE_RATE, 5)
  
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
    # find a proper start of the signal
    # YYY: There may be an offset compare to the start of the recording and the start of the signal.
    # YYY: Divide signal into certain small piceses to find the proper start of the signal.
    # YYY: We assume the signal is initiiatlly good. If we cannot detect the preamble at a first try, we use a heavier denoise algorithm to improve audio quality. 
    denoised = False
    div = 10
    found = False
    start_index = -1
    next_offset = -1
    for start_jump_index in range(self.JUMP_FREQ_AREA_PER_SYMBOL):
      if found:
        break
      for offset in [ int(self.SINGLE_DURATION / div * (i) * self.F_SAMPLE_RATE) for i in range(div) ]:
        small_step = 1
        small_step_len = self.SINGLE_DURATION / small_step
        step = int(small_step_len * self.F_SAMPLE_RATE)
  
        sliced = []
        power_threshold = self.POWER_THRESHOLD * step
        jump_index = start_jump_index
        for i in range(offset, len(signal) - step,  step ):
          # draw_fft_result(signal[i: i + step] )
          fft_result = np.fft.fft(signal[i : i + step])
          fft_freqs = np.fft.fftfreq(len(fft_result), 1/self.F_SAMPLE_RATE)
          for j in range(self.SYMBOL_PER_DURATION):
            possible_bit = self.select_frequency(fft_result, fft_freqs, power_threshold, jump_index = jump_index, symbol_index = j )
            sliced.append(possible_bit)
          jump_index = (jump_index + 1) % self.JUMP_FREQ_AREA_PER_SYMBOL
  
        preamble = self.PREAMBLE_TEMPLATE
        payload_index = detect_preamble(sliced, preamble) 
        if payload_index is None or payload_index % self.SYMBOL_PER_DURATION != 0:
          # A failure lead to a heavier denoise algorithm.
          # TODO: premature quit if it is a sequence of obvious noise.
          # if not denoised:
            # denoised = True
            # signal = librosa.effects.trim(signal, top_db=80)[0]
          continue
        else:
          payload_index = payload_index // self.SYMBOL_PER_DURATION
          step = int(self.SINGLE_DURATION * self.F_SAMPLE_RATE)
          found = True
          next_offset = offset + int(payload_index * small_step_len * self.F_SAMPLE_RATE)
          start_index = start_jump_index + payload_index
          # print("offset:", offset)
          break
  
    if not found:
      # print(sliced, "not found.")
      return signal[-self.MAX_SAMPLE_MSG:]
    # print("good!")
    # exit()
    # TODO: variable-sized message
    # print(sliced, "found.")
    # print("good:", sliced[:self.MSG_SIZE])

    offset = next_offset
    if offset + self.MAX_SAMPLE_MSG > len(signal):
      return signal[offset:]
  
    sliced = []
    power_threshold = self.POWER_THRESHOLD * step 
    jump_index = 0
    for i in range(offset, len(signal) - step,  step ):
      fft_result = np.fft.fft(signal[i : i + step])
      fft_freqs = np.fft.fftfreq(len(fft_result), 1/self.F_SAMPLE_RATE)
      for j in range(self.SYMBOL_PER_DURATION):
        possible_bit = self.select_frequency(fft_result, fft_freqs, power_threshold, jump_index = jump_index, symbol_index = j )
        sliced.append(possible_bit)
      # possible_bit = self.select_frequency(fft_result, fft_freqs, power_threshold, (current_index + start_index) % 2)
      # sliced.append(possible_bit)
      jump_index = (jump_index + 1) % self.JUMP_FREQ_AREA_PER_SYMBOL
  
    # TODO: check preamble the second time.
    # print("* sliced from preamble:", sliced)
    # TODO: variable-sized message
    msg = sliced[:self.MSG_SIZE * self.SYMBOL_PER_DURATION]
    # print("msg:", msg)
    result = msg[ len(self.PREAMBLE_CODE) * self.SYMBOL_PER_DURATION: (len(self.PREAMBLE_CODE) + self.PAYLOAD_SIZE) * self.SYMBOL_PER_DURATION ]
    # print("* payload:", msg[ len(self.PREAMBLE_CODE): len(self.PREAMBLE_CODE) + self.PAYLOAD_SIZE ] )
    print("* payload:", result)
    print("** decode:", self.decode_payload(result))
    # TODO: check postamble.
    # TODO: variable-sized message
    return signal[offset + self.MAX_SAMPLE_MSG: ]

  def draw_audio(self, path):
    raw = wave.open(path) 
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="float32")
    # Load the WAV file
    while len(signal) >= self.MAX_SAMPLE_MSG:
      signal = self.decode_once(signal) 


  def keep_recording(self, record_length = 100, debug = True):
    frame_queue = multiprocessing.Queue(maxsize=4096)
    process = multiprocessing.Process(target=self.keep_decoding, args=(frame_queue, ))
    process.start()

    print("keep recording...")
    # self.debug = True # !!! change later.
    self.debug = False
    fn = datetime.datetime.now().strftime("record.%m%d-%H-%M-%S.%f.wav")
    with wave.open(fn, 'wb') as wf:
      p = pyaudio.PyAudio()
      if self.debug:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
        wf.setframerate(self.F_SAMPLE_RATE)
      stream = p.open(format=pyaudio.paFloat32, channels=1, rate=self.F_SAMPLE_RATE, input=True, frames_per_buffer=self.FRAMES_PER_BUFFER)
      print('Recording...')
      for _ in range(0, self.F_SAMPLE_RATE // Player.CHUNK * record_length):
          v = stream.read(Player.CHUNK)
          if self.debug:
            wf.writeframes(np.frombuffer(v, dtype=np.float32).tobytes())
          else:
            frame_queue.put(v)
      print('Done')
      stream.close()
      p.terminate()

if __name__ == "__main__":
  player = Player()
  if args.function == "draw":
    player.draw_audio(args.file)
  elif args.function == "record":
    player.keep_recording()
  elif args.function == "send":
    player.send_text(args.file)
  else:
    print("i don't understand.")