from utils import *
from tkinter import ttk

import multiprocessing
import argparse
import pyaudio
import wave
import numpy as np
import pygame
import pywt
import multiprocessing
import tkinter as tk

parser = argparse.ArgumentParser(description="Audio helper.")
functions = ['draw', 'record', 'send', 'gui' ]
parser.add_argument('function', choices=functions, help="")
parser.add_argument('--file', help="audio(wav) filepath (read/write)")
parser.add_argument('--frequency', help="frequency for genernating(Q1)")
parser.add_argument('--sampling_rate', help="samping rate for generating(Q1) and recording(Q3)")
parser.add_argument('--duration', help="duration for genernating(Q1) and recording(Q3)")
args = parser.parse_args()

TEST_STR1 = "QQcTghKTkklwnWWmXtsOHnwHwZHzRZwUsKFiXvsrsucaUCekywtbzTkOyVYjJEbrRlBudXpCgzGgRIMZLbebHASoIFLjZHaxUQNg"

TEST_STR2 = "vRCfkzdELlciuBAUIrBAzEIbPtuhsehbsiCwBQjqcVpWvxxyZZzFAcIVKfXGYBTJDbKXpsPBfzfllyOCQtPTDwIUtlPpeQiZlDCJ"
    
TO_TEST = TEST_STR2


class Player:

  CHUNK = 4096
  def __init__(self):
    self.F_SAMPLE_RATE = 44100
    self.FRAMES_PER_BUFFER = 4096
    self.SINGLE_DURATION = 0.065

    self.switch = 0
    self.count = 0

    self.symall = None

    # message layout
    # YYY: preamble code + payload + postamble
    self.PAYLOAD_SIZE = 50
    self.PREAMBLE_CODE= "7777111177772222"
    self.POSTAMBLE_CODE="1100"
    self.MSG_SIZE = len(self.PREAMBLE_CODE) + len(self.POSTAMBLE_CODE) + self.PAYLOAD_SIZE

    self.MIN_SAMPLE_MSG = int(self.MSG_SIZE * self.SINGLE_DURATION * self.F_SAMPLE_RATE)
    self.MAX_SAMPLE_MSG = int(self.MSG_SIZE * self.SINGLE_DURATION * self.F_SAMPLE_RATE)

    self.JUMP_FREQ_AREA_PER_SYMBOL = 2 # Increase this number to get more accurate result.
    self.SYMBOL_PER_DURATION = 1 # Increase this number to transfer more info at one duration
    self.CHANNELS = 54 # Increaase this number to transfer more info at one duration.
    # self.CHANNELS = 8
    self.FGAP = 40 # Increase this number to get more accurate result.

    preamble = []
    for s in self.PREAMBLE_CODE:
      for i in range(self.SYMBOL_PER_DURATION):
        preamble.append(float(ord(s) - ord("0")))
      # self.PREAMBLE_TEMPLATE[step] = preamble
    self.PREAMBLE_TEMPLATE = preamble

    self.SYMBOL_PER_PAYLOAD = self.PAYLOAD_SIZE * self.SYMBOL_PER_DURATION

    self.POWER_THRESHOLD = 1.0 / self.F_SAMPLE_RATE

    self.BASE_FREQ = 1000
    self.GAP_RATE = 2.2 # Increase this number to get more accurate result
    self.AREA_GAP = int( self.FGAP * self.CHANNELS * self.GAP_RATE * 1.05 )
    self.F_MAP = []
    for a in range(self.JUMP_FREQ_AREA_PER_SYMBOL * self.SYMBOL_PER_DURATION):
      self.F_MAP.extend([ self.BASE_FREQ + self.AREA_GAP * a + i * (self.FGAP * self.GAP_RATE) for i in range(self.CHANNELS) ])
    print(self.F_MAP)

    self.debug = False

  def generate_bit(self, index, stream, duration, sampling_rate, wf = None):
    audio = np.sum( [ generate_beep(duration, self.F_MAP[_index] , sampling_rate) for _index in index ], axis=0)
    audio = audio / len(index)
    sbytes = np.array(audio, dtype=np.float32)
    stream.write(sbytes.tobytes())
    # if wf:
      # save_signal(audio, 44100)
      # print(wf.get
      # sbytes = (g_audio * 32767).astype(dtype=np.int16)
      # wf.writeframes(sbytes.tobytes())
    return audio

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
  
  def encode_payload(self, payload):
    if self.CHANNELS < 53:
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
      return [ e % self.CHANNELS for e in ec ]
    else:
      ec = []
      for c in payload:
        if ord(c) <= ord('Z') and ord(c) >= ord('A'): # directly map.
          v = ord(c) - ord('A') + 1
          ec.extend( [ v ]  )
          # ec.append(ord(c) - ord('A') + 1)
        elif ord(c) <= ord('z') and ord(c) >= ord('a'):
          v = ord(c) - ord('a') + 27
          ec.extend( [ v ] )
        else:
          ec.append( 0 )
      return [ e % self.CHANNELS for e in ec ]

  def decode_payload(self, payload):
    if self.CHANNELS < 53:
      ec = ""
      for i in range(0, len((payload)), 2):
        c = payload[i] * 8 + payload[i + 1]
        if c <= 26 and c >= 1: # directly map.
          ec += chr(c + ord('A') - 1)
        elif c <= 26 + 26 and c >= 26 + 1:
          ec += chr(c + ord('a') - 27)
        else:
          ec += ' '
      return ec
    else:
      ec = ""
      for i in range(0, len((payload)), 1):
        c = payload[i] 
        if c <= 26 and c >= 1: # directly map.
          ec += chr(c + ord('A') - 1)
        elif c <= 26 + 26 and c >= 26 + 1:
          ec += chr(c + ord('a') - 27)
        else:
          ec += ' '
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
    chunks = [ payload[i:i+self.SYMBOL_PER_PAYLOAD] for i in range(0, len(payload), self.SYMBOL_PER_PAYLOAD)]
    for chunk in chunks:
      self.send_text_n_bit(chunk, stream, None)
      print("chunk:", self.decode_payload(chunk), chunk)
    stream.close()
    p.terminate()
    save_signal(self.symall, 44100, "output.wav")
  
  def copy_amble_code(self, code, time):
    a = []
    for c in code:
      for t in range(time):
        a.append((ord(c) - ord('0')))
    return a

  def send_text_n_bit(self, payload, stream, wf = None, area_id = 0):

    seq = []
    # seq.extend( self.copy_amble_code(self.POSTAMBLE_CODE, self.SYMBOL_PER_DURATION) )
    seq.extend( self.copy_amble_code(self.PREAMBLE_CODE, self.SYMBOL_PER_DURATION) )
    
    seq.extend(payload)
    seq.extend( self.copy_amble_code(self.POSTAMBLE_CODE, self.SYMBOL_PER_DURATION)  ) # TODO!
    s = self.generate_bit( [0], stream, self.SINGLE_DURATION * 3, self.F_SAMPLE_RATE, wf = wf)
    if self.symall is None:
        self.symall = s
    else:
        self.symall = np.append(self.symall, s)
    for symbol_group in range(0, len(seq) // self.SYMBOL_PER_DURATION ):
      bits = []
      jump_index = symbol_group % self.JUMP_FREQ_AREA_PER_SYMBOL
      for symbol_index in range(self.SYMBOL_PER_DURATION):
        s = seq[ symbol_group * self.SYMBOL_PER_DURATION + symbol_index ]
        bits.append( self.compute_target_frequency_index( s , jump_index, symbol_index)  )
      sym = self.generate_bit(bits, stream, self.SINGLE_DURATION, self.F_SAMPLE_RATE, wf = wf)
      if self.symall is None:
        self.symall = sym
      else:
        self.symall = np.append(self.symall, sym)
    
    print(self.symall.shape)
  
  def send_text_two_bit(self, payload, stream, wf = None):
    print("send text with two bit is only used for testing purpose. deprecated.")
    exit()
    seq = self.PREAMBLE_CODE + payload + self.POSTAMBLE_CODE
    for s in seq:
      self.generate_bit([0 if s == "0" else 1], stream, self.SINGLE_DURATION, self.F_SAMPLE_RATE, wf = wf)
  
  def wrap_signal(signal):
    l = len(signal)
    return signal[int(l * 0.02): int(l - (l * 0.02))]
  
  def keep_decoding(self, frame_queue, data_queue=None):
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
        signal = self.decode_once(signal, data_queue)
        print("decode.", len(signal), signal[0])
  
  def decode_once(self,signal, data_queue=None):
    if len(signal) < self.MIN_SAMPLE_MSG: 
      return signal
  
    # Apply the band-pass filter
    signal = butter_bandpass_filter(signal, self.F_MAP[0] // 2, self.F_MAP[-1] + 1000, self.F_SAMPLE_RATE, 5)
  
    offset = -1
    # find a proper start of the signal
    # YYY: There may be an offset compare to the start of the recording and the start of the signal.
    # YYY: Change offset and working freq area to find the proper start of the signal.
    div = 10
    found = False
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
          # if not denoised:
            # denoised = True
            # signal = librosa.effects.trim(signal, top_db=80)[0]
          continue
        else:
          payload_index = payload_index // self.SYMBOL_PER_DURATION
          step = int(self.SINGLE_DURATION * self.F_SAMPLE_RATE)
          found = True
          next_offset = offset + int(payload_index * small_step_len * self.F_SAMPLE_RATE)
          break
  
    if not found:
      return signal[-self.MAX_SAMPLE_MSG:]

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
      jump_index = (jump_index + 1) % self.JUMP_FREQ_AREA_PER_SYMBOL
  
    msg = sliced[:self.MSG_SIZE * self.SYMBOL_PER_DURATION]
    result = msg[ len(self.PREAMBLE_CODE) * self.SYMBOL_PER_DURATION: (len(self.PREAMBLE_CODE) + self.PAYLOAD_SIZE) * self.SYMBOL_PER_DURATION ]
    print("* payload:", result)
    decoded_r = self.decode_payload(result)
    print("** decode:", decoded_r)
    decoded_r = decoded_r.rstrip('.')
    
    if data_queue is not None:
      data_queue.put(self.decode_payload(result))
    return signal[offset + self.MAX_SAMPLE_MSG: ]

  def draw_audio(self, path):
    raw = wave.open(path) 
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="float32")
    # Load the WAV file
    while len(signal) >= self.MAX_SAMPLE_MSG:
      signal = self.decode_once(signal) 
  
  def keep_recording(self, record_length = 100,  data_queue=None):
    frame_queue = multiprocessing.Queue(maxsize=4096)
    process = multiprocessing.Process(target=self.keep_decoding, args=(frame_queue, data_queue, ))
    process.start()

    print("keep recording...")
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


window = tk.Tk()

send_player = None
receive_player = None
insert_var = tk.StringVar()
len_var = tk.StringVar()
receive_var = tk.StringVar()
compare_var = tk.StringVar()

def on_send_clicked():
  send_player = Player()
  send_player.send_text(insert_var.get())

def on_insert_change(event):
  if len_var:
    len_var.set("input string length:"+ str(len( insert_var.get() )))
  compare_io()
  pass

def on_clear_clicked():
  receive_var.set("")
  compare_io()

def compare_io():
  l = min(len(receive_var.get()), len(insert_var.get()))
  a = receive_var.get()[:l]
  b = insert_var.get()[:l]
  if a != b:
    label = "IO differ: "
    for i in range(l):
      if a[i] != b[i]:
         label = label + f"#at index {i}, receive-*{a[i]}*, input-*{b[i]}* #\n" 
  else:
    label = "IO are the same.\n"
  compare_var.set(label)

Gdata_queue = multiprocessing.Queue(maxsize=4096)

def start_receive():
  def update_data():
    print("get into the loop.")
    while not Gdata_queue.empty():
      d = Gdata_queue.get()
      receive_var.set( receive_var.get() + d )
      compare_io()
    window.after(200, update_data)
  window.after(200, update_data)

  receive_player = Player()
  process = multiprocessing.Process(target=receive_player.keep_recording, args=(100, Gdata_queue, ))
  process.start()




def gui_start():
  # Create the main window
  window.title("Audio Transmitter")
  # text1_var = "INPUT != OUTPUT"
  # text2_var = tk.StringVar()

  insert_var.set(TO_TEST)

  # insert_var.set("pkmnhuygbeAusVEFhjiuasdfBERabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZlkajsdhflkjqhwelrkjbl")
  # insert_var.set("QQcTghKTkklwnWWmXtsOHnwHwZHzRZwUsKFiXvsrsucaUCekywtbzTkOyVYjJEbrRlBudXpCgzGgRIMZLbebHASoIFLjZHaxUQNg")
  insert_entry = tk.Entry(window, textvariable=insert_var, width=100)
  insert_entry.bind("<Key>", on_insert_change)  # Bind the event handler to the insert entry

  len_var.set("input string length:"+ str(len( insert_var.get() )))
  len_text = tk.Label(window, textvariable=len_var)

  # Create the first button and text field
  button1 = tk.Button(window, text="send", command=on_send_clicked)
  button2 = tk.Button(window, text="clear received", command=on_clear_clicked)
  button3 = tk.Button(window, text="start receive", command=start_receive)

  receive_text = tk.Entry(window, textvariable=receive_var, width=100)
  
  # Create the second button and text field
  # button2 = tk.Button(window, text="Button 2", command=on_check_clicked)
  # text2 = tk.Entry(window, textvariable=text2_va

  separator = ttk.Separator(window, orient='horizontal')
  
  # Place the widgets in the window
  tk.Label(window, text="Sender:").pack(pady=5)
  insert_entry.pack(pady=10)
  len_text.pack(pady=5)
  button1.pack(pady=10)
  separator.pack(fill='x', pady=5)
  tk.Label(window, text="Receiver").pack(pady=5)
  button3.pack(pady=10)
  receive_text.pack(pady=10)
  tk.Label(window, textvariable=compare_var, wraplength=400).pack(pady=5)
  button2.pack(pady=5)
  
  receive_var.set("")
  compare_io()

  # Start the GUI event loop
  window.mainloop()


if __name__ == "__main__":
  player = Player()
  if args.function == "draw":
    player.draw_audio(args.file)
  elif args.function == "record":
    player.keep_recording()
  elif args.function == "send":
    player.send_text(args.file)
  elif args.function == "gui":
    gui_start()
  else:
    print("i don't understand.")