import pyaudio
import numpy as np

audio = pyaudio.PyAudio()
chunk_size = 1024
format = pyaudio.paFloat32
nchannels = 1
rate = 44100
stream: pyaudio.Stream = None
input_device_index = 1

p = pyaudio.PyAudio()

stream = audio.open(format=format, channels=nchannels,
                                rate=rate, input=True,
                                frames_per_buffer=chunk_size,
                                input_device_index=input_device_index)

while True:

    raw_data = stream.read(chunk_size)
    input_data = np.frombuffer(raw_data, dtype=np.float32)
    print(" shape of input_data: ", input_data.shape)

# close the system
stream.stop_stream()
stream.close()
audio.terminate()