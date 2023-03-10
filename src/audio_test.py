import pyaudio
import numpy as np
from save_audio import save_audio

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
i=0
input_data=np.zeros((chunk_size))
while i<500:

    raw_data = stream.read(chunk_size)
    data = np.frombuffer(raw_data, dtype=np.float32)
    input_data=np.concatenate((input_data, data))
    i+=1

# close the system
stream.stop_stream()
stream.close()
audio.terminate()
save_audio('test_audio.wav',input_data, rate,16)