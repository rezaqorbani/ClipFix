import pyaudio
import numpy as np
from overlap_add import mix
from save_audio import save_audio

class LiveAudio():
    def __init__(self, nchannels, curves):

        self.curves = curves
        self.recording = False

        # Set up audio stream
        self.audio = pyaudio.PyAudio()
        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.nchannels = nchannels
        self.rate = 16000
        self.stream: pyaudio.Stream = None
        self.input_device_index = 1
        self.record_length = 15
        self.n_bits = 16

        # Set up x and y arrays
        self.x = np.arange(0, 2 * self.chunk_size, 2)
        self.input_data = np.zeros((self.rate * self.record_length, self.nchannels))
        self.output_data = np.zeros((self.rate * self.record_length, self.nchannels))
        self.temp_inputs = np.zeros((self.chunk_size, self.nchannels))

        # Set up plot length
        self.plot_length = 1  # set the plot length to 1 second
        self.stream = self.audio.open(format=self.format,
                                        channels=self.nchannels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.chunk_size,
                                        input_device_index=self.input_device_index)
    @profile
    def update(self):
            # Get audio input_data

            
            raw_data = self.stream.read(self.chunk_size)
            numpy_data = np.frombuffer(raw_data, dtype=np.float32)
            

            # seperate the channels
            for i in range(self.nchannels):
                self.temp_inputs[:, i] = numpy_data[i::self.nchannels]

            self.input_data = np.append(self.input_data, self.temp_inputs, axis=0)
            
            # Update plot
            time_array = np.arange(len(self.input_data)) / float(self.rate)

            for i in range(self.nchannels):
                self.curves["input"][i].setData(time_array[-self.plot_length * self.rate:, ],
                                    self.input_data[-self.plot_length * self.rate:, i])

            if self.nchannels != 1:
                self.output_data = mix(self.input_data         
    def save_audio(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        for i in range(self.nchannels):
            save_audio(f"channel_{i+1}.wav", self.input_data[:,i], self.rate, self.n_bits)
            # Save output audio
            save_audio('output_audio.wav', self.output_data, self.rate, self.n_bits)

                

if __name__ == "__main__":
    live_audio = LiveAudio(2, None)
    for i in range(100000):
        live_audio.update()
    live_audio.save_audio()