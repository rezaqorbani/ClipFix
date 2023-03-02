"""
This code creates a PyQtGraph window with a single plot that shows the live audio signal from the
microphone. The update() function is called every time the timer fires, which reads a chunk of audio
data from the microphone and updates the plot with the new data. The run() function starts the
PyQtGraph event loop, which allows the window to stay open and update in real time. Finally, the
close() function stops the audio stream and terminates the PyAudio instance when the window is closed.
"""

import pyaudio
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import wave


class LiveAudioPlot(object):
    def __init__(self):
        # Set up audio stream
        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.channels = 2
        self.rate = 44100
        self.p = pyaudio.PyAudio()
        
        # define gain levels in dB
        self.whisper_gain = -40  # dB
        self.shout_gain = 20  # dB       

        # define gain adjustments for each channel
        self.gain_adjustments = [self.whisper_gain, self.shout_gain]

        # create buffers for each channel
        self.buffers = [[] for _ in range(self.channels+1)]
        
       # define callback function for recording
        def callback(in_data, frame_count, time_info, status):
            if status:
                print(status, flush=True)

            # convert audio signal to NumPy array
            signal = np.frombuffer(in_data, dtype=np.float32).reshape(frame_count, self.channels)

            # store audio signal in buffers for each channel
            for channel in range(self.channels):
                # adjust gain level based on desired gain adjustment
                self.buffers[-1].append(signal[:, 0])
                adjusted_signal = signal[:, channel] * 10**(self.gain_adjustments[channel]/20)
                # store adjusted signal in buffer
                self.buffers[channel].append(adjusted_signal)
                
            self.update(self.buffers[0], self.buffers[-1], self.buffers[1])
            
            return (in_data, pyaudio.paContinue)
        
        self.stream = self.p.open(format=self.format, channels=self.channels,
                                      rate=self.rate, input=True,
                                      frames_per_buffer=self.chunk_size,
                                      stream_callback=callback)
        
        # Set up plot
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title="Live Audio Plot")
        self.win.resize(800, 600)
        self.win.setWindowTitle('Live Audio Plot')
        self.plot1 = self.win.addPlot(title='Audio Signal 1')
        self.plot1.setYRange(-1, 1)
        self.curve1 = self.plot1.plot(pen='y')
        self.ploto = self.win.addPlot(title='Audio Signal')
        self.ploto.setYRange(-1, 1)
        self.curveo = self.ploto.plot(pen='g')
        self.plot2 = self.win.addPlot(title='Audio Signal 2')
        self.plot2.setYRange(-1, 1)
        self.curve2 = self.plot2.plot(pen='b')
        self.x = np.arange(0, 2 * self.chunk_size, 2)
        self.data1 = np.zeros(self.chunk_size)
        self.data2 = np.zeros(self.chunk_size)
        self.data = np.zeros(self.chunk_size)

        # Set up plot length
        self.plot_length = 10  # set the plot length to 5 seconds

        # Set up timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)
        # start recording audio
        self.stream.start_stream()
        while True:
            if input("Press spacebar to stop recording: ") == ' ':
                break
            
    def update(self, raw_data1, raw_data, raw_data2):
        
        print(self.data)
        # Get audio data
        #data = np.frombuffer(raw_data, dtype=np.float32)
        self.data = np.concatenate((self.data, raw_data))
        #signal 1
        #data1=np.frombuffer(raw_data1, dtype=np.float32)
        self.data1 = np.concatenate((self.data1, raw_data1))
        #signal 2
        #data2 = np.frombuffer(raw_data2, dtype=np.float32)
        self.data2 = np.concatenate((self.data2, raw_data2))

        # Update plot
        time_array = np.arange(len(self.data1)) / float(self.rate)
        self.curve1.setData(time_array[-self.plot_length * self.rate:], self.data1[-self.plot_length * self.rate:])
        self.curve2.setData(time_array[-self.plot_length * self.rate:], self.data2[-self.plot_length * self.rate:])
        self.curveo.setData(time_array[-self.plot_length * self.rate:], self.data[-self.plot_length * self.rate:])

    def run(self):
        self.app.exec_()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


if __name__ == '__main__':
    lap = LiveAudioPlot()
    lap.run()
    lap.close()