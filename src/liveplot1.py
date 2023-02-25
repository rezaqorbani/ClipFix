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

        self.p = pyaudio.PyAudio()
        



        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 44100
        self.stream1 = self.p.open(format=self.format, channels=self.channels,
                                      rate=self.rate, input=True,
                                      frames_per_buffer=self.chunk_size)
        self.stream2 = self.p.open(format=self.format, channels=self.channels,
                                      rate=self.rate, input=True,
                                      frames_per_buffer=self.chunk_size)
        # Set up plot
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title="Live Audio Plot")
        self.win.resize(800, 600)
        self.win.setWindowTitle('Live Audio Plot')
        self.plot1 = self.win.addPlot(title='Audio Signal 1 (low gain)')
        self.plot1.setYRange(-1, 1)
        self.curve1 = self.plot1.plot(pen='y')
        self.ploto = self.win.addPlot(title='Original Signal')
        self.ploto.setYRange(-1, 1)
        self.curveo = self.ploto.plot(pen='g')
        self.plot2 = self.win.addPlot(title='Audio Signal 2 (high gain)')
        self.plot2.setYRange(-1, 1)
        self.curve2 = self.plot2.plot(pen='b')
        self.x = np.arange(0, 2 * self.chunk_size, 2)
        self.data1 = np.zeros(self.chunk_size)
        self.data2 = np.zeros(self.chunk_size)
        self.data = np.zeros(self.chunk_size)

        # Set up plot length
        self.plot_length = 2  # set the plot length to 5 seconds

        # Set up timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)

    def update(self):
        # Get audio data
        raw_data1 = self.stream1.read(self.chunk_size)
        data1 = np.frombuffer(raw_data1, dtype=np.float32)
        self.data = np.concatenate((self.data, data1))
        #signal 1
        data1=data1.copy()*0.5
        data1=np.clip(data1, -0.7, 0.7)
        self.data1 = np.concatenate((self.data1, data1))
        #signal 2
        raw_data2 = self.stream2.read(self.chunk_size)
        data2 = np.frombuffer(raw_data2, dtype=np.float32)
        data2=data2.copy()*5.5
        data2=np.clip(data2, -0.7, 0.7)
        self.data2 = np.concatenate((self.data2, data2))
        
        
        
        
        

        # Update plot
        time_array = np.arange(len(self.data1)) / float(self.rate)
        self.curve1.setData(time_array[-self.plot_length * self.rate:], self.data1[-self.plot_length * self.rate:])
        self.curve2.setData(time_array[-self.plot_length * self.rate:], self.data2[-self.plot_length * self.rate:])
        self.curveo.setData(time_array[-self.plot_length * self.rate:], self.data[-self.plot_length * self.rate:])

    def run(self):
        self.app.exec_()

    def close(self):
        self.stream1.stop_stream()
        self.stream2.stop_stream()
        self.stream1.close()
        self.stream2.close()
        self.p.terminate()


if __name__ == '__main__':
    lap = LiveAudioPlot()
    lap.run()
    lap.close()