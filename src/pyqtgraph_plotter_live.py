"""
This code creates a PyQtGraph window with a single plot that shows the live audio signal from the
microphone. The update() function is called every time the timer fires, which reads a chunk of audio
data from the microphone and updates the plot with the new data. The run() function starts the
PyQtGraph event loop, which allows the window to stay open and update in real time. Finally, the
close() function stops the audio stream and terminates the PyAudio instance when the window is closed.
"""

import pyaudio
import numpy as np
from PyQt6 import QtCore, QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys


class LiveAudioPlot(QtWidgets.QMainWindow):
    def __init__(self, channels, *args, **kwargs):
        super(LiveAudioPlot, self).__init__(*args, **kwargs)
        # Set up audio stream
        self.audio = pyaudio.PyAudio()
        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.channels = channels
        self.rate = 44100
        self.stream = self.audio.open(format=self.format, channels=self.channels,
                                      rate=self.rate, input=True,
                                      frames_per_buffer=self.chunk_size)

        # Set up plot
        self.win = pg.GraphicsLayoutWidget(title='Live Audio Plot')
        self.win.resize(800, 600)
        self.win.setWindowTitle('Live Audio Plot')
        self.plots = []
        self.curves = []	
        for i in range(self.channels):
            
            self.plots.append(self.win.addPlot(title='Channel {}'.format(i)))

            self.plots[i].setYRange(-1, 1)

            self.curves.append(self.plots[i].plot(pen='y'))

        self.x = np.arange(0, 2 * self.chunk_size, 2)
        self.data = np.zeros(self.chunk_size)

        # Set up plot length
        self.plot_length = 10  # set the plot length to 5 seconds

        # Set up timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)

    def update(self):
        # Get audio data
        raw_data = self.stream.read(self.chunk_size)
        data = np.frombuffer(raw_data, dtype=np.float32)
        self.data = np.concatenate((self.data, data))

        # Update plot
        time_array = np.arange(len(self.data)) / float(self.rate)
        for i in range(self.channels):
            self.curves[i].setData(time_array[-self.plot_length * self.rate:], self.data[-self.plot_length * self.rate:])
       # self.curve.setData(time_array[-self.plot_length * self.rate:], self.data[-self.plot_length * self.rate:])


    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


if __name__ == '__main__':

    app = QtWidgets.QApplication([])
    lap = LiveAudioPlot(channels=3)
    lap.show()
    sys.exit(app.exec())
