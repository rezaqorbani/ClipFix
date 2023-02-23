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


class LiveAudioPlot(object):
    def __init__(self):
        # Set up audio stream
        self.audio = pyaudio.PyAudio()
        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 44100
        self.stream = self.audio.open(format=self.format, channels=self.channels,
                                      rate=self.rate, input=True,
                                      frames_per_buffer=self.chunk_size)

        # Set up plot
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title="Live Audio Plot")
        self.win.resize(800, 600)
        self.win.setWindowTitle('Live Audio Plot')
        self.plot = self.win.addPlot(title='Audio Signal')
        self.plot.setYRange(-1, 1)
        self.curve = self.plot.plot(pen='y')
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
        self.curve.setData(time_array[-self.plot_length * self.rate:], self.data[-self.plot_length * self.rate:])

    def run(self):
        self.app.exec_()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


if __name__ == '__main__':
    lap = LiveAudioPlot()
    lap.run()
    lap.close()
