"""
This code creates a PyQtGraph window with a single plot that shows the live audio signal from the
microphone. The update() function is called every time the timer fires, which reads a chunk of audio
input_data from the microphone and updates the plot with the new input_data. The run() function starts the
PyQtGraph event loop, which allows the window to stay open and update in real time. Finally, the
close() function stops the audio stream and terminates the PyAudio instance when the window is closed.
"""

import pyaudio
import numpy as np
from PyQt6.QtWidgets import QMessageBox
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import sys
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
        self.rate = 44100
        self.stream: pyaudio.Stream = None
        self.input_device_index = 1

        # Set up x and y arrays
        self.x = np.arange(0, 2 * self.chunk_size, 2)
        self.input_data = np.zeros((1, self.nchannels))
        self.output_data = np.zeros(1)

        # Set up plot length
        self.plot_length = 1  # set the plot length to 1 second

        # Set up timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)

    def update(self):
        # Get audio input_data
        if self.recording:
            raw_data = self.stream.read(self.chunk_size)
            numpy_data = np.frombuffer(raw_data, dtype=np.float32)
            inputs = np.zeros((self.chunk_size, self.nchannels))

            # seperate the channels
            for i in range(self.nchannels):
                inputs[:, i] = numpy_data[i::self.nchannels]
            
            self.input_data = np.concatenate((self.input_data, inputs))
            
            # Update plot
            time_array = np.arange(len(self.input_data)) / float(self.rate)

            for i in range(self.nchannels):
                self.curves["input"][i].setData(time_array[-self.plot_length * self.rate:, ],
                                       self.input_data[-self.plot_length * self.rate:, i])

            if self.nchannels != 1:
                self.output_data = mix(self.input_data)
                #self.output_data = self.input_data[:, 0]
                self.curves["output"][0].setData(time_array[-self.plot_length * self.rate:, ], 
                                                self.output_data[-self.plot_length * self.rate:])
            else:
                self.curves["output"][0].setData(time_array[-self.plot_length * self.rate:, ], 
                                                self.input_data[-self.plot_length * self.rate:, 0])

    def startRecording(self):
        self.recording = True
        self.stream = self.audio.open(format=self.format, channels=self.nchannels,
                                      rate=self.rate, input=True,
                                      frames_per_buffer=self.chunk_size,
                                      input_device_index=self.input_device_index)

    def stopRecording(self):
        self.recording = False
        self.stream.stop_stream()

    def closeSession(self):
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        
        ## Save audios
        n_bits = 16
        # Save input audio
        for i in range(self.nchannels):
            save_audio(f"channel_{i+1}.wav", self.input_data[:,i], self.rate, n_bits)
        # Save output audio
        save_audio('output_audio.wav', self.output_data, self.rate, n_bits)
        

    def isReocrding(self):
        return self.recording


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, nchannels, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.nchannels = nchannels

        # Set up plot
        self.win = pg.GraphicsLayoutWidget(title='Live Audio Plot')
        self.setCentralWidget(self.win)
        self.win.resize(800, 600)
        self.win.setWindowTitle('Live Audio Plot')
        self.inputPlots = []
        self.curves = {"input": [], "output": []}
        
        for i in range(self.nchannels):
            self.inputPlots.append(self.win.addPlot(row=i, col=0, colspan=2, title='Channel {}'.format(i+1)))
            self.inputPlots[i].setYRange(-1, 1)
            self.inputPlots[i].setLabels(left='Amplitude', bottom='Time (s)')
            self.inputPlots[i].getAxis('bottom').setStyle(tickFont=QFont('Arial', 10), autoExpandTextSpace=True)
            self.inputPlots[i].getAxis('left').setStyle(tickFont=QFont('Arial', 10), autoExpandTextSpace=True)
            self.inputPlots[i].showGrid(x=True, y=True, alpha=0.5)
            self.curves["input"].append(self.inputPlots[i].plot(pen=pg.mkPen('y', width=2)))

        self.outputPlot = self.win.addPlot(row=self.nchannels, col=0, colspan=2, title='Output')
        self.outputPlot.setYRange(-1, 1)
        self.outputPlot.setLabels(left='Amplitude', bottom='Time (s)')
        self.outputPlot.getAxis('bottom').setStyle(tickFont=QFont('Arial', 10), autoExpandTextSpace=True)
        self.outputPlot.getAxis('left').setStyle(tickFont=QFont('Arial', 10), autoExpandTextSpace=True)
        self.outputPlot.showGrid(x=True, y=True, alpha=0.5)
        self.curves["output"].append(self.outputPlot.plot(pen=pg.mkPen('y', width=2)))

        # Set up live audio
        self.liveAudio = LiveAudio(nchannels=self.nchannels, curves=self.curves)

        self.recordButtonProxy = QtWidgets.QGraphicsProxyWidget()
        self.recordButton = QtWidgets.QPushButton('Record')
        self.recordButton.clicked.connect(self.record)
        self.recordButtonProxy.setWidget(self.recordButton)
        self.win.addItem(self.recordButtonProxy, row=self.nchannels+1, col=0)

        self.stopButtonProxy = QtWidgets.QGraphicsProxyWidget()
        self.stopButton = QtWidgets.QPushButton('Stop')
        self.stopButton.clicked.connect(self.stop)
        self.stopButtonProxy.setWidget(self.stopButton)
        self.win.addItem(self.stopButtonProxy, row=self.nchannels+1, col=1)

    def record(self):
        self.liveAudio.startRecording()

    def stop(self):
        self.liveAudio.stopRecording()

    def closeEvent(self, event):
        if self.liveAudio.isReocrding():
            # Ask the user if they really want to exit
            reply = QMessageBox.question(self, 'Confirm Exit', 'Are you sure you want to exit?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.liveAudio.closeSession()
                event.accept()
            else:
                event.ignore()
        else:
            # Close session
            self.liveAudio.closeSession()
            # Perform cleanup here
            event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    win = MainWindow(nchannels=2)
    win.show()
    sys.exit(app.exec())
