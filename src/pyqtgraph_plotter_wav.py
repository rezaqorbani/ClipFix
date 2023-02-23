"""
This code first defines a list of audio files to plot, then creates a PyQtGraph application and window.
It then creates a plot item and adds it to the window, sets the plot labels, and creates a legend.
It then loops over each audio file in the list, loads the file, converts it to mono if necessary,
creates a time array, and plots the audio data. It also adds each plot to the legend with a unique
color and name.
"""

import pyqtgraph as pg
import numpy as np
from scipy.io import wavfile

# Load audio files
audio_files = [
    'pathtofile1.wav',
    'pathtofile2.wav',
]

# Create PyQtGraph application and window
app = pg.QtGui.QApplication([])
win = pg.GraphicsWindow(title='Audio Signal Visualization')
win.resize(800, 600)

# Create plot item and add to window
plot_item = win.addPlot(title='Audio Signals in Time Domain')
plot_item.setLabel('left', 'Amplitude', units='V')
plot_item.setLabel('bottom', 'Time', units='s')

# Create legend
legend = plot_item.addLegend()

# Loop over audio files and plot each one
for idx, audio_file in enumerate(audio_files):
    # Load audio file
    fs, audio_data = wavfile.read(audio_file)

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Create time array
    time = np.arange(len(audio_data)) / float(fs)

    # Plot audio data and add to legend
    plot = plot_item.plot(time, audio_data, pen=(idx, len(audio_files)))
    legend.addItem(plot, f'Audio Signal {idx+1}')

# Start PyQtGraph application
app.exec_()
