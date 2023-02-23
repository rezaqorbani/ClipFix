import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import seaborn as sns

sns.set()

file = 'whatever.wav'
spf = wave.open(file, "r")

# Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, "Int16")
fs = spf.getframerate()

# If Stereo
if spf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

Time = np.linspace(0, len(signal) / fs, num=len(signal))

plt.figure(1)
plt.title("Signal Wave")
plt.plot(Time, signal)
plt.show()


# Version that will also handle stereo inputs,

# with wave.open(file,'r') as wav_file:
#     #Extract Raw Audio from Wav File
#     signal = wav_file.readframes(-1)
#     signal = np.fromstring(signal, 'Int16')
#
#     #Split the data into channels
#     channels = [[] for channel in range(wav_file.getnchannels())]
#     for index, datum in enumerate(signal):
#         channels[index%len(channels)].append(datum)
#
#     #Get time from indices
#     fs = wav_file.getframerate()
#     Time=np.linspace(0, len(signal)/len(channels)/fs, num=len(signal)/len(channels))
#
#     #Plot
#     plt.figure(1)
#     plt.title('Signal Wave...')
#     for channel in channels:
#         plt.plot(Time,channel)
#     plt.show()
