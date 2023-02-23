
"""
This code records live audio signal from the microphone and applies different gain levesl to amplify the audio data. 
It then clips the audio data to ensure that the output is within the range of 16-bit integers. Finally, the clipped 
and amplified audios data is written to a new WAV files with the same parameters as the input file.
"""

import pyaudio
import numpy as np
import wave

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Set the parameters for the audio recordings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Record the input audio 

# Set the input device
input_device_index = audio.get_default_input_device_info()['index']
    
# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=input_device_index)


frames = []

# Record audio for N seconds
N=10
for j in range(0, int(RATE / CHUNK * N)):
        data = stream.read(CHUNK)
        frames.append(data)

# Stop recording
stream.stop_stream()
stream.close()

# Save the recording to a WAV file
filename = f"recording_input.wav"
wf = wave.open(filename, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


## Clipping part

# Load the pre-recorded WAV file
with wave.open("recording_input.wav", "rb") as input_wav:
    input_data = input_wav.readframes(input_wav.getnframes())
    input_data = np.frombuffer(input_data, dtype=np.int16).copy()

# Set the gain levels for the recordings
gains = [0.2, 10]

# Apply gain to the audio data for each gain level and save the resulting WAV files
for i, gain in enumerate(gains):
    
    # Apply the gain level to the input audio
    audio_data = (input_data * gain).astype(np.int16)
    
    # Determine the maximum amplitude allowed by the audio format
    max_amplitude = np.iinfo(np.int16).max
    print(max_amplitude)
    
    # Clip the audio data to prevent distortion
    audio_data = np.clip(audio_data, -max_amplitude, max_amplitude - 1)
    
    # Save the clipped audio data to a new WAV file
    filename = f"clipped_{i}.wav"
    with wave.open(filename, "wb") as output_wav:
        # Set the same parameters as the input WAV file
        output_wav.setnchannels(input_wav.getnchannels())
        output_wav.setsampwidth(input_wav.getsampwidth())
        output_wav.setframerate(input_wav.getframerate())

        # Write the clipped audio data to the new WAV file
        output_wav.writeframes(audio_data.tobytes())
# Close the input and output WAV files
input_wav.close()
output_wav.close()
# Close PyAudio
audio.terminate()