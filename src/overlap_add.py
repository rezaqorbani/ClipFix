# Import Packages
import wave
import numpy as np
from scipy import signal as sg


def mix(signals):
    """_summary_

    Args:
        signals (List of String elements): list of the audio files names to be mixed
        
    Output:
        The new synthetized signal by mixing the input signals using overlap_add method
    """
    audio_arrays = []
    audio_params = []
    
    # Read the two input audio files
    for signal in signals:
        with wave.open(signal,'rb') as audio_file:
            # Get the audio file parameters
            params = audio_file.getparams()
            # Read all audio frames into a byte string
            audio_frames = audio_file.readframes(params.nframes)
            # Convert the byte string to a NumPy array
            audio_array = np.frombuffer(audio_frames, dtype=np.int16)/32768.0 ##
            audio_params.append(params)
            audio_arrays.append(audio_array)

    # Ensure that both audio signals have the same number of channels


    # Ensure that both audio signals have the same sample rate


    # Define the window size and overlap
    window_size = 1024
    overlap = 650 #window_size // 2

    # Initialize the overlap-add buffers
    overlap_add_buffer1 = np.zeros((overlap))
    overlap_add_buffer2 = np.zeros((overlap))
    
    # Process the audio signals in chunks
    output = np.zeros((len(audio_arrays[0])))
    for i in range(0, len(audio_arrays[0]) - window_size, overlap):
        # Extract a chunk of audio data from each signal
        chunk1 = audio_arrays[0][i:i+window_size]
        chunk2 = audio_arrays[1][i:i+window_size]
        # Apply a window function to the data
        window = np.hanning(window_size)
        chunk1 *= window
        chunk2 *= window

        # Add the overlap buffers to the start of the data
        chunk1[:overlap] += overlap_add_buffer1
        chunk2[:overlap] += overlap_add_buffer2

        # Calculate the energy of each channel in each chunk
        energies1 = np.sqrt(np.mean(np.square(chunk1), axis=0))
        energies2 = np.sqrt(np.mean(np.square(chunk2), axis=0))
        print(energies1,energies2)
        

        # Select the chunk with the higher energy for each channel
        mask1 = energies1 > energies2
        mask2 = ~mask1
        vars_arr=np.array([mask1,mask2])
        coefficients = np.where(vars_arr == 1, 0.6, 0.4 )
        # Mix the chunks using the selected gains
        mixed_chunk = coefficients[0] * chunk1 + coefficients[1] * chunk2

        # Reconstruct the audio signal by summing the mixed chunks
        output[i:i+window_size] += mixed_chunk

        # Update the overlap-add buffers
        overlap_add_buffer1 = chunk1[-overlap:]
        overlap_add_buffer2 = chunk2[-overlap:]
        
   
        # Define the filter parameters
        cutoff_freq = 7000  # Hz
        nyquist_freq = 0.5 * 44100  # Nyquist frequency
        cutoff_norm = cutoff_freq / nyquist_freq
        order = 6  # Filter order

        # Create the filter coefficients
        b, a = sg.butter(order, cutoff_norm, btype='lowpass')

        # Apply the filter to the audio signal
        filtered_audio = sg.filtfilt(b, a, output*32767)



    # Save mixed signal to file
    with wave.open('mixed_wighted_avg_signal.wav', 'wb') as wav:
        wav.setparams(audio_params[0])
        wav.writeframes((filtered_audio+10).astype(np.int16).tobytes())#output * 32767
        

