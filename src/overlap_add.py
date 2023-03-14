# Import Packages
import numpy as np
from scipy import signal
import noisereduce as nr


def mix(signals, sampling_rate=44100, bit_depth=16, cf=0.7,low_channel=1):

    """_summary_

    Args:
        signals (Numpy Array): Multi-channel signal at different gain levels
        
        
    Output:
        The new synthetized signal by mixing the input signals using overlap_add method
    """

    
    # Define the window size and overlap
    frame_len = int(0.03 * sampling_rate)
    frame_hop = int(0.015 * sampling_rate)
    
    
    # Process the audio signals in chunks
    
    # Calculate the maximum absolute value of each column
    normalized_signals=signals/np.amax(np.abs(signals), axis=0)
    
    output = np.zeros((signals.shape[0]))
    
    for i in range(0, signals.shape[0] - frame_len, frame_hop):
        # Extract a chunk of audio data from each signal
        chunk1 = normalized_signals[i:i+frame_len,0]
        chunk2 = normalized_signals[i:i+frame_len,1]
        
        # Detect the clipping and keep the chunk with high energy
        is_clipped_1=is_clipped(chunk1)
        is_clipped_2=is_clipped(chunk2)
        
        if is_clipped_1:
            frame_to_keep=chunk2
        elif is_clipped_2:
            frame_to_keep=chunk1
        else:
            
            # Calculate the energy of each channel in each chunk
            energies1 = np.sqrt(np.mean(np.square(chunk1), axis=0))
            energies2 = np.sqrt(np.mean(np.square(chunk2), axis=0))
            

            # Select the chunk with the higher energy for each channel
            mask1 = energies1 > energies2
            mask2 = ~mask1
            vars_arr=np.array([mask1,mask2])
            coefficients = np.where(vars_arr == 1, cf, 1-cf )
            # Mix the chunks using the selected gains
            frame_to_keep = coefficients[0] * chunk1 + coefficients[1] * chunk2

        
        # Apply a window function to the data
        window = signal.hann(frame_len)
        frame_to_keep = (frame_to_keep*window)     
        
        # Reconstruct the audio signal by summing the mixed chunks
        output[i:i+frame_len] = frame_to_keep

        
    reduced_noise = nr.reduce_noise(y=output, y_noise=signals[:,low_channel], sr=sampling_rate, prop_decrease=0.9)
    return reduced_noise

def is_clipped(signal_chunk):
    # Define clipping threshold value based on bit depth
    threshold = 0.98

    # Check if any value in the signal chunk exceeds the threshold
    indices=np.argwhere(np.abs(signal_chunk.astype(np.int32)) >= threshold)
    
    return len(indices)!=0
        
if __name__ == "__main__":
    # Get signals at different gain levels using the script
    sample_rate1, signals1 = wavfile.read('audio_files/one_channel_clipping/channel_1.wav')
    sample_rate2, signals2 = wavfile.read('audio_files/one_channel_clipping/channel_2.wav')
    signals=np.array([signals1,signals2]).T
    output = mix(signals, 22050)
    save_audio('mixed.wav',output, 22050, 16)

