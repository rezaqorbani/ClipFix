# Import Packages
import numpy as np
from scipy import signal
from sequence_generation2 import gen
import matplotlib.pyplot as plt

# Get signals at different gain levels using the script
signals=gen(plot_flag = False)
two_channel_signal = signals[:,0:5:4]

def mix(signals):
    """_summary_

    Args:
        signals (Numpy Array): Multi-channel signal at different gain levels
        
    Output:
        The new synthetized signal by mixing the input signals using overlap_add method
    """
    

    # Define sampling rate
    sr = 16000
    bit_depth= 16

    
    # Define the window size and overlap
    frame_len = int(0.03 * sr)
    frame_hop = int(0.015 * sr)

    # Initialize the overlap-add buffer
    overlap_add_buffer = np.zeros((frame_hop))
    
    
    # Process the audio signals in chunks
    output = np.zeros((signals.shape[0]))
    
    for i in range(0, signals.shape[0] - frame_len, frame_hop):
        # Extract a chunk of audio data from each signal
        chunk1 = signals[i:i+frame_len,0]
        chunk2 = signals[i:i+frame_len,1]
        
        # Detect the clipping and keep the chunk with high energy
        is_clipped_1=is_clipped(chunk1,bit_depth)
        is_clipped_2=is_clipped(chunk2,bit_depth)
        
        if is_clipped_1:
            frame_to_keep=chunk2
        elif is_clipped_2:
            frame_to_keep=chunk1
        else:
            # TODO: Detect quantisation noise
            
            
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
            frame_to_keep = coefficients[0] * chunk1 + coefficients[1] * chunk2

        
        # Apply a window function to the data
        window = signal.hann(frame_len)
        frame_to_keep = (frame_to_keep*window)     

        # Add the overlap buffers to the start of the data
        frame_to_keep[:frame_hop] += overlap_add_buffer
        
        # Reconstruct the audio signal by summing the mixed chunks
        output[i:i+frame_len] += frame_to_keep

        # Update the overlap-add buffers
        overlap_add_buffer = frame_to_keep[-frame_hop:]

    plt.figure()
    plt.plot(output)
    plt.show()
    
    return output


        
def detect_clipping(signal_chunk, bit_depth):
    threshold = 2 ** (bit_depth - 1) - 1
    clipped_segments = []
    inside_clip = False
    clip_start = 0
    clip_end = 0
    
    for i, sample in enumerate(signal_chunk):
        nmin=min(signal_chunk)
        nmax=max(signal_chunk)
        #sample equal to or extremely close to max or min
        if (sample <= nmin+1) or (sample >= nmax-1):
            if not inside_clip:
                # declare we are inside clipped segment
                inside_clip = True
                # this is the first clipped sample
                clip_start = i
        elif inside_clip:
            inside_clip = False # no longer inside clipped segment
            clip_end = i-1  # previous sample is end of segment
            # save segment as tuple
            clipped_segment = [clip_start, clip_end]
            # store tuple in list of clipped segments
            clipped_segments.append(clipped_segment)
    clipped_segments=np.array(clipped_segments)
    is_clipped=np.any(np.diff(clipped_segments,axis=1)!=0)
    
    return is_clipped, clipped_segments

def is_clipped(signal_chunk, bit_depth):
    # Define clipping threshold value based on bit depth
    threshold = 2 ** (bit_depth - 1) - 1
    # Check if any value in the signal chunk exceeds the threshold
    indices=np.argwhere(np.abs(signal_chunk.astype(np.int32)) >= threshold)
    
    return len(indices)!=0

def quantization_noise(signal_chunk):
    return None

#print(is_clipped(signals[:,7],16))
#detect_clipping(signals[:,7],16)
mix(two_channel_signal)
        

