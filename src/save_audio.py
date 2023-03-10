
import numpy as np
import torch
import torchaudio


def save_audio(path, audio_array, rate, n_bits): 
    """ Save an audio locally from a numpy array

    Args:
        path (String): Path to the audio file
        audio_array (Numpy Array): Audio array
        rate (Int): Sampling Rate
        n_bits (Int): number of bits
    """
    waveform=torch.from_numpy(audio_array.reshape(1, -1))
    torchaudio.save(path, waveform, rate, bits_per_sample=n_bits, format='wav')
    
