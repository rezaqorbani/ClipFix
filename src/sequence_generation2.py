
"""
Author: Pablo Perez Zarazaga
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt


"""
Generate signals with multiple resolution for analysis.
Signals will be quantized to 16 bit integer it depth... Begin signal generation with 24 or 32 bits and add dynamic option later.

- Harmonic signals - Sequence of sine signals at multiples of the initial frequency "Pitch".
- Random signal - Generate signals with sine signals in multiple prime frequencies. Ensure they are all mutual primes.

- Each sine signal has a random phase.
- Amplitude going down uniformly -- [option] dB per octave.

- Randomize gain envelope for multichannel division.
"""

def gen(plot_flag = False):
    """
    Routine to generate test audio signal with great bit depth, segment and de-clip.
    """

    # Initialise parameters

    # Signal parameters
    signal_duration = 60
    sr = 16000
    harm_freq = None

    # Quantization parameters
    n_bits_in = 24
    n_bits_q = 16

    # OLA for signal reconstruction
    frame_len = int(0.03 * sr)
    frame_hop = int(0.015 * sr)

        
    # Generate audio samples Function returns signals with multiple levels (Divide segmentation to use generation funcion in other places).

    unscaled_signal = generate_signal(sig_len = signal_duration, sr = sr, num_bits = n_bits_in, harmonic = harm_freq, plot_flag = plot_flag)
    print(unscaled_signal.shape)

    # Multi level segmentation and divide in multiple channels

    q_signals = divide_and_quantize(sig_input = unscaled_signal, n_bits_in = n_bits_in, n_bits_q = n_bits_q, plot_flag = plot_flag)
    print(q_signals.shape)
    
    if plot_flag==True:
        plt.figure()
        plt.plot(q_signals[:,1])
        
        plt.show()
    return q_signals


def generate_signal(sig_len = 60, sr = 16000, num_bits = 24, harmonic = None, plot_flag = False):

    """
    Generate signal for testing of multigain recording.
    Params:
        sig_len: Duration of the signal in seconds.
        sr: Sampling frequency in Hz
        num_bits: Bit resolution of the generated signal.
        harmonic: Frequency at which to generate the harmonics in HZ. If None, generate signal with tones in prime frequencies.
    """

    # Initial params
    num_samples = int(sig_len * sr)
    nyquist_freq = int(sr / 2)
    # Bit value
    bit_lim = np.power(2,num_bits - 1)
    print('Bit limits values in range = ' + str([-bit_lim,bit_lim - 1]))

    # Generate signal as a sequence of sine signals. Frequnecy dependent amplitude set as ...

    out_sig = np.zeros((num_samples,1))

    if harmonic is not None:
        freqs = np.arange(harmonic,nyquist_freq,harmonic)
        print('Generating signal with ' + str(freqs.shape[0]) + ' harmonics.')

        phases = np.linspace(np.pi, -np.pi, freqs.shape[0])

        for f,phi in zip(freqs,phases):
            omega = 2*np.pi*f/sr
            phi = np.random.random() * 2 * np.pi

            sample_idx = np.arange(0,num_samples).reshape((-1,1))

            sig_single_f = np.sin(omega*sample_idx + phi)

            out_sig += sig_single_f

    else:
            # Generate list of prime numbers.
        prime_list = np.concatenate((np.array([2]),primesfrom3to(n = nyquist_freq)))
        print(prime_list.shape)
        print(prime_list)
        
        for f in prime_list:
            omega = 2*np.pi*f/sr
            phi = np.random.random() * 2 * np.pi

            sample_idx = np.arange(0,num_samples).reshape((-1,1))

            sig_single_f = np.sin(omega*sample_idx + phi)

            out_sig += sig_single_f
        

    # Amplitude envelope over time. Random values between 0 and 1 --  Match levels for gains between the exceeding number of bits between 0 and 1

    scale = bit_lim / np.amax(out_sig)

    out_sig_q = scale * out_sig
    out_sig_q = out_sig_q.astype(np.intc)

    out_sig /= np.amax(out_sig)

    lin_spec = np.fft.rfft(out_sig[:nyquist_freq].squeeze(), n = nyquist_freq) # Due to random phasing, uniform window will end up in weird values.


    # Segment signal
    segmented_sig = segment_and_rescale(out_sig_q, sr = sr)

    if plot_flag:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(out_sig_q)
        plt.subplot(2,1,2)
        plt.plot(segmented_sig)
        plt.figure()
        plt.plot(20 * np.log10(np.abs(lin_spec) + 1e-10))
        plt.show()

    return segmented_sig

def primesfrom3to(n = 8000):
    """
    Returns a array of primes, 3 <= p < n
    Params:
        n: last prime number in the list to return.
    """
    sieve = np.ones(n//2, dtype=bool)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i//2]:
            sieve[i*i//2::i] = False
    return 2*np.nonzero(sieve)[0][1::]+1

def segment_and_rescale(signal, sr, max_segment_len = 1, min_segment_len = 0.1):
    """
    Divide signal in time segments and scale each segment to a random level.
    Params:
        signal: single channel audio signal with shape (n,1)
        sr: Sampling rate
        max_segment_len: 
    """

    new_sig = np.zeros(signal.shape)

    segment_lengths = np.arange(min_segment_len, max_segment_len, 0.1) * sr # Posible segment lengths in intervals of 100 ms
    segment_lengths = segment_lengths.astype(np.intc)
    scales = np.power(10,np.arange(-30,0, 3)/10) # Minimum 8 bits difference 24 dB.

    segment_start = 0

    end_signal = False

    while not end_signal:
        segment_end = segment_start + np.random.choice(segment_lengths)

        if segment_end >= len(signal):
            segment_end = len(signal)
            end_signal = True

        new_sig[segment_start:segment_end] = signal[segment_start:segment_end] * np.random.choice(scales)

        segment_start = segment_end

    return np.round(new_sig).astype(np.intc)

def divide_and_quantize(sig_input, n_bits_in, n_bits_q, plot_flag = False):
    """
    Function to convert a signal sampled with high bit depth into n "quantized" signals with lower number of bits and different gains.

    Input:
        - sig_input: Input signal quantized with high bit-depth.
        - n_bits_in: Bit resolution of the input signal.
        - n_bits_out: Bit resolution of the output signals.

    The division of signals will be done according to the difference in the number of bits between input and output, such that if the input is 24 bits and the output 16, the function will return 8 signals. This is only done because this is a test signal, we assume that signals will enter with different gains from the multilevel recording setup.

    Retur:
        - Matrix of signals with one channel for each bit in the resolution difference.
    """

    assert(n_bits_in > n_bits_q)

    n_levels = n_bits_in - n_bits_q
    diff_scale = 2**n_levels

    bit_max = 2**(n_bits_q-1)

    if n_bits_q == 16:
        out_datatype = np.int16
    else:
        NotImplementedError("Ouptut bit-depth not implemented")
    
    # Save separated signals as a matrix with a column for each signal.

    separated_signals = np.zeros((sig_input.shape[0],n_levels), dtype = out_datatype)

    for i in range(n_levels):
        scale = 2**i
        clip_level = 2**(n_bits_in-i)

        # Clip signal to the corresponding level
        sig_i = np.clip(sig_input,-clip_level, clip_level)

        # Scale signal to fit in [min,max]
        sig_i *= scale

        # Divide samples down to the corresponding
        sig_i = np.divide(sig_i,diff_scale)

        separated_signals[:,i] = sig_i.reshape((-1,)).astype(out_datatype)

    if n_levels == 8 and plot_flag:
        plt.figure()
        for i in range(1,9):
            plt.subplot(4,2,i)
            plt.plot(separated_signals[:,i-1])
            plt.ylim([-bit_max-10,bit_max+10])
        plt.show(block=False)
    
    return separated_signals

if __name__ == "__main__":
    gen(plot_flag = False)
    print("Signals generated")