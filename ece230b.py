# useful functions for ECE 230B (Digital Communications) with Prof. Ian Roberts, Spring 2025
import numpy as np
import math
import matplotlib.pyplot as plt

def gen_rand_qam_symbols (N, M=4):
    """
    Generate N random symbols from a square M-QAM constellation normalized to unit average symbol energy.
    
    Inputs:
        N: Number of symbols to generate.
        M: Order of QAM constellation. Default is 4 (QPSK).
    
    Outputs:
        symbols: N randomly selected M-QAM symbols.
        constellation: The full M-QAM constellation. 
    """

    if M <= 0 or (M**0.5) % 2 != 0:
        raise ValueError("M must be a positive power of 2.")
    
    # Calculate bits per symbol (bps)
    bps = int(math.log2(M))

    # Generate the M-QAM constellation
    constellation = []
    m_sqrt = int(M**0.5)
    for idx0 in range(m_sqrt):
        for idx1 in range(m_sqrt):
            constellation.append( (2*idx0-(m_sqrt-1)) + 1j*(2*idx1-(m_sqrt-1)) )

    # Normalize the constellation to unit average symbol energy
    constellation /= np.sqrt(np.mean(np.abs(constellation)**2))

    # Randomly select N symbols from the constellation
    symbols = np.random.choice(constellation, N)

    return symbols, constellation
    
def create_pulse_train (symbols, sps):
    """
    Forms a pulse train from a sequence of symbols by inserting zeros between symbols.
    
    Inputs: 
        symbols: QAM symbols.
        sps: Samples per symbol, i.e., the number of discrete-time samples from one symbol to the next.

    Outputs: 
        t: The time vector (samples) for the pulse train, ranging from 0 to len(symbols)*sps-1.
        pulse_train: A discrete-time signal where each symbol is seperated by (sps-1) zeros.
    """
    # Generate time vector (samples) for the pulse train
    t = np.arange(len(symbols) * sps)

    # Generate pulse train from vector of zeros
    pulse_train = np.zeros(len(symbols) * sps, dtype=complex)
    pulse_train[::sps] = symbols

    return t, pulse_train

def get_rc_pulse (beta, span, sps): 
    """
    Generates a raised cosine pulse shape.

    Inputs: 
        beta: Rolloff factor (between 0 and 1, inclusive).
        span: The integer number of symbol durations spanned by the pulse, not including the symbol at t=0. 
        sps: Samples per symbol. 

    Output: 
        t: The time vector (samples) for the pulse, ranging from -span/2 to span/2.
        pulse: A raised cosine pulse (normalized to unit peak), symmetric and centered at t=0. The number of zero crossings should be equal to span. 
    """

    if beta < 0 or beta > 1: 
        raise ValueError("Rolloff factor (beta) must be between 0 and 1.")
    if span < 1 or not isinstance(span, int):
        raise ValueError("Span must be a positive integer.")
    if sps < 1 or not isinstance(sps, int):
        raise ValueError("Samples per symbol (sps) must be a positive integer.")
    
    # Generate time vector (samples) assuming span is single-sided
    # length = 2*span * sps + 1
    length = span*sps + 1
    t = np.linspace(-span/2, span/2, length)

    # Calculate raised cosine pulse
    if (beta == 0):
        pulse = np.sinc(t)
    else:
        pulse = np.sinc(t) * ( np.cos(np.pi * beta * t) / (1 - ((2 * beta * t)**2)) )
        pulse[np.abs(2 * beta * t) == 1] = (np.pi / 4) * np.sinc(1 / (2 * beta))

    # Normalize to unit peak
    # pulse /= np.max(np.abs(pulse))

    # plt.plot(t, pulse)
    # plt.grid(True)
    # plt.show()

    return t, pulse

def get_rrc_pulse (beta, span, sps): 
    """
    Generates a root raised cosine pulse shape.

    Inputs: 
        beta: Rolloff factor (between 0 and 1, inclusive).
        span: The integer number of symbol durations spanned by the pulse, not including the symbol at t=0. 
        sps: Samples per symbol. 

    Output: 
        t: The time vector (samples) for the pulse, ranging from -span/2 to span/2.
        pulse: A root raised cosine pulse (normalized to unit peak), symmetric and centered at t=0. 
    """

    if beta < 0 or beta > 1: 
        raise ValueError("Rolloff factor (beta) must be between 0 and 1.")
    if span < 1 or not isinstance(span, int):
        raise ValueError("Span must be a positive integer.")
    if sps < 1 or not isinstance(sps, int):
        raise ValueError("Samples per symbol (sps) must be a positive integer.")
    
    # Generate time vector (samples)
    # length = 2*span * sps + 1
    length = span*sps + 1
    t = np.linspace(-span/2, span/2, length)

    # Calculate root raised cosine pulse
    if (beta == 0):
        pulse = np.sinc(t)
    else:
        # impulse response from Wikipedia
        pulse = np.zeros_like(t)
        t_nonzero = t[(np.abs(t) != 0) & (np.abs(4 * beta * t) != 1)]
        pulse[(np.abs(t) != 0) & (np.abs(4 * beta * t) != 1)] = ( np.sin(np.pi * t_nonzero * (1 - beta)) + 4 * beta * t_nonzero * np.cos(np.pi * t_nonzero * (1 + beta)) ) / (np.pi * t_nonzero * (1 - (4 * beta * t_nonzero)**2))
        pulse[np.abs(t) == 0] = 1 + beta * (4 / np.pi - 1)
        pulse[np.abs(4 * beta * t) == 1] = beta / np.sqrt(2) * ( (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)) )

    # Normalize to unit norm
    pulse /= np.sqrt(np.sum(np.abs(pulse)**2))
    # pulse /= np.max(np.abs(pulse))

    # plt.plot(t, pulse)
    # plt.grid(True)
    # plt.show()

    return t, pulse

def gen_primes (N): 
    """
    Generates a list of prime numbers up to N (inclusive). "Borrowed" from https://rebrained.com/?p=458 

    Inputs:
        N: The upper limit for generating primes.

    Output:
        primes: A list of prime numbers up to N.
    """
    
    primes=np.arange(3,N+1,2)
    isprime=np.ones((N-1)//2,dtype=bool)
    for factor in primes[:int(math.sqrt(N))//2]:
        if isprime[(factor-2)//2]: isprime[(factor*3-2)//2::factor]=0
    return np.insert(primes[isprime],0,2)

def gen_zadoff_chu_sequence (N, root=1): 
    """
    Generates a Zadoff-Chu sequence of length N_zc with a given root index.

    Inputs:
        N: Length of the Zadoff-Chu sequence.
        root: Root index for the sequence (default is 1).

    Output:
        N_zc: Length of the Zadoff-Chu sequence equal to largest prime less than or equal to N. 
        zc_seq: The generated Zadoff-Chu sequence.
    """
    if N <= 0 or not isinstance(N, int):
        raise ValueError("N must be a positive integer.")
    if N != gen_primes(N)[-1]: 
        N = gen_primes(N)[-1]
        print(f"Warning: N is not prime, choosing the largest prime less than N: {N}.")
    if root < 1 or root >= N:
        raise ValueError("Root index must be in the range [1, N-1].")

    n = np.arange(N)
    zc_seq = np.exp(-1j * np.pi * root * n * (n + 1) / N)

    return N, zc_seq