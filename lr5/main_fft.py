import soundfile as sf
import numpy as np
import cmath
import math

def fft(x):
    N = len(x)
    if N == 1:
        return x

    if N % 2 != 0:
        raise ValueError("Длина входного массива должна быть степенью двойки")

    even = fft(x[0::2])
    odd = fft(x[1::2])

    T = [0] * (N // 2)
    for k in range(N // 2):
        angle = -2j * math.pi * k / N
        T[k] = cmath.exp(angle) * odd[k]

    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]

def ifft(X):
    N = len(X)
    if N == 1:
        return X

    if N % 2 != 0:
        raise ValueError("Длина спектра должна быть степенью двойки")

    even = ifft(X[0::2])
    odd = ifft(X[1::2])

    T = [0] * (N // 2)
    for k in range(N // 2):
        angle = 2j * math.pi * k / N
        T[k] = cmath.exp(angle) * odd[k]

    return [(even[k] + T[k]) / 2 for k in range(N // 2)] + \
           [(even[k] - T[k]) / 2 for k in range(N // 2)]

def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p

def noise_reduction_fft(signal):
    original_len = len(signal)
    N = next_pow2(original_len)

    padded = np.zeros(N)
    padded[:original_len] = signal

    print("Выполняется FFT...")
    spectrum = fft(list(padded))

    ampl = np.abs(spectrum)
    phase = np.angle(spectrum)

    noise_level = np.percentile(ampl, 10)
    threshold = noise_level * 1.3

    ampl_filtered = np.where(ampl > threshold, ampl - threshold, 0)
    filtered_spectrum = ampl_filtered * np.exp(1j * phase)

    print("Выполняется IFFT...")
    cleaned = ifft(filtered_spectrum)

    cleaned = np.real(np.array(cleaned))
    cleaned /= np.max(np.abs(cleaned))

    return cleaned[:original_len]

input_file = "input.mp3"
output_file = "output_clean_fft.wav"

signal, samplerate = sf.read(input_file)

if len(signal.shape) > 1:
    signal = signal.mean(axis=1)

print("Начало обработки...")
clean = noise_reduction_fft(signal)

sf.write(output_file, clean, samplerate)
print("Файл сохранён как:", output_file)
