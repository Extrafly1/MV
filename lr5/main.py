import soundfile as sf
import numpy as np
import math
import cmath

def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        s = 0j
        for n in range(N):
            angle = -2j * math.pi * k * n / N
            s += signal[n] * cmath.exp(angle)
        result.append(s)
    return np.array(result)

def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        s = 0j
        for k in range(N):
            angle = 2j * math.pi * k * n / N
            s += spectrum[k] * cmath.exp(angle)
        result.append(s / N)
    return np.array(result).real

def noise_reduction(x):
    print("Выполняется ручное Фурье...")
    spec = dft(x)
    ampl = np.abs(spec)
    phase = np.angle(spec)

    noise_level = np.percentile(ampl, 10)
    threshold = noise_level * 1.3

    ampl_filtered = np.where(ampl > threshold, ampl - threshold, 0)

    spec_filtered = ampl_filtered * np.exp(1j * phase)

    print("Обратное Фурье...")
    clean = idft(spec_filtered)
    clean /= np.max(np.abs(clean))
    return clean

input_file = "input.mp3"
signal, samplerate = sf.read(input_file)

if len(signal.shape) > 1:
    signal = signal.mean(axis=1)

clean = noise_reduction(signal)

sf.write("output_clean_main.wav", clean, samplerate)

print("Ручное преобразование Фурье выполнено.")
