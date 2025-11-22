import soundfile as sf
import numpy as np
from scipy.fft import fft, ifft

input_file = "input.mp3"
signal, samplerate = sf.read(input_file)

if len(signal.shape) > 1:
    signal = signal.mean(axis=1)

spec = fft(signal)
ampl = np.abs(spec)
phase = np.angle(spec)

noise_level = np.percentile(ampl, 10)
threshold = noise_level * 1.3

ampl_filtered = np.where(ampl > threshold, ampl - threshold, 0)

spec_filtered = ampl_filtered * np.exp(1j * phase)
clean_signal = np.real(ifft(spec_filtered))

clean_signal /= np.max(np.abs(clean_signal))

sf.write("output_clean.wav", clean_signal, samplerate)

print("Файл сохранён как output_clean.wav")
