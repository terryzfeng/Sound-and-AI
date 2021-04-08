import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.wav"

# load waveform with Librosa
signal, sr = librosa.load(file, sr=22050)
# signal is a numpy array
# len(signal) = sr * time

'''
librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
'''

# fft - fast fourier transform
fft = np.fft.fft(signal)

magnitude = np.abs(fft) # magnitude of all frequencies
frequency = np.linspace(0, sr, len(magnitude)) # how much each frequency occurs

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

'''
plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
'''

# stft - short time fourier transform
n_fft = 2048 # window for single fourier transform
hop_length = 512 # window shift

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

'''
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time (s)")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()
'''

# MFCC
mfcc = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length)
plt.xlabel("Time (s)")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()