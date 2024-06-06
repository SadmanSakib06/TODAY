# import numpy as np
# import matplotlib.pyplot as plt
#
# fs = 8000
# N = 8
# t = np.arange(N) / fs
#
# X_t = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t + (3 * np.pi / 4))
#
# print(len(X_t))
# print(X_t)
#
# def DFT(x):
#     N = len(x)
#     X = np.zeros(N, dtype=complex)
#     for k in range(N):
#         for n in range(N):
#             X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
#     return X
#
# def IDFT(X):
#     N = len(X)
#     X = np.zeros(N, dtype=complex)
#     for n in range(N):
#         for k in range(N):
#             X[n] += X[k] * np.exp(2j * np.pi * k * n / N)
#     return X / N
#
# X_f = DFT(X_t)
#
#
# magnitude_spectrum = np.abs(X_f)
# X_t_reconstructed = IDFT(X_f)
#
# print(X_t_reconstructed)
# phase_spectrum = np.angle(X_f)
#
# frequencies = np.fft.fftfreq(N, 1 / fs)
#
# plt.figure(figsize=(12, 10))
#
# plt.subplot(3, 2, 1)
# plt.plot(t, X_t, 'o-')
# plt.title('Time-Domain Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.grid()
#
# plt.subplot(3, 2, 2)
# plt.stem(frequencies, magnitude_spectrum, 'b')
# plt.title('Magnitude Spectrum')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude')
# plt.grid()
#
#
# plt.subplot(3, 2, 3)
# plt.plot(t, X_t_reconstructed.real, 'o-')
# plt.title('Reconstructed Time-Domain Signal from IDFT')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.grid()
#
# plt.subplot(3, 2, 4)
# plt.stem(frequencies, phase_spectrum, 'g')
# plt.title('Phase Spectrum')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Phase [radians]')
# plt.grid()
#
# plt.tight_layout()
# plt.show()







# import numpy as np
# import matplotlib.pyplot as plt
# import math
#
# Frequency_1 = 1000
# Amplitude_1 = 1
# phaseShift_1 = 0
#
# Frequency_2 = 2000
# Amplitude_2 = 0.5
# phaseShift_2 = (3 * np.pi) / 4
#
# sampleRate = 8000
# startTime = 0
# endTime = 0.001
#
# sampleCount = sampleRate * (endTime - startTime)
#
# t = np.linspace(startTime, endTime, int(sampleCount), endpoint=False)
# x = (Amplitude_1 * np.sin(2 * np.pi * Frequency_1 * t + phaseShift_1)
#      + Amplitude_2 * np.sin(2 * np.pi * Frequency_2 * t + phaseShift_2))
#
# plt.figure(figsize=(10,5))
#
# plt.stem(t,x)
# plt.title('Sampling')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.show()                  #Sampling
#
# N = 8
# fs = 8000
# X_real = []
# X_img = []
# magnitude = []
#
# for i in range(0,N):
#     real = 0
#     for n in range(0,N):
#         real = real + x[n] * np.cos(math.radians(2*180*i*n)/N)
#         X_real.append(round(real,3))
#
# for i in range(0,N):
#     img = 0
#     for n in range(0,N):
#         img = img - x[n] * np.sin(math.radians(2*180*i*n)/N)
#         X_img.append(round(img,3))
#
# for i in range(0,N):
#     magnitude.append(math.sqrt(X_real[i] * X_real[i] + X_img[i] * X_img[i]))
#
# plt.figure(figsize=(10,5))
#
# plt.stem(range(0,8), magnitude)
# plt.title('Phase of X(m)')
# plt.xlabel('Frequency')
# plt.ylabel('Angle')
# plt.show()
#
#
#
#
#



import numpy as np
import matplotlib.pyplot as plt
import math

Frequency_1 = 1000
Amplitude_1 = 1
phaseShift_1 = 0

Frequency_2 = 2000
Amplitude_2 = 0.5
phaseShift_2 = (3 * np.pi) / 4

sampleRate = 8000
startTime = 0
endTime = 0.001

sampleCount = sampleRate * (endTime - startTime)

t = np.linspace(startTime, endTime, int(sampleCount), endpoint=False)
x = (Amplitude_1 * np.sin(2 * np.pi * Frequency_1 * t + phaseShift_1)
     + Amplitude_2 * np.sin(2 * np.pi * Frequency_2 * t + phaseShift_2))

plt.figure(figsize=(12, 10))
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust the spacing between subplots

# Time-domain signal
plt.subplot(3, 2, 1)
plt.stem(t, x)
plt.title('Sampling')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

# Magnitude spectrum
N = 8
fs = 8000
X_real = []
X_img = []
magnitude = []

for i in range(0, N):
    real = 0
    for n in range(0, N):
        real = real + x[n] * np.cos(math.radians(2 * 180 * i * n) / N)
    X_real.append(round(real, 3))

for i in range(0, N):
    img = 0
    for n in range(0, N):
        img = img - x[n] * np.sin(math.radians(2 * 180 * i * n) / N)
    X_img.append(round(img, 3))

for i in range(0, N):
    magnitude.append(math.sqrt(X_real[i] * X_real[i] + X_img[i] * X_img[i]))

plt.subplot(3, 2, 2)
plt.stem(range(0, 8), magnitude)
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

phase = []
for i in range(0,N):
    if X_real[i] != 0:
        phase.append(math.degrees(np.arctan(X_img[i] / X_real[i])))
    else:
        if X_img[i] < 0:
            phase.append(-90)
        elif X_img[i] > 0:
            phase.append(90)
        else: phase.append(0)

plt.subplot(3, 2, 3)
plt.stem(range(0,8), phase)
plt.title('Phase of X(m)')
plt.xlabel('Frequency')
plt.ylabel('Angle')

reconstructed_x = []
for n in range(N):
    temp = 0
    for m in range(N):
        temp = temp + X_real[m] * np.cos((2 * np.pi * m * n) / N) - X_img[m] * np.sin((2 * np.pi * m * n) / N)
    reconstructed_x.append(temp/N)
print(reconstructed_x)
plt.subplot(3,2,4)
#plt.plot(t, reconstructed_x)
plt.stem(range(0,8), reconstructed_x)
plt.title('IDFT')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

plt.show()