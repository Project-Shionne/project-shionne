import numpy as np
from scipy.io.wavfile import write

sr = 44100
duration = 600  

# 1. Binaural Beat
t = np.linspace(0, duration, int(sr * duration))
left = np.sin(2 * np.pi * 100 * t)
right = np.sin(2 * np.pi * 103 * t)
stereo = np.vstack([left, right]).T * 0.3

# 2. Pink Noise (approx)
def pink_noise(n):
    b = np.random.randn(n)
    b = np.cumsum(b)
    return b / np.max(np.abs(b))

pink = pink_noise(len(t)) * 0.1
stereo[:, 0] += pink
stereo[:, 1] += pink

# 3. 62.5 Hz drone
drone = np.sin(2 * np.pi * 62.5 * t) * 0.05
stereo[:, 0] += drone
stereo[:, 1] += drone

write('synaptic_calm.wav', sr, stereo.astype(np.float32))
