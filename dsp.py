
import numpy as np

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, signal_frequency=618):
    
    # split V and I from interleaved raw data
    # EI sends data as [V0, I0, V1, I1, V2, I2, ...]
    V = np.array(raw_data[0::2])
    I = np.array(raw_data[1::2])

    n = len(V)
    t = np.arange(n) / sampling_freq  # time axis in seconds

    # --- remove DC offset ---
    V = V - np.mean(V)
    I = I - np.mean(I)

    # --- reference sine and cosine at signal frequency ---
    ref_sin = np.sin(2 * np.pi * signal_frequency * t)
    ref_cos = np.cos(2 * np.pi * signal_frequency * t)

    # --- DFT-like projection to get magnitude and phase ---
    # Voltage
    V_real = np.mean(V * ref_cos)
    V_imag = np.mean(V * ref_sin)
    V_mag   = np.sqrt(V_real**2 + V_imag**2)
    V_phase = np.arctan2(V_imag, V_real)

    # Current
    I_real = np.mean(I * ref_cos)
    I_imag = np.mean(I * ref_sin)
    I_mag   = np.sqrt(I_real**2 + I_imag**2)
    I_phase = np.arctan2(I_imag, I_real)

    # --- Impedance ---
    Z_mag   = V_mag / (I_mag + 1e-10)           # magnitude |Z| in ohms
    Z_phase = V_phase - I_phase                  # phase angle in radians
    Z_phase_deg = np.degrees(Z_phase)            # phase angle in degrees

    Z_real = Z_mag * np.cos(Z_phase)             # resistance  (R)
    Z_imag = Z_mag * np.sin(Z_phase)             # reactance   (X)

    # --- extra stats ---
    V_rms = np.sqrt(np.mean(V**2))
    I_rms = np.sqrt(np.mean(I**2))

    features = [
        Z_mag,          # impedance magnitude |Z|
        Z_phase_deg,    # phase angle in degrees
        Z_real,         # real part (resistance R)
        Z_imag,         # imaginary part (reactance X)
        V_rms,          # V RMS
        I_rms,          # I RMS
    ]

    return { 'features': features }
```

---

**`requirements.txt`**
```
numpy
