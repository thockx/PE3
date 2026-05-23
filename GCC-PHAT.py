import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
CSV_FILE = r"C:\Users\thijs\Documents\GitHub\PE3\2026-05-22-15-48-20.csv"
SAMPLE_RATE = 200000           # Sample rate [Hz]
MIC_SPACING = 0.055            # Distance between the two microphones [m]
SPEED_OF_SOUND = 343.0         # Speed of sound [m/s]

# ── Load signals ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
x = df.iloc[:, 1].to_numpy(dtype=float)
y = df.iloc[:, 2].to_numpy(dtype=float)

# ── Plot signals ───────────────────────────────────────────────────────────
plt.plot(x, label='mic 1')
plt.plot(y, label='mic 2')
plt.legend()
plt.show()

# ── Moving average helper ──────────────────────────────────────────────────
def mov_avg(data, N=1000000):
    if N <= 1:
        return data
    return np.convolve(data, np.ones(N)/N, mode='same')

# ── GCC-PHAT helper ────────────────────────────────────────────────────────
def gcc_phat(x, y):
    n = len(x) + len(y) - 1
    nfft = 1 << (n - 1).bit_length()   # next power of 2
    X = np.fft.fft(x, n=nfft)          # zero-padded
    Y = np.fft.fft(y, n=nfft)          # zero-padded
    # smooth the magnitude spectra before normalisation
    X_mag = mov_avg(np.abs(X))
    Y_mag = mov_avg(np.abs(Y))
    G = X * np.conj(Y)
    G /= (X_mag * Y_mag) + 1e-9
    cc = np.real(np.fft.ifft(G))
    cc = np.concatenate([cc[nfft // 2:], cc[: nfft // 2]])
    lags = np.arange(-nfft // 2, nfft // 2) / SAMPLE_RATE
    return cc, lags, nfft

tau_max = MIC_SPACING / SPEED_OF_SOUND

gcc, lags, N_fft = gcc_phat(x, y)

# ── Remove near-zero lags (suppress τ=0 artefact) ─────────────────────────
zero_mask = np.abs(lags * 1000) < 0.015   # lags in ms
gcc_plot = gcc.copy()
gcc_plot[zero_mask] = np.nan               # NaN creates a visual gap in the plot
gcc[zero_mask] = -np.inf                   # exclude from peak search
best_tau = lags[np.argmax(gcc)]

# ── Plot ───────────────────────────────────────────────────────────────────
BG = '#0d0d1a'
TX = '#ccccdd'

fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor=BG)
fig.suptitle("GCC-PHAT Sound Source Localisation", color='white', fontweight='bold', fontsize=11)

ax.set_facecolor(BG)
ax.tick_params(colors=TX, labelsize=8.5)
for sp in ax.spines.values():
    sp.set_edgecolor('#334')
ax.grid(True, alpha=0.3)

ax.plot(lags * 1000, gcc_plot, color='#66ccff', linewidth=1.2)
ax.axvline(0, color='#334455', linewidth=0.8, linestyle='--')
ax.axvline(best_tau * 1000, color='#ffdd44', linewidth=1.5, linestyle='--',
           label=f'Peak τ = {best_tau*1000:.3f} ms')
ax.set_xlabel('Lag (ms)', color=TX, fontsize=9)
ax.set_ylabel('GCC-PHAT', color=TX, fontsize=9)
ax.set_title('GCC-PHAT Correlation (full clip)', color=TX, fontsize=9.5, pad=4)
ax.set_xlim(-tau_max * 1200, tau_max * 1200)
leg0 = ax.legend(fontsize=8, framealpha=0.20, facecolor='#1a1a2e')
for txt in leg0.get_texts():
    txt.set_color(TX)

print(f"Best lag:          {best_tau * 1000:.4f} ms")
print(f"TDOA resolution:   {1/SAMPLE_RATE * 1000:.4f} ms (1 sample)")
angle_deg = np.degrees(np.arcsin(np.clip(SPEED_OF_SOUND * best_tau / MIC_SPACING, -1.0, 1.0)))
print(f"Angle estimation:  {angle_deg:.4f}°")

plt.tight_layout()
plt.show()