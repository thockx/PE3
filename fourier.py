import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simple settings
CSV_PATH = r'C:\Users\s4460308\Downloads\PE3-main\2026-05-22-15-42-26.csv'
FS = 200000.0

# Load CSV
df = pd.read_csv(CSV_PATH)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) < 2:
    raise RuntimeError('CSV must contain at least two numeric columns')

# detect time column if present (first numeric col named 'time'/'t'/etc.)
time = None
time_col_name = None
for name in df.columns:
    if str(name).lower() in ('time', 't', 'timestamp', 'ts'):
        time = df[name].to_numpy(dtype=float)
        time_col_name = name
        break

# all numeric columns that are NOT the time column are signal channels
sig_cols = [c for c in num_cols if c != time_col_name]
if len(sig_cols) < 1:
    raise RuntimeError('No signal columns found after excluding time column')
signals = [df[c].to_numpy(dtype=float) for c in sig_cols]

# build time if missing and compute dt
if time is None:
    dt = 1.0 / float(FS)
    time = np.arange(len(signals[0])) * dt
else:
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0 / float(FS)
    if dt <= 0:
        dt = 1.0 / float(FS)

# trim all channels to common length
minN = min(len(time), *[len(s) for s in signals])
if minN <= 0:
    raise RuntimeError('No data available after trimming')
time = time[:minN]
signals = [s[:minN] for s in signals]
N = minN

# compute FFT for all channels (positive frequencies only)
fs = 1.0 / dt if dt > 0 else FS
f_nyquist = fs / 2.0
f_upper = min(100000.0, f_nyquist)

F = np.fft.rfftfreq(N, d=dt)
Xall = [np.fft.rfft(s) for s in signals]

# limit to <= f_upper
mask = F <= f_upper
F = F[mask]
Xall = [X[mask] for X in Xall]

# smoothing
def mov_avg(data, N=10000):
    if N <= 1:
        return data
    return np.convolve(data, np.ones(N)/N, mode='same')

# compute magnitude (dB) and phase for every channel
mags_db  = [20.0 * np.log10(np.abs(X) + 1e-12) for X in Xall]
phases   = [np.angle(X) for X in Xall]
mags_s   = [mov_avg(m) for m in mags_db]
phases_s = [mov_avg(p) for p in phases]

# plotting: magnitude and phase subplots
fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# x-axis: log scale over positive frequencies
pos = F > 0
if pos.any():
    F_plot = F[pos]
    left = max(F_plot.min(), 1e-6)
    ax_mag.set_xscale('log')
    ax_mag.set_xlim(left, max(f_upper, left * 10))
else:
    F_plot = F
    ax_mag.set_xlim(0, f_upper)

fixed_colors = ['black', 'orange', 'red', 'blue']
for idx, col in enumerate(sig_cols):
    c = fixed_colors[idx] if idx < len(fixed_colors) else np.random.rand(3,)
    m_plot = mags_s[idx][pos]  if pos.any() else mags_s[idx]
    p_plot = phases_s[idx][pos] if pos.any() else phases_s[idx]
    ax_mag.plot(F_plot, m_plot, label=col, color=c)
    ax_phase.plot(F_plot, p_plot, label=col, color=c)

ax_mag.set_title(f'{os.path.basename(CSV_PATH)} - Magnitude (dB)')
ax_mag.set_ylabel('Magnitude (dB)')
ax_mag.grid(True, which='both', linestyle=':', linewidth=0.5)
ax_mag.legend()

ax_phase.set_title(f'{os.path.basename(CSV_PATH)} - Phase (rad)')
ax_phase.set_xlabel('Frequency (Hz)')
ax_phase.set_ylabel('Phase (rad)')
ax_phase.grid(True, which='both', linestyle=':', linewidth=0.5)
ax_phase.legend()

plt.tight_layout()
plt.show()
plt.close(fig)