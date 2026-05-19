import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ── Configuration ──────────────────────────────────────────────────────────
CSV_FILE = "data_sweep.csv"    # path to your CSV file
SAMPLE_RATE = 192000           # Sample rate [Hz]
MIC_SPACING = 0.21             # Distance between the two microphones [m]
SPEED_OF_SOUND = 343.0         # Speed of sound [m/s]

# ── Settings ───────────────────────────────────────────────────────────────
smooth_heatmap_amount = 0         # Gaussian smoothing amount for the angle heatmap (0 to disable)
tracking_mode        = True       # Show 2D time-angle heatmap for moving sources
tracking_frame_ms    = 10         # Frame length in ms for the sliding window (tracking_mode only)
tracking_hop_ms      = 5          # Hop size in ms between frames (tracking_mode only)

# ── Load signals ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
x1 = df.iloc[:, 0].to_numpy(dtype=float)
x2 = df.iloc[:, 1].to_numpy(dtype=float)

# ── GCC-PHAT helper ────────────────────────────────────────────────────────
def gcc_phat(s1, s2):
    """Return (gcc, lags_seconds) for a pair of signal segments."""
    n = len(s1) + len(s2) - 1
    nfft = 1 << (n - 1).bit_length()
    F1 = np.fft.rfft(s1, n=nfft)
    F2 = np.fft.rfft(s2, n=nfft)
    G  = F1 * np.conj(F2)
    G  = G / (np.abs(G) + 1e-9)
    g  = np.fft.irfft(G, n=nfft)
    g  = np.concatenate([g[nfft // 2:], g[: nfft // 2]])
    lag_s = np.arange(-nfft // 2, nfft // 2) / SAMPLE_RATE
    return g, lag_s, nfft

def build_heatmap(gcc, lags):
    """Accumulate one GCC output into a 1-D angular heatmap."""
    tau_max = MIC_SPACING / SPEED_OF_SOUND
    valid   = np.abs(lags) <= tau_max
    v_lags  = lags[valid]
    v_gcc   = np.where(gcc[valid] > 0, gcc[valid], 0.0)
    thetas  = np.degrees(np.arcsin(np.clip(v_lags / tau_max, -1.0, 1.0)))
    edges   = np.linspace(-90, 90, len(angles_deg) + 1)
    row, _  = np.histogram(thetas, bins=edges, weights=v_gcc)
    if smooth_heatmap_amount > 0:
        row = gaussian_filter1d(row, sigma=smooth_heatmap_amount)
    return row

# ── Shared angle axis ──────────────────────────────────────────────────────
tau_max    = MIC_SPACING / SPEED_OF_SOUND
angles_deg = np.linspace(-90, 90, 361)

# ── Full-clip GCC-PHAT (always computed) ──────────────────────────────────
gcc, lags, N_fft = gcc_phat(x1, x2)
heatmap = build_heatmap(gcc, lags)
best_tau = lags[np.argmax(gcc)]

# ── Sliding-window tracking (optional) ─────────────────────────────────────
if tracking_mode:
    frame_len = int(SAMPLE_RATE * tracking_frame_ms / 1000)
    hop_len   = int(SAMPLE_RATE * tracking_hop_ms   / 1000)
    starts    = range(0, len(x1) - frame_len + 1, hop_len)
    n_frames  = len(list(starts))
    tracking_map = np.zeros((n_frames, len(angles_deg)))
    frame_times  = []
    for fi, s in enumerate(starts):
        e = s + frame_len
        row = build_heatmap(*gcc_phat(x1[s:e], x2[s:e])[:2])
        tracking_map[fi] = row
        frame_times.append((s + frame_len / 2) / SAMPLE_RATE * 1000)   # ms
    frame_times = np.array(frame_times)

# ── Plot ───────────────────────────────────────────────────────────────────
BG = '#0d0d1a'
TX = '#ccccdd'

n_rows = 2 if tracking_mode else 2
row_heights = [1, 3] if tracking_mode else [2, 1]
fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 + 3 * n_rows),
                         facecolor=BG, gridspec_kw={'height_ratios': row_heights})
fig.suptitle("GCC-PHAT Sound Source Localisation", color='white', fontweight='bold', fontsize=11)

for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TX, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334')
    ax.grid(True, alpha=0.3)

# ── Row 0: GCC-PHAT correlation ────────────────────────────────────────────
axes[0].plot(lags * 1000, gcc, color='#66ccff', linewidth=1.2)
axes[0].axvline(0, color='#334455', linewidth=0.8, linestyle='--')
axes[0].axvline(best_tau * 1000, color='#ffdd44', linewidth=1.5, linestyle='--',
                label=f'Peak τ = {best_tau*1000:.3f} ms')
axes[0].set_xlabel('Lag (ms)', color=TX, fontsize=9)
axes[0].set_ylabel('GCC-PHAT', color=TX, fontsize=9)
axes[0].set_title('GCC-PHAT Correlation (full clip)', color=TX, fontsize=9.5, pad=4)
axes[0].set_xlim(-tau_max * 1200, tau_max * 1200)
leg0 = axes[0].legend(fontsize=8, framealpha=0.20, facecolor='#1a1a2e')
for txt in leg0.get_texts():
    txt.set_color(TX)

# ── Row 1: integrated angular heatmap (1-row) — only shown without tracking
if not tracking_mode:
    peak_angle   = angles_deg[np.argmax(heatmap)]
    heatmap_norm = heatmap / (heatmap.max() + 1e-9)
    im1 = axes[1].imshow(
        heatmap_norm[np.newaxis, :],
        aspect='auto', cmap='magma',
        extent=[-90, 90, 0, 1],
        vmin=0, vmax=1, interpolation='bilinear',
    )
    axes[1].axvline(peak_angle, color='#ffdd44', linewidth=1.5, linestyle='--',
                    label=f'Dominant = {peak_angle:.1f}°')
    axes[1].set_xlabel('Angle (degrees)', color=TX, fontsize=9)
    axes[1].set_yticks([])
    axes[1].set_title('Integrated Angular Heatmap', color=TX, fontsize=9.5, pad=4)
    axes[1].set_xlim(-90, 90)
    axes[1].set_xticks(range(-90, 91, 15))
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', pad=0.01, fraction=0.015)
    cbar1.set_label('Norm. energy', color=TX, fontsize=8)
    cbar1.ax.yaxis.set_tick_params(colors=TX, labelsize=7)
    leg1 = axes[1].legend(fontsize=8, framealpha=0.20, facecolor='#1a1a2e')
    for txt in leg1.get_texts():
        txt.set_color(TX)

# ── Row 1 (tracking) / Row 2 (static): time-angle tracking heatmap ────────
if tracking_mode:
    track_ax = axes[1]
    tm_norm = tracking_map / (tracking_map.max() + 1e-9)
    im2 = track_ax.imshow(
        tm_norm,
        aspect='auto', cmap='magma', origin='upper',
        extent=[-90, 90, frame_times[-1], frame_times[0]],
        vmin=0, vmax=1, interpolation='bilinear',
    )
    track_ax.set_xlabel('Angle (degrees)', color=TX, fontsize=9)
    track_ax.set_ylabel('Time (ms)', color=TX, fontsize=9)
    track_ax.set_title(f'Time-Angle Tracking  (frame {tracking_frame_ms} ms, hop {tracking_hop_ms} ms)',
                       color=TX, fontsize=9.5, pad=4)
    track_ax.set_xlim(-90, 90)
    track_ax.set_xticks(range(-90, 91, 15))
    track_ax.grid(False)
    cbar2 = fig.colorbar(im2, ax=track_ax, orientation='vertical', pad=0.01, fraction=0.015)
    cbar2.set_label('Norm. energy', color=TX, fontsize=8)
    cbar2.ax.yaxis.set_tick_params(colors=TX, labelsize=7)

plt.tight_layout()
plt.savefig('gcc_phat_result.png', dpi=150)
plt.show()

print(f"Dominant angle: {angles_deg[np.argmax(heatmap)]:.1f}°")
print(f"Best lag: {best_tau*1000:.4f} ms")
