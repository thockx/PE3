import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless rendering — no display window needed
import matplotlib.pyplot as plt
from scipy.signal import stft as _scipy_stft, butter as _butter, sosfiltfilt as _sosfiltfilt

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# ── Files ──────────────────────────────────────────────────────────────────
CSV_FILE    = r"C:\Users\thijs\Documents\GitHub\PE3\2026-05-22-15-52-50.csv"
MIC_POS_CSV = r"C:\Users\thijs\Documents\GitHub\PE3\mic_positions.csv"  # columns: x, y [m]

# ── Hardware / Acquisition ─────────────────────────────────────────────────
SAMPLE_RATE        = 200000   # [Hz]
DAQ2_SAMPLE_OFFSET = -10105   # [samples]; 0 = disable

# ── Physics ────────────────────────────────────────────────────────────────
SPEED_OF_SOUND = 343.0        # [m/s]

# ── Acoustic imaging ───────────────────────────────────────────────────────
Z_SCAN            = 0.1       # [m]    initial focal depth (also adjustable via slider)
FOV_DEG           = 90.0      # [°]    full field-of-view cone angle
GRID_RES          = 0.01      # [m]    spatial grid resolution (smaller = sharper, slower)
N_FRAMES          = 50       # number of time frames to split the recording into
FRAME_OVERLAP_PCT = 50        # overlap between consecutive frames (0–99 %)
ANIM_FPS          = 10        # export frame rate [fps]

# ── MUSIC ──────────────────────────────────────────────────────────────────
N_SOURCES   = 1      # assumed number of simultaneous sources
MUSIC_NFFT  = 2**12    # STFT window length [samples]
MUSIC_F_MIN = 200.0  # [Hz]  lower frequency limit for MUSIC integration
MUSIC_F_MAX = 5000.0 # [Hz]  upper frequency limit

# ── Signal conditioning ────────────────────────────────────────────────────
BP_FILTER         = False
BP_LOW_HZ         = 200.0
BP_HIGH_HZ        = 8000.0

# ── Appearance ─────────────────────────────────────────────────────────────
BG = '#0d0d1a'
TX = '#ccccdd'

# ── Output ─────────────────────────────────────────────────────────────────
EXPORT_MP4 = True   # export MUSIC animation + mic-1 audio to MP4

# ═══════════════════════════════════════════════════════════════════════════

# ── Load signals (all mic columns) ─────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
# assume first column is time/index, remaining columns are mic channels
signals = df.iloc[:, 1:].to_numpy(dtype=float)
Nsamples, N_mics = signals.shape
signals_raw = signals.copy()

# ── Apply inter-device timing correction ───────────────────────────────────
# Shift myDAQ2 channels (assumed to be the second half of columns) earlier by
# DAQ2_SAMPLE_OFFSET samples to compensate for software-timed start delay.
if DAQ2_SAMPLE_OFFSET != 0 and N_mics > 2:
    n_dev1 = 2   # number of channels on myDAQ1
    offset = int(DAQ2_SAMPLE_OFFSET)
    for ch in range(n_dev1, N_mics):
        signals[:, ch] = np.roll(signals[:, ch], -offset)
    # zero out the wrap-around samples at the end
    signals[-abs(offset):, n_dev1:] = 0.0
    print(f"Applied DAQ2 timing correction: shifted channels {n_dev1}..{N_mics-1} by -{offset} samples.")


# ── Load microphone positions ──────────────────────────────────────────────
pos_df = pd.read_csv(MIC_POS_CSV, comment='#')
# expect columns 'x' and 'y'
if not set(['x', 'y']).issubset(set(pos_df.columns)):
    raise ValueError("mic_positions.csv must contain columns 'x' and 'y'")
mic_pos = pos_df[['x', 'y']].to_numpy(dtype=float)  # shape (N_mics, 2)
if mic_pos.shape[0] != N_mics:
    raise ValueError(f"Number of mic positions ({mic_pos.shape[0]}) does not match number of channels ({N_mics})")

# ── Bandpass filter (if enabled) ───────────────────────────────────────────
if BP_FILTER:
    _bp_sos = _butter(4, [BP_LOW_HZ, BP_HIGH_HZ],
                      btype='bandpass', fs=SAMPLE_RATE, output='sos')
    print(f'Bandpass filter: {BP_LOW_HZ:.0f}–{BP_HIGH_HZ:.0f} Hz  (4th-order Butterworth, zero-phase)')

# ── Frame parameters ────────────────────────────────────────────────────────
_ai_cx = mic_pos[:, 0].mean()
_ai_cy = mic_pos[:, 1].mean()
_frame_hop     = max(1, Nsamples // N_FRAMES)
_frame_samples = min(Nsamples,
                     int(round(_frame_hop / max(0.01, 1 - FRAME_OVERLAP_PCT / 100))))
_frame_starts  = np.arange(0, Nsamples - _frame_samples + 1, _frame_hop)
_n_frames      = len(_frame_starts)
print(f'Frames: {_n_frames}  |  window = {_frame_samples} samples '
      f'({_frame_samples/SAMPLE_RATE*1000:.1f} ms)  |  hop = {_frame_hop} samples')

# ── MUSIC frame computation ──────────────────────────────────────────────────
def _compute_music_frame(fi, dist_flat, grid_shape, freqs, freq_mask):
    """Incoherent wideband MUSIC pseudospectrum for one time frame.

    dist_flat : (N_mics, Ny*Nx)  precomputed distances to every scan point
    Returns   : (Ny, Nx) MUSIC power map
    """
    s = int(_frame_starts[fi])
    sigs_f = signals[s:s + _frame_samples].copy()
    if BP_FILTER:
        sigs_f = _sosfiltfilt(_bp_sos, sigs_f, axis=0)

    noverlap = MUSIC_NFFT * 3 // 4
    Xall = []
    for m in range(N_mics):
        _, _, Zxx = _scipy_stft(sigs_f[:, m], fs=SAMPLE_RATE,
                                nperseg=MUSIC_NFFT, noverlap=noverlap)
        Xall.append(Zxx)
    X = np.stack(Xall, axis=0)   # (N_mics, N_bins, N_t)

    N_grid = dist_flat.shape[1]
    music_map = np.zeros(N_grid)

    for k in np.where(freq_mask)[0]:
        f_k = freqs[k]
        Xk  = X[:, k, :]                             # (N_mics, N_t)
        R   = (Xk @ Xk.conj().T) / Xk.shape[1]
        R  += 1e-6 * np.eye(N_mics)                  # diagonal loading
        _, v = np.linalg.eigh(R)                     # ascending eigenvalues
        Vn = v[:, :-N_SOURCES]                       # noise subspace
        En = Vn @ Vn.conj().T
        a  = np.exp(-1j * 2 * np.pi * f_k * dist_flat / SPEED_OF_SOUND)
        En_a  = En @ a
        denom = np.real(np.sum(a.conj() * En_a, axis=0))
        music_map += 1.0 / np.maximum(denom, 1e-10)

    return music_map.reshape(grid_shape)

# ── Precompute grid + distances, run all frames ──────────────────────────────
_st = {'z': Z_SCAN, 'frames': [], 'adeg': None, 'edeg': None,
       'vmin': 0.0, 'vmax': 1.0}

def _precompute(z):
    hs = z * np.tan(np.radians(FOV_DEG / 2))
    sx = np.arange(-hs, hs + GRID_RES, GRID_RES)
    sy = np.arange(-hs, hs + GRID_RES, GRID_RES)
    XX, YY = np.meshgrid(sx, sy)
    PX = _ai_cx + XX
    PY = _ai_cy + YY
    Ny, Nx = XX.shape

    dist_flat = np.stack([
        np.sqrt((PX - mic_pos[m, 0])**2 + (PY - mic_pos[m, 1])**2 + z**2).ravel()
        for m in range(N_mics)
    ])

    freqs     = np.fft.rfftfreq(MUSIC_NFFT, 1.0 / SAMPLE_RATE)
    freq_mask = (freqs >= MUSIC_F_MIN) & (freqs <= MUSIC_F_MAX)
    adeg = np.degrees(np.arctan2(sx, z))
    edeg = np.degrees(np.arctan2(sy, z))

    n_bins = freq_mask.sum()
    print(f'MUSIC: {_n_frames} frames at z={z:.2f} m  '
          f'| {n_bins} bins  {MUSIC_F_MIN:.0f}–{MUSIC_F_MAX:.0f} Hz '
          f'| grid {Ny}×{Nx} …')
    frames = []
    for fi in range(_n_frames):
        frames.append(_compute_music_frame(fi, dist_flat, (Ny, Nx), freqs, freq_mask))
        print(f'  {fi + 1}/{_n_frames}', end='\r')
    print('\nDone.')

    _gmin = min(f.min() for f in frames)
    _gmax = max(f.max() for f in frames)
    _st.update(z=z, frames=frames, adeg=adeg, edeg=edeg,
               vmin=_gmin, vmax=_gmax)

_precompute(Z_SCAN)

# ── Figure for frame rendering (Agg — no interactive window) ─────────────────
fig_ai, ax_ai = plt.subplots(figsize=(8, 8), facecolor=BG)
ax_ai.set_facecolor(BG)
im_ai = ax_ai.imshow(
    _st['frames'][0],
    extent=[_st['adeg'][0], _st['adeg'][-1], _st['edeg'][0], _st['edeg'][-1]],
    origin='lower', aspect='equal', cmap='inferno', interpolation='bilinear',
    vmin=_st['vmin'], vmax=_st['vmax'],
)
cbar_ai = fig_ai.colorbar(im_ai, ax=ax_ai)
cbar_ai.set_label('MUSIC Power', color=TX)
cbar_ai.ax.yaxis.set_tick_params(color=TX)
plt.setp(cbar_ai.ax.yaxis.get_ticklabels(), color=TX)
ttl_ai = ax_ai.set_title('', color=TX, fontweight='bold')
ax_ai.set_xlabel('Azimuth (°)', color=TX)
ax_ai.set_ylabel('Elevation (°)', color=TX)
ax_ai.axhline(0, color='#334455', linewidth=0.8, linestyle='--')
ax_ai.axvline(0, color='#334455', linewidth=0.8, linestyle='--')
ax_ai.tick_params(colors=TX)
for sp in ax_ai.spines.values():
    sp.set_edgecolor('#334')
plt.tight_layout()

def _show_frame(fi):
    im_ai.set_data(_st['frames'][fi])
    im_ai.set_clim(_st['vmin'], _st['vmax'])
    t_ms = float(_frame_starts[fi]) / SAMPLE_RATE * 1000
    ttl_ai.set_text(f't = {t_ms:.1f} ms  |  frame {fi + 1}/{_n_frames}'
                    f'  |  z = {_st["z"]:.2f} m')
    fig_ai.canvas.draw()

# ── MP4 export ────────────────────────────────────────────────────────────────
if EXPORT_MP4:
    import os, subprocess
    import imageio
    import imageio_ffmpeg
    from scipy.io import wavfile
    from scipy.signal import resample as _resample

    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _out_stem = os.path.splitext(os.path.basename(CSV_FILE))[0]
    _out_dir  = os.path.dirname(os.path.abspath(CSV_FILE))
    _out_mp4  = os.path.join(_out_dir, _out_stem + '_music.mp4')
    _tmp_vid  = os.path.join(_out_dir, '_tmp_video.mp4')
    _tmp_wav  = os.path.join(_out_dir, '_tmp_audio.wav')

    _rec_duration = Nsamples / SAMPLE_RATE
    _export_fps   = _n_frames / _rec_duration
    print(f'Exporting {_n_frames} frames at {_export_fps:.2f} fps '
          f'(recording = {_rec_duration:.3f} s) → {_out_mp4}')
    _vid_writer = imageio.get_writer(
        _tmp_vid, fps=_export_fps, codec='libx264',
        output_params=['-pix_fmt', 'yuv420p'],
    )
    for _fi in range(_n_frames):
        _show_frame(_fi)
        _frame_img = np.asarray(fig_ai.canvas.buffer_rgba())[:, :, :3]
        _vid_writer.append_data(_frame_img)
        print(f'  frame {_fi + 1}/{_n_frames}', end='\r')
    _vid_writer.close()
    print()

    print('Saving mic-1 audio …')
    _audio_sr = 48000
    _mic1 = signals[:, 0].astype(np.float64)
    _peak = np.abs(_mic1).max()
    if _peak > 0:
        _mic1 /= _peak
    _n_audio_out = int(round(len(_mic1) * _audio_sr / SAMPLE_RATE))
    _mic1_rs = _resample(_mic1, _n_audio_out).astype(np.float32)
    wavfile.write(_tmp_wav, _audio_sr, _mic1_rs)

    print('Muxing video + audio …')
    subprocess.run(
        [_ffmpeg_exe, '-y',
         '-i', _tmp_vid,
         '-i', _tmp_wav,
         '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
         '-shortest', _out_mp4],
        check=True,
    )
    os.remove(_tmp_vid)
    os.remove(_tmp_wav)
    print(f'Saved: {_out_mp4}')

plt.close(fig_ai)
