import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# ── Files ──────────────────────────────────────────────────────────────────
CSV_FILE    = r"C:\Users\thijs\Documents\GitHub\PE3\2026-05-22-15-48-20.csv"
MIC_POS_CSV = r"C:\Users\thijs\Documents\GitHub\PE3\mic_positions.csv"  # columns: x, y [m]

# ── Hardware / Acquisition ─────────────────────────────────────────────────
SAMPLE_RATE        = 200000   # [Hz]       recording sample rate
DAQ2_SAMPLE_OFFSET = 0 #-10105   # [samples]  shift myDAQ2 channels to correct timing; 0 = disable
OFFSET_SWEEP_MIN   = -77000-2000   # [samples]  start of DAQ2 offset search range
OFFSET_SWEEP_MAX   = -77000-1000   # [samples]  end of DAQ2 offset search range
OFFSET_SWEEP_N     = 100   # number of sweep points (spread evenly over min…max)
# ── Physics ────────────────────────────────────────────────────────────────
SPEED_OF_SOUND = 343.0        # [m/s]

# ── Acoustic imaging ───────────────────────────────────────────────────────
Z_SCAN            = 1.0       # [m]    initial focal depth (also adjustable via slider)
FOV_DEG           = 90.0      # [°]    full field-of-view cone angle
GRID_RES          = 0.01      # [m]    spatial grid resolution (smaller = sharper, slower)
N_FRAMES          = 50       # number of time frames to split the recording into
FRAME_OVERLAP_PCT = 50        # overlap between consecutive frames (0–99 %)
ANIM_FPS          = 10        # interactive animation playback speed [fps]

# ── Signal conditioning ────────────────────────────────────────────────────
BP_FILTER         = False      # apply bandpass filter before GCC-PHAT
BP_LOW_HZ         = 200.0     # [Hz]  bandpass lower cutoff
BP_HIGH_HZ        = 6000.0    # [Hz]  bandpass upper cutoff

# ── Appearance ─────────────────────────────────────────────────────────────
BG = '#0d0d1a'                # figure background colour
TX = '#ccccdd'                # text and tick colour

# ── Output ─────────────────────────────────────────────────────────────────
PLOT_RAW_SIGNALS    = False  # show raw microphone waveforms
PLOT_OFFSET_SWEEP   = False  # animate SRP-PHAT heatmap swept over DAQ2 offset values
PLOT_GCC_PHAT       = False  # show GCC-PHAT correlation panels
PLOT_ACOUSTIC_IMAGE = False  # show SRP-PHAT front-view acoustic image
EXPORT_MP4          = False  # export animation + mic-1 audio to MP4

# ═══════════════════════════════════════════════════════════════════════════

# ── Load signals (all mic columns) ─────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
# assume first column is time/index, remaining columns are mic channels
signals = df.iloc[:, 1:].to_numpy(dtype=float)
Nsamples, N_mics = signals.shape
signals_raw = signals.copy()   # unshifted copy used by the offset-sweep animation

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

# quick plot of first four channels (if available)
if PLOT_RAW_SIGNALS:
    from matplotlib.widgets import Slider

    _plot_colors = ['black', 'orange', 'red', 'blue']

    fig_raw, ax_raw = plt.subplots(figsize=(13, 5))
    fig_raw.subplots_adjust(bottom=0.18)
    ax_raw.set_xlabel('Sample index')
    ax_raw.set_ylabel('Amplitude')

    _raw_lines = []
    for i in range(min(4, N_mics)):
        c = _plot_colors[i] if i < len(_plot_colors) else np.random.rand(3,)
        ln, = ax_raw.plot(signals_raw[:, i], label=f'Mic {i+1}',
                          color=c, linewidth=0.7, alpha=0.9)
        _raw_lines.append(ln)

    _raw_title = ax_raw.set_title(
        f'Raw signals  —  DAQ2 offset = {DAQ2_SAMPLE_OFFSET:+d} samples  '
        f'(drag slider to adjust)')

    # ── Slider ─────────────────────────────────────────────────────────────
    ax_sl = fig_raw.add_axes([0.15, 0.05, 0.70, 0.04])
    sl_offset = Slider(ax_sl, 'DAQ2 offset', -15000, 15000,
                       valinit=DAQ2_SAMPLE_OFFSET, valstep=1)

    def _update_raw(val):
        o = int(round(sl_offset.val))
        for ch in range(len(_raw_lines)):
            if ch < 2:
                _raw_lines[ch].set_ydata(signals_raw[:, ch])
            else:
                if o != 0:
                    shifted = np.roll(signals_raw[:, ch], -o)
                    shifted[-abs(o):] = 0.0
                else:
                    shifted = signals_raw[:, ch].copy()
                _raw_lines[ch].set_ydata(shifted)
        _raw_title.set_text(
            f'Raw signals  —  DAQ2 offset = {o:+d} samples  '
            f'(drag slider to adjust)')
        fig_raw.canvas.draw_idle()

    sl_offset.on_changed(_update_raw)
    plt.show()

# ── Load microphone positions ──────────────────────────────────────────────
pos_df = pd.read_csv(MIC_POS_CSV, comment='#')
# expect columns 'x' and 'y'
if not set(['x', 'y']).issubset(set(pos_df.columns)):
    raise ValueError("mic_positions.csv must contain columns 'x' and 'y'")
mic_pos = pos_df[['x', 'y']].to_numpy(dtype=float)  # shape (N_mics, 2)
if mic_pos.shape[0] != N_mics:
    raise ValueError(f"Number of mic positions ({mic_pos.shape[0]}) does not match number of channels ({N_mics})")

# ── Moving average helper (small window stabiliser) ────────────────────────
def mov_avg(data, N=101):
    if N <= 1:
        return data
    return np.convolve(data, np.ones(N)/N, mode='same')

# ── GCC-PHAT helper for a pair of signals ──────────────────────────────────
def gcc_phat_pair(x, y, nfft=None):
    n = len(x) + len(y) - 1
    if nfft is None:
        nfft = 1 << (n - 1).bit_length()
    X = np.fft.fft(x, n=nfft)
    Y = np.fft.fft(y, n=nfft)
    X_mag = mov_avg(np.abs(X))
    Y_mag = mov_avg(np.abs(Y))
    G = X * np.conj(Y)
    G /= (X_mag * Y_mag) + 1e-9
    cc = np.real(np.fft.ifft(G))
    cc = np.concatenate([cc[nfft // 2:], cc[: nfft // 2]])
    lags = np.arange(-nfft // 2, nfft // 2) / SAMPLE_RATE
    return cc, lags, nfft

# choose a common nfft for all pairs to make lags consistent
_common_nfft = 1 << ((2 * Nsamples - 1).bit_length())

# ── Precompute GCC-PHAT for every microphone pair ──────────────────────────
pair_cc = {}   # (i,j) -> cc array
pair_lags = None
for i, j in itertools.combinations(range(N_mics), 2):
    x = signals[:, i]
    y = signals[:, j]
    cc, lags, nfft = gcc_phat_pair(x, y, nfft=_common_nfft)
    pair_cc[(i, j)] = cc
    pair_lags = lags
# ── DAQ2 offset sweep: animated SRP-PHAT heatmap ──────────────────────────────
if PLOT_OFFSET_SWEEP:
    from scipy.interpolate import interp1d as _interp1d_os
    from matplotlib.widgets import Slider as _Slider_os
    import matplotlib.animation as _manim_os

    _os_n_dev1   = 2   # channels on myDAQ1 (not shifted)
    _os_offsets  = np.linspace(OFFSET_SWEEP_MIN, OFFSET_SWEEP_MAX, OFFSET_SWEEP_N).astype(int)
    _os_n        = len(_os_offsets)
    _os_nfft     = 1 << ((2 * Nsamples - 1).bit_length())
    _os_cx       = mic_pos[:, 0].mean()
    _os_cy       = mic_pos[:, 1].mean()
    _os_hs       = Z_SCAN * np.tan(np.radians(FOV_DEG / 2))
    _os_sx       = np.arange(-_os_hs, _os_hs + GRID_RES, GRID_RES)
    _os_sy       = np.arange(-_os_hs, _os_hs + GRID_RES, GRID_RES)
    _os_XX, _os_YY = np.meshgrid(_os_sx, _os_sy)
    _os_PX       = _os_cx + _os_XX
    _os_PY       = _os_cy + _os_YY
    _os_adeg     = np.degrees(np.arctan2(_os_sx, Z_SCAN))
    _os_edeg     = np.degrees(np.arctan2(_os_sy, Z_SCAN))

    # precompute distances (fixed, independent of offset)
    _os_dist = {}
    for _i, _j in itertools.combinations(range(N_mics), 2):
        di = np.sqrt((_os_PX - mic_pos[_i, 0])**2 + (_os_PY - mic_pos[_i, 1])**2 + Z_SCAN**2)
        dj = np.sqrt((_os_PX - mic_pos[_j, 0])**2 + (_os_PY - mic_pos[_j, 1])**2 + Z_SCAN**2)
        _os_dist[(_i, _j)] = (di, dj)

    def _os_compute_srp(offset_val):
        sigs = signals_raw.copy()
        if offset_val != 0:
            o = int(offset_val)
            for ch in range(_os_n_dev1, N_mics):
                sigs[:, ch] = np.roll(sigs[:, ch], -o)
                sigs[-abs(o):, ch] = 0.0
        srp = np.zeros_like(_os_XX)
        for _i, _j in itertools.combinations(range(N_mics), 2):
            cc, lags, _ = gcc_phat_pair(sigs[:, _i], sigs[:, _j], nfft=_os_nfft)
            ifn = _interp1d_os(lags, cc, bounds_error=False, fill_value=0.0)
            di, dj = _os_dist[(_i, _j)]
            srp += ifn((di - dj) / SPEED_OF_SOUND)
        return srp

    print(f'Offset sweep: computing {_os_n} SRP images '
          f'({OFFSET_SWEEP_MIN:+d} … {OFFSET_SWEEP_MAX:+d}, {OFFSET_SWEEP_N} points) …')
    _os_frames = []
    for _k, _ov in enumerate(_os_offsets):
        _os_frames.append(_os_compute_srp(_ov))
        print(f'  {_k + 1}/{_os_n}  offset = {_ov:+d}', end='\r')
    print('\nDone.')

    _os_vmin = min(f.min() for f in _os_frames)
    _os_vmax = max(f.max() for f in _os_frames)

    _os_fig, _os_ax = plt.subplots(figsize=(8, 8), facecolor=BG)
    _os_fig.subplots_adjust(bottom=0.15)
    _os_ax.set_facecolor(BG)
    _os_im = _os_ax.imshow(
        _os_frames[0],
        extent=[_os_adeg[0], _os_adeg[-1], _os_edeg[0], _os_edeg[-1]],
        origin='lower', aspect='equal', cmap='inferno', interpolation='bilinear',
        vmin=_os_vmin, vmax=_os_vmax,
    )
    _os_cbar = _os_fig.colorbar(_os_im, ax=_os_ax)
    _os_cbar.set_label('SRP-PHAT Power', color=TX)
    _os_cbar.ax.yaxis.set_tick_params(color=TX)
    plt.setp(_os_cbar.ax.yaxis.get_ticklabels(), color=TX)
    _os_ttl = _os_ax.set_title(
        f'DAQ2 offset sweep  —  offset = {_os_offsets[0]:+d} samples',
        color=TX, fontweight='bold')
    _os_ax.set_xlabel('Azimuth (°)', color=TX)
    _os_ax.set_ylabel('Elevation (°)', color=TX)
    _os_ax.axhline(0, color='#334455', linewidth=0.8, linestyle='--')
    _os_ax.axvline(0, color='#334455', linewidth=0.8, linestyle='--')
    _os_ax.tick_params(colors=TX)
    for _sp in _os_ax.spines.values():
        _sp.set_edgecolor('#334')

    _os_ax_sl = _os_fig.add_axes([0.15, 0.05, 0.70, 0.04], facecolor='#1a1a2e')
    _os_sl = _Slider_os(_os_ax_sl, 'DAQ2 offset', _os_offsets[0], _os_offsets[-1],
                        valinit=_os_offsets[0], valstep=1, color='#3366cc')
    _os_sl.label.set_color(TX)
    _os_sl.valtext.set_color(TX)
    _os_busy = {'v': False}

    def _os_on_slider(val):
        if _os_busy['v']:
            return
        idx = int(np.argmin(np.abs(_os_offsets - val)))
        _os_im.set_data(_os_frames[idx])
        _os_ttl.set_text(f'DAQ2 offset sweep  —  offset = {_os_offsets[idx]:+d} samples')
        _os_fig.canvas.draw_idle()

    def _os_anim_update(fi):
        _os_busy['v'] = True
        _os_sl.set_val(_os_offsets[fi])
        _os_busy['v'] = False
        _os_im.set_data(_os_frames[fi])
        _os_ttl.set_text(f'DAQ2 offset sweep  —  offset = {_os_offsets[fi]:+d} samples')
        _os_fig.canvas.draw_idle()
        return _os_im, _os_ttl

    _os_sl.on_changed(_os_on_slider)
    _os_ani = _manim_os.FuncAnimation(
        _os_fig, _os_anim_update, frames=_os_n,
        interval=int(1000 / ANIM_FPS), blit=False, repeat=True,
    )
    plt.show()
# ── SRP-PHAT 2D scanning grid ──────────────────────────────────────────────

# Plot GCC-PHAT correlations for all mic pairs
if PLOT_GCC_PHAT:
    plot_pairs = list(pair_cc.keys())[:6]
    n_plots = len(plot_pairs)
    ncols = 2
    nrows = (n_plots + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), facecolor=BG)
    fig.suptitle('GCC-PHAT Correlations', color='white', fontweight='bold', fontsize=12)
    axes = np.array(axes).flatten()

    for ax_idx, (i, j) in enumerate(plot_pairs):
        ax = axes[ax_idx]
        ax.set_facecolor(BG)
        ax.tick_params(colors=TX, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#334')
        ax.grid(True, alpha=0.3)

        cc = pair_cc[(i, j)].copy()
        lags_ms = pair_lags * 1000  # convert to ms

        # suppress tau=0 artefact
        zero_mask = np.abs(lags_ms) < 0.015
        cc_plot = cc.copy()
        cc_plot[zero_mask] = np.nan
        cc[zero_mask] = -np.inf

        tau_max_pair = np.linalg.norm(mic_pos[i] - mic_pos[j]) / SPEED_OF_SOUND * 1000

        # clamp peak search to physically possible delays only
        physical_mask = np.abs(lags_ms) <= tau_max_pair
        cc_physical = cc.copy()
        cc_physical[~physical_mask] = -np.inf
        best_tau_ms = lags_ms[np.argmax(cc_physical)]
        best_tau_s = best_tau_ms / 1000.0

        # estimate angle from TDOA and pair baseline (projected onto x-axis)
        baseline = mic_pos[j] - mic_pos[i]
        baseline_len = np.linalg.norm(baseline)
        sin_arg = np.clip(SPEED_OF_SOUND * best_tau_s / baseline_len, -1.0, 1.0)
        best_angle = np.degrees(np.arcsin(sin_arg))

        ax.plot(lags_ms, cc_plot, color='#66ccff', linewidth=1.2)
        ax.axvline(0, color='#334455', linewidth=0.8, linestyle='--')
        ax.axvline(best_tau_ms, color='#ffdd44', linewidth=1.5, linestyle='--',
                   label=f'Peak τ = {best_tau_ms:.3f} ms  ({best_angle:.1f}°)')
        ax.set_xlim(-tau_max_pair * 1.2, tau_max_pair * 1.2)
        ax.set_xlabel('Lag (ms)', color=TX, fontsize=9)
        ax.set_ylabel('GCC-PHAT', color=TX, fontsize=9)
        ax.set_title(f'Mic {i+1} vs Mic {j+1}', color=TX, fontsize=10)
        leg = ax.legend(fontsize=8, framealpha=0.2, facecolor='#1a1a2e')
        for txt in leg.get_texts():
            txt.set_color(TX)
        print(f'Mic {i+1} vs Mic {j+1}: best lag = {best_tau_ms:.4f} ms  →  angle = {best_angle:.2f}°')

    # hide any unused subplots
    for ax_idx in range(n_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    plt.show()

# ── SRP-PHAT Acoustic Image (front-view) ─────────────────────────────────
# The mic array sits in the XY-plane at z=0 (vertical panel, like a camera
# sensor).  We scan a 2-D grid at depth Z_SCAN in front of the array.
# For each candidate point P=(Px,Py,Z_SCAN) and pair (i,j):
#   τ_ij(P) = (‖P−m_i‖ − ‖P−m_j‖) / c   (exact 3-D TDOA)
if PLOT_ACOUSTIC_IMAGE or EXPORT_MP4:
    from scipy.interpolate import interp1d
    from scipy.signal import butter as _butter, sosfiltfilt as _sosfiltfilt
    from matplotlib.widgets import Slider as _Slider
    import matplotlib.animation as _manim

    if BP_FILTER:
        _bp_sos = _butter(4, [BP_LOW_HZ, BP_HIGH_HZ],
                          btype='bandpass', fs=SAMPLE_RATE, output='sos')
        print(f'Bandpass filter: {BP_LOW_HZ:.0f}–{BP_HIGH_HZ:.0f} Hz  (4th-order Butterworth, zero-phase)')

    _ai_cx = mic_pos[:, 0].mean()
    _ai_cy = mic_pos[:, 1].mean()
    _frame_hop     = max(1, Nsamples // N_FRAMES)
    _frame_samples = min(Nsamples,
                         int(round(_frame_hop / max(0.01, 1 - FRAME_OVERLAP_PCT / 100))))
    _frame_nfft    = 1 << ((2 * _frame_samples - 1).bit_length())
    _frame_starts  = np.arange(0, Nsamples - _frame_samples + 1, _frame_hop)
    _n_frames      = len(_frame_starts)
    print(f'Frames: {_n_frames}  |  window = {_frame_samples} samples  '
          f'({_frame_samples/SAMPLE_RATE*1000:.1f} ms)  |  hop = {_frame_hop} samples')

    def _compute_srp_frame(fi, z):
        s = int(_frame_starts[fi])
        sigs_f = signals[s:s + _frame_samples]   # offset-corrected signals
        if BP_FILTER:
            sigs_f = _sosfiltfilt(_bp_sos, sigs_f, axis=0)
        hs = z * np.tan(np.radians(FOV_DEG / 2))
        sx = np.arange(-hs, hs + GRID_RES, GRID_RES)
        sy = np.arange(-hs, hs + GRID_RES, GRID_RES)
        XX, YY = np.meshgrid(sx, sy)
        PX = _ai_cx + XX
        PY = _ai_cy + YY
        srp = np.zeros_like(XX)
        for i, j in itertools.combinations(range(N_mics), 2):
            cc, lags, _ = gcc_phat_pair(sigs_f[:, i], sigs_f[:, j],
                                        nfft=_frame_nfft)
            ifn = interp1d(lags, cc, bounds_error=False, fill_value=0.0)
            d_i = np.sqrt((PX - mic_pos[i, 0])**2 + (PY - mic_pos[i, 1])**2 + z**2)
            d_j = np.sqrt((PX - mic_pos[j, 0])**2 + (PY - mic_pos[j, 1])**2 + z**2)
            srp += ifn((d_i - d_j) / SPEED_OF_SOUND)
        adeg = np.degrees(np.arctan2(sx, z))
        edeg = np.degrees(np.arctan2(sy, z))
        return srp, adeg, edeg

    # ── State (mutable container avoids nonlocal) ──────────────────────────
    _st = {'z': Z_SCAN, 'frames': [], 'adeg': None, 'edeg': None,
           'vmin': 0.0, 'vmax': 1.0, 'busy': False}

    def _precompute(z):
        print(f'Computing {_n_frames} frames at z = {z:.2f} m …')
        frames = []
        adeg = edeg = None
        for fi in range(_n_frames):
            srp, ad, ed = _compute_srp_frame(fi, z)
            frames.append(srp)
            if fi == 0:
                adeg, edeg = ad, ed
            print(f'  {fi + 1}/{_n_frames}', end='\r')
        print('\nDone.')
        _gmin = min(f.min() for f in frames)
        _gmax = max(f.max() for f in frames)
        _st.update(z=z, frames=frames, adeg=adeg, edeg=edeg,
                   vmin=_gmin, vmax=_gmax)

    _precompute(Z_SCAN)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig_ai, ax_ai = plt.subplots(figsize=(8, 8), facecolor=BG)
    fig_ai.subplots_adjust(bottom=0.20)
    ax_ai.set_facecolor(BG)

    _srp0 = _st['frames'][0]
    im_ai = ax_ai.imshow(
        _srp0,
        extent=[_st['adeg'][0], _st['adeg'][-1], _st['edeg'][0], _st['edeg'][-1]],
        origin='lower', aspect='equal', cmap='inferno', interpolation='bilinear',
        vmin=_st['vmin'], vmax=_st['vmax'],
    )
    cbar_ai = fig_ai.colorbar(im_ai, ax=ax_ai)
    cbar_ai.set_label('SRP-PHAT Power', color=TX)
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

    # ── Sliders ────────────────────────────────────────────────────────────
    ax_tsl = fig_ai.add_axes([0.15, 0.11, 0.70, 0.03], facecolor='#1a1a2e')
    sl_t = _Slider(ax_tsl, 'Frame', 0, _n_frames - 1,
                   valinit=0, valstep=1, color='#336633')
    sl_t.label.set_color(TX); sl_t.valtext.set_color(TX)

    ax_zsl = fig_ai.add_axes([0.15, 0.05, 0.70, 0.03], facecolor='#1a1a2e')
    sl_z = _Slider(ax_zsl, 'Focal depth (m)', 0.1, 5.0,
                   valinit=Z_SCAN, valstep=0.05, color='#3366cc')
    sl_z.label.set_color(TX); sl_z.valtext.set_color(TX)

    def _show_frame(fi):
        srp = _st['frames'][fi]
        im_ai.set_data(srp)
        im_ai.set_clim(_st['vmin'], _st['vmax'])
        t_ms = float(_frame_starts[fi]) / SAMPLE_RATE * 1000
        ttl_ai.set_text(f't = {t_ms:.1f} ms  |  frame {fi + 1}/{_n_frames}'
                        f'  |  z = {_st["z"]:.2f} m')
        fig_ai.canvas.draw_idle()

    def _on_time_slider(val):
        if not _st['busy']:
            _show_frame(int(round(sl_t.val)))

    def _on_z_slider(val):
        _precompute(float(sl_z.val))
        _show_frame(int(round(sl_t.val)))

    sl_t.on_changed(_on_time_slider)
    sl_z.on_changed(_on_z_slider)

    def _anim_update(fi):
        _st['busy'] = True
        sl_t.set_val(fi)
        _st['busy'] = False
        _show_frame(fi)
        return im_ai, ttl_ai

    _ani_ai = _manim.FuncAnimation(
        fig_ai, _anim_update, frames=_n_frames,
        interval=int(1000 / ANIM_FPS), blit=False, repeat=True,
    )

    plt.tight_layout()

    # ── MP4 export ──────────────────────────────────────────────────────────
    if EXPORT_MP4:
        import os, subprocess
        import imageio
        import imageio_ffmpeg
        from scipy.io import wavfile
        from scipy.signal import resample as _resample

        _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        _out_stem = os.path.splitext(os.path.basename(CSV_FILE))[0]
        _out_dir  = os.path.dirname(os.path.abspath(CSV_FILE))
        _out_mp4  = os.path.join(_out_dir, _out_stem + '_acoustic.mp4')
        _tmp_vid  = os.path.join(_out_dir, '_tmp_video.mp4')
        _tmp_wav  = os.path.join(_out_dir, '_tmp_audio.wav')

        # fps that makes video duration == recording duration
        _rec_duration = Nsamples / SAMPLE_RATE          # seconds of actual audio
        _export_fps   = _n_frames / _rec_duration       # frames-per-second for sync
        print(f'Exporting {_n_frames} frames at {_export_fps:.2f} fps '
              f'(recording = {_rec_duration:.3f} s) → {_out_mp4}')
        _vid_writer = imageio.get_writer(
            _tmp_vid, fps=_export_fps, codec='libx264',
            output_params=['-pix_fmt', 'yuv420p'],
        )
        for _fi in range(_n_frames):
            _show_frame(_fi)
            fig_ai.canvas.draw()
            _frame = np.asarray(fig_ai.canvas.buffer_rgba())[:, :, :3]
            _vid_writer.append_data(_frame)
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

    if PLOT_ACOUSTIC_IMAGE:
        plt.show()


