import numpy as np
import pandas as pd
import os
import subprocess
from scipy.io import wavfile
from math import gcd
from scipy.signal import resample_poly

# ═══════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# ── Input ──────────────────────────────────────────────────────────────────
# Single file path, or a list of file paths to batch-process.
CSV_FILES = [
    #r"C:\Users\thijs\Documents\GitHub\PE3\2026-05-22-15-52-50.csv",
    r"C:\Users\thijs\Documents\GitHub\PE3\2026-05-22-15-48-20.csv",
]

MIC_CHANNEL = 0        # 0-based index (0 = Mic 1, 1 = Mic 2, ...)
SAMPLE_RATE = 200000   # [Hz]  original recording sample rate

# ── Output ─────────────────────────────────────────────────────────────────
AUDIO_SR   = 44100   # [Hz]  output sample rate (44100 = CD quality)
OUTPUT_DIR = None    # None = same folder as each CSV; or e.g. r"C:\output"
NORMALIZE  = True    # normalise peak amplitude to -1 dBFS

# ═══════════════════════════════════════════════════════════════════════════


def _get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


def extract_audio(csv_path, mic_channel, sample_rate, audio_sr, out_dir, normalize):
    print(f'\n── {os.path.basename(csv_path)}  (channel {mic_channel}) ──')

    df = pd.read_csv(csv_path)
    signals = df.iloc[:, 1:].to_numpy(dtype=float)   # skip time/index column
    n_samples, n_mics = signals.shape

    if mic_channel >= n_mics:
        raise ValueError(
            f'Channel {mic_channel} requested but CSV only has {n_mics} channels.')

    audio = signals[:, mic_channel].astype(np.float64)
    print(f'  Loaded {n_samples} samples  ({n_samples / sample_rate:.3f} s)'
          f'  from channel {mic_channel} of {n_mics}')

    # ── Resample (polyphase FIR — no aliasing/ringing artifacts) ───────────
    g = gcd(sample_rate, audio_sr)
    audio_rs = resample_poly(audio, audio_sr // g, sample_rate // g)
    print(f'  Resampled {sample_rate} Hz → {audio_sr} Hz  ({len(audio_rs)} samples)')

    # ── Normalise ──────────────────────────────────────────────────────────
    if normalize:
        peak = np.abs(audio_rs).max()
        if peak > 0:
            audio_rs = audio_rs / peak * 0.891   # -1 dBFS headroom

    audio_f32 = audio_rs.astype(np.float32)

    # ── Output paths ───────────────────────────────────────────────────────
    stem    = os.path.splitext(os.path.basename(csv_path))[0]
    folder  = out_dir if out_dir else os.path.dirname(os.path.abspath(csv_path))
    out_mp3 = os.path.join(folder, f'{stem}_ch{mic_channel}.mp3')
    tmp_wav = os.path.join(folder, f'_tmp_{stem}_ch{mic_channel}.wav')

    # ── Write temp WAV then encode to MP3 ──────────────────────────────────
    wavfile.write(tmp_wav, audio_sr, audio_f32)
    result = subprocess.run(
        [_get_ffmpeg(), '-y', '-i', tmp_wav,
         '-codec:a', 'libmp3lame', '-qscale:a', '2',   # VBR ~190 kbps
         out_mp3],
        capture_output=True,
    )
    os.remove(tmp_wav)

    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed:\n{result.stderr.decode()}')

    size_kb = os.path.getsize(out_mp3) / 1024
    print(f'  Saved: {out_mp3}  ({size_kb:.0f} kB)')


if __name__ == '__main__':
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    for csv_path in CSV_FILES:
        extract_audio(
            csv_path    = csv_path,
            mic_channel = MIC_CHANNEL,
            sample_rate = SAMPLE_RATE,
            audio_sr    = AUDIO_SR,
            out_dir     = OUTPUT_DIR,
            normalize   = NORMALIZE,
        )

    print('\nAll done.')
