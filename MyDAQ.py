import nidaqmx as dx
import nidaqmx.constants
from nidaqmx.constants import AcquisitionType, Edge
import numpy as np
from datetime import datetime
import os

# ── Configuration ──────────────────────────────────────────────────────────
device_name  = 'myDAQ1'
device_name2 = 'myDAQ2'
NUM_DAQS = 2
input_channels = ['ai0', 'ai1']
rate = 200000
duration = 5.0
filename = None
OUTPUT_ONLY = False

AO_WAVEFORM = 'noise'
AO_FREQ = 40000.0
AO_AMP  = 10.0
AO_DC   = 0.0

# ── Wiring required for NUM_DAQS=2 ────────────────────────────────────────
#   myDAQ1 DIO0 → myDAQ2 DIO0   (start trigger)
#   myDAQ1 DIO1 → myDAQ2 DIO1   (sample clock)
#   myDAQ1 DGND → myDAQ2 DGND   (common ground)

n_samples = int(rate * duration)
print(f"n_samples={n_samples}, rate={rate}, duration={duration}")

# ── Generate AO waveform ───────────────────────────────────────────────────
t = np.arange(n_samples) / rate
if AO_WAVEFORM == 'noise':
    waveform = np.random.uniform(-10.0, 10.0, size=n_samples)
elif AO_WAVEFORM == 'sine':
    waveform = AO_DC + AO_AMP * np.sin(2.0 * np.pi * AO_FREQ * t)
elif AO_WAVEFORM == 'square':
    waveform = AO_DC + AO_AMP * np.where(np.sin(2.0 * np.pi * AO_FREQ * t) >= 0, 1.0, -1.0)
else:
    raise ValueError(f'Unknown AO_WAVEFORM: {AO_WAVEFORM!r}')
waveform = np.clip(waveform, -10.0, 10.0)

timeout = max(10.0, duration + 5.0)

if OUTPUT_ONLY:
    with dx.Task() as ao_task:
        ao_task.ao_channels.add_ao_voltage_chan(f'{device_name}/ao0', min_val=-10.0, max_val=10.0)
        ao_task.timing.cfg_samp_clk_timing(rate,
            sample_mode=AcquisitionType.FINITE, samps_per_chan=n_samples)
        ao_task.write(waveform.tolist(), auto_start=False)
        ao_task.start()
        ao_task.wait_until_done(timeout + 1.0)
else:
    with dx.Task() as ao_task, dx.Task() as ai_task1, dx.Task() as ai_task2:

        # ── AO setup ──────────────────────────────────────────────────────
        ao_task.ao_channels.add_ao_voltage_chan(f'{device_name}/ao0',
            min_val=-10.0, max_val=10.0)
        ao_task.timing.cfg_samp_clk_timing(rate,
            sample_mode=AcquisitionType.FINITE, samps_per_chan=n_samples)
        ao_task.write(waveform.tolist(), auto_start=False)

        # ── AI setup: separate task per device ────────────────────────────
        for ch in input_channels:
            ai_task1.ai_channels.add_ai_voltage_chan(f'{device_name}/{ch}')
        ai_task1.timing.cfg_samp_clk_timing(rate,
            sample_mode=AcquisitionType.FINITE, samps_per_chan=n_samples)

        if NUM_DAQS == 2:
            for ch in input_channels:
                ai_task2.ai_channels.add_ai_voltage_chan(f'{device_name2}/{ch}')
            ai_task2.timing.cfg_samp_clk_timing(rate,
                sample_mode=AcquisitionType.FINITE, samps_per_chan=n_samples)
            print('Note: USB MyDAQ uses software-timed sync between devices (small start jitter expected).')

        # ── Start ─────────────────────────────────────────────────────────
        ao_task.start()
        ai_task1.start()
        if NUM_DAQS == 2:
            ai_task2.start()

        print('Reading...')
        data1 = ai_task1.read(number_of_samples_per_channel=n_samples, timeout=timeout)
        if NUM_DAQS == 2:
            data2 = ai_task2.read(number_of_samples_per_channel=n_samples, timeout=timeout)
        ao_task.wait_until_done(timeout + 1.0)
        print('Done.')

    # ── Save CSV ──────────────────────────────────────────────────────────
    arr1 = np.array(data1)
    if arr1.ndim == 1:
        arr1 = arr1[np.newaxis, :]
    if NUM_DAQS == 2:
        arr2 = np.array(data2)
        if arr2.ndim == 1:
            arr2 = arr2[np.newaxis, :]
        arr = np.vstack([arr1, arr2])
    else:
        arr = arr1

    times = np.arange(n_samples) / rate
    ch_labels = [f'{device_name}_{ch}' for ch in input_channels]
    if NUM_DAQS == 2:
        ch_labels += [f'{device_name2}_{ch}' for ch in input_channels]

    out = np.vstack([times] + [arr[i] for i in range(arr.shape[0])]).T
    header = 'time,' + ','.join(ch_labels)

    if filename is None:
        filename = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
    np.savetxt(filename, out, delimiter=',', header=header, comments='')
    print('Saved:', os.path.abspath(filename))