import soundfile as sf
import numpy as np
from scipy.signal import resample_poly


def preproc(audio_path):

    waveform, sr = sf.read(audio_path)

    # stereo → mono
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    # resample to 16kHz
    target_sr = 16000
    if sr != target_sr:
        waveform = resample_poly(waveform, target_sr, sr)
        sr = target_sr

    # normalize
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-6)

    return waveform, sr