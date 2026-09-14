"""Post-processing to give Rocky's voice a synthesized "alien translator
device" quality — pitch-shifted down for depth, plus light ring modulation
for a metallic/robotic texture. Applied on top of a chosen base Kokoro voice,
not a clone of any real performer's voice."""

import librosa
import numpy as np


def robotize(
    audio: np.ndarray,
    sample_rate: int,
    pitch_shift_semitones: float = -4.0,
    ring_mod_freq: float = 30.0,
    ring_mod_mix: float = 0.15,
) -> np.ndarray:
    audio = audio.astype(np.float32)
    shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift_semitones)

    t = np.arange(len(shifted)) / sample_rate
    carrier = np.sin(2 * np.pi * ring_mod_freq * t).astype(np.float32)
    modulated = shifted * carrier

    result = (1 - ring_mod_mix) * shifted + ring_mod_mix * modulated

    peak = np.max(np.abs(result))
    original_peak = np.max(np.abs(shifted))
    if peak > 0:
        result = result / peak * original_peak

    return result.astype(np.float32)
