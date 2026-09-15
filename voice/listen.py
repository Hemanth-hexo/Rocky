"""Speech-to-text via faster-whisper, running fully local/offline."""

from faster_whisper import WhisperModel

_model = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        # "base" — accurate enough for commands, light enough to leave headroom
        # for the LLM and TTS running alongside it.
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str) -> str:
    # Without a pinned language, Whisper auto-detects per clip from the
    # first few seconds of audio — on short or slightly noisy push-to-talk
    # recordings it sometimes misfires onto an unrelated language (Chinese
    # is a well-known false-positive for this model family) and transcribes
    # gibberish in that language instead of the English actually spoken.
    segments, _ = _get_model().transcribe(audio_path, language="en")
    return " ".join(seg.text.strip() for seg in segments)
