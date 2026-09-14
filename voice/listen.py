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
    segments, _ = _get_model().transcribe(audio_path)
    return " ".join(seg.text.strip() for seg in segments)
