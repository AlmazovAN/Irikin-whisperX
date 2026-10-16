"""Actionable, redacted errors for gated diarization and WhisperX alignment.

No torch, Gradio, pyannote or network imports. Never include tokens, paths,
third-party exception messages or request payloads in public error strings.
"""


class ModelStageError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


HF_ACCESS_GUIDANCE = (
    "Cannot access the Pyannote speaker-diarization model. "
    "Check HF_TOKEN read access and accept the gated model conditions "
    "on Hugging Face for pyannote/speaker-diarization-3.1 and "
    "pyannote/segmentation-3.0. Verify the models are reachable."
)


def require_hf_token(token: str | None) -> str:
    """Explicit preflight; never return a masked token or store it on a job."""
    if not isinstance(token, str) or not token.strip():
        raise ModelStageError(
            "HF_TOKEN_MISSING",
            "Diarization requires HF_TOKEN. Set a Hugging Face read token "
            "in your private environment and accept the gated Pyannote model terms.",
        )
    return token


def require_device(device: str, cuda_available: bool) -> None:
    """Fail clearly when the user explicitly selected unavailable CUDA."""
    if device not in ("cpu", "cuda"):
        raise ModelStageError("DEVICE_UNSUPPORTED", "MVP diarization supports CPU or NVIDIA CUDA only")
    if device == "cuda" and not cuda_available:
        raise ModelStageError(
            "CUDA_UNAVAILABLE",
            "CUDA was requested but no NVIDIA CUDA device is available. Use CPU on AMD graphics.",
        )
