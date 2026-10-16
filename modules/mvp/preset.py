"""Single source of truth for the first Russian WhisperX MVP.

The preset is intentionally independent of torch, Gradio and model downloads.
The existing Pydantic pipeline schema is imported lazily by the adapter.
No user preferences, HF tokens or outputs are written to repository YAML.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MvpPreset:
    """Fixed functional contract; device settings are per-run options."""

    model_size: str = "small"
    language: str = "ru"
    task: str = "transcribe"
    vad_enabled: bool = True
    alignment_enabled: bool = True
    diarization_enabled: bool = True
    output_format: str = "txt"
    default_device: str = "cpu"
    default_compute_type: str = "int8"
    default_batch_size: int = 1


MVP_PRESET = MvpPreset()


def build_mvp_pipeline_params(
    *,
    device: str = MVP_PRESET.default_device,
    compute_type: str = MVP_PRESET.default_compute_type,
    batch_size: int = MVP_PRESET.default_batch_size,
    diarization_enabled: bool = MVP_PRESET.diarization_enabled,
):
    """Build a fresh TranscriptionPipelineParams for UI, jobs or CLI.

    diarization_enabled=False is ONLY an ASR diagnostic mode, not a passed
    MVP result. Secrets are resolved by the existing pipeline from HF_TOKEN at
    execution time, never inserted into this serializable parameter object.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("MVP device must be cpu or cuda")
    if compute_type not in ("int8", "float32", "float16"):
        raise ValueError("Unsupported MVP compute type")
    if device == "cpu" and compute_type == "float16":
        raise ValueError("float16 is not supported on CPU")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
        raise ValueError("MVP batch size must be an integer from 1 through 8")
    if not isinstance(diarization_enabled, bool):
        raise ValueError("diarization_enabled must be boolean")

    # Import lazily: importing the preset must never load ML dependencies.
    from modules.whisper.data_classes import (
        BGMSeparationParams,
        DiarizationParams,
        TranscriptionPipelineParams,
        VadParams,
        WhisperParams,
    )

    return TranscriptionPipelineParams(
        whisper=WhisperParams(
            model_size=MVP_PRESET.model_size,
            lang=MVP_PRESET.language,
            is_translate=MVP_PRESET.task == "translate",
            compute_type=compute_type,
            batch_size=batch_size,
            enable_whisperx_alignment=MVP_PRESET.alignment_enabled,
            word_timestamps=False,  # WhisperX alignment supplies word timestamps.
            enable_offload=True,
        ),
        vad=VadParams(vad_filter=MVP_PRESET.vad_enabled),
        diarization=DiarizationParams(
            is_diarize=diarization_enabled,
            use_whisperx_diarization=diarization_enabled,
            diarization_device=device,
            compute_type="float32",
            hf_token="",  # The existing runtime reads HF_TOKEN from the environment.
            assign_word_speakers=diarization_enabled,
            enable_offload=True,
        ),
        bgm_separation=BGMSeparationParams(is_separate_bgm=False),
    )
