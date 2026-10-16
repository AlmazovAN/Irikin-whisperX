"""MVP settings shared by future UI, job service and diagnostic CLI."""

from .preset import MVP_PRESET, MvpPreset, build_mvp_pipeline_params

__all__ = ("MVP_PRESET", "MvpPreset", "build_mvp_pipeline_params")
