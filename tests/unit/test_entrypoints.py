"""Basic repository integrity checks; no application imports or downloads."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative_path",
    [
        "app.py",
        "modules/whisper/base_transcription_pipeline.py",
        "modules/whisper/whisperx_inference.py",
        "backend/main.py",
        "requirements.txt",
        "Install.bat",
        "start-webui.bat",
    ],
)
def test_core_project_entrypoints_exist(relative_path):
    assert (ROOT / relative_path).is_file(), relative_path


def test_install_script_references_requirements():
    installer = (ROOT / "Install.bat").read_text(encoding="utf-8").lower()
    assert "requirements.txt" in installer
