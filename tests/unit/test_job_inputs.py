"""P08 intake validation: no network, no ASR/model imports, no user audio."""

import json
from pathlib import Path
import shutil
import subprocess
import wave

import pytest

from modules.jobs.inputs import (
    InputKind, InputLimits, InputSource, InputValidationError, validate_input,
)

pytestmark = pytest.mark.fast


@pytest.fixture
def roots(tmp_path):
    upload = tmp_path / "gradio-cache"
    local = tmp_path / "approved-recordings"
    forbidden = tmp_path / "outside"
    for directory in (upload, local, forbidden):
        directory.mkdir()
    return upload, local, forbidden


def pcm_wav(path: Path, seconds: float = 0.4) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def inspect(source, roots, **kwargs):
    upload, local, _ = roots
    return validate_input(source, upload_roots=[upload], local_roots=[local], **kwargs)


def check_error(code, func):
    with pytest.raises(InputValidationError) as exc:
        func()
    assert exc.value.code == code
    # UI-safe error messages must not reveal absolute paths.
    assert "/tmp/" not in str(exc.value)
    assert "C:\\Users" not in str(exc.value)


def test_real_wav_ffprobe_local_and_upload(roots):
    if shutil.which("ffprobe") is None:
        pytest.skip("FFprobe not installed in lightweight CI")
    upload, local, _ = roots
    local_audio = pcm_wav(local / "встреча.WAV")
    uploaded = pcm_wav(upload / "record.wav")
    first = inspect(InputSource.local_file(local_audio), roots)
    second = inspect(InputSource.upload(uploaded, original_name="Встреча 1.WAV"), roots)
    assert first.source_kind is InputKind.LOCAL_FILE
    assert first.safe_filename == "встреча.wav"
    assert second.source_kind is InputKind.UPLOAD
    assert second.safe_filename == "Встреча 1.wav"
    assert second.codec_name == "pcm_s16le"
    assert first.duration_seconds == pytest.approx(0.4, abs=0.01)
    assert first.size_bytes > 1000
    assert first.path == local_audio.resolve()
    assert first.storage_filename != inspect(InputSource.local_file(local_audio), roots).storage_filename
    assert "/" not in second.storage_filename and "\\" not in second.storage_filename


@pytest.mark.parametrize("extension,codec", [
    (".mp3", "libmp3lame"), (".m4a", "aac"), (".ogg", "libvorbis"),
])
def test_real_compressed_formats(roots, extension, codec):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg/FFprobe not installed")
    _, local, _ = roots
    wav = pcm_wav(local / "reference.wav", seconds=0.5)
    output = local / ("sample" + extension)
    run = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", str(wav), "-c:a", codec, str(output)],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if run.returncode != 0:
        pytest.skip("FFmpeg encoder unavailable")
    result = inspect(InputSource.local_file(output), roots)
    assert result.duration_seconds > 0
    assert result.safe_filename == output.name
    assert result.codec_name


def test_damaged_audio_is_rejected_without_inference(roots):
    if shutil.which("ffprobe") is None:
        pytest.skip("FFprobe not installed in lightweight CI")
    _, local, _ = roots
    path = local / "corrupted.mp3"
    path.write_bytes(b"this is definitely not an MP3")
    check_error("INVALID_MEDIA", lambda: inspect(InputSource.local_file(path), roots))


@pytest.mark.parametrize("name", ["bad.exe", "fake.webm", "test.mp4", "unknown"])
def test_extension_whitelist_rejects_unsupported_media(roots, name):
    _, local, _ = roots
    path = local / name
    path.write_bytes(b"not media")
    check_error("UNSUPPORTED_FORMAT", lambda: inspect(InputSource.local_file(path), roots))


def test_declared_upload_filename_is_validated_separately(roots):
    upload, _, _ = roots
    audio = pcm_wav(upload / "upload-temp.wav")
    check_error("UNSUPPORTED_FORMAT", lambda: inspect(
        InputSource.upload(audio, original_name="harmless.wav.exe"), roots,
    ))


def test_paths_outside_authorized_roots_are_rejected(roots):
    upload, local, outside = roots
    private = pcm_wav(outside / "private.wav")
    check_error("ACCESS_DENIED", lambda: inspect(InputSource.local_file(private), roots))
    check_error("ACCESS_DENIED", lambda: inspect(InputSource.upload(private), roots))
    check_error("ACCESS_DENIED", lambda: validate_input(
        InputSource.local_file(pcm_wav(local / "audio.wav")), local_roots=[upload],
    ))


def test_symlink_out_of_root_is_rejected(roots):
    upload, _, outside = roots
    target = pcm_wav(outside / "secret.wav")
    link = upload / "link.wav"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this system")
    check_error("ACCESS_DENIED", lambda: inspect(InputSource.upload(link), roots))


def test_deny_root_level_grants_even_if_file_exists(roots):
    _, local, _ = roots
    audio = pcm_wav(local / "test.wav")
    check_error("ACCESS_DENIED", lambda: validate_input(
        InputSource.local_file(audio), local_roots=[Path(local.anchor)],
    ))


def test_missing_and_empty_audio(roots):
    _, local, _ = roots
    check_error("FILE_NOT_FOUND", lambda: inspect(InputSource.local_file(local / "missing.wav"), roots))
    empty = local / "empty.wav"
    empty.touch()
    check_error("EMPTY_FILE", lambda: inspect(InputSource.local_file(empty), roots))
    check_error("ACCESS_DENIED", lambda: validate_input(InputSource.local_file(empty)))


def test_size_duration_and_timeout_limits(roots, monkeypatch):
    _, local, _ = roots
    audio = pcm_wav(local / "long.wav", seconds=0.5)
    check_error("FILE_TOO_LARGE", lambda: inspect(
        InputSource.local_file(audio), roots, limits=InputLimits(max_file_bytes=50),
    ))
    from types import SimpleNamespace
    monkeypatch.setattr("modules.jobs.inputs.shutil.which", lambda _: "/fake/ffprobe")
    monkeypatch.setattr("modules.jobs.inputs.subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps({
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}],
            "format": {"duration": "0.5"},
        }),
    ))
    check_error("AUDIO_TOO_LONG", lambda: inspect(
        InputSource.local_file(audio), roots, limits=InputLimits(max_duration_seconds=0.1),
    ))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 0.01)
    monkeypatch.setattr("modules.jobs.inputs.subprocess.run", timeout)
    check_error("FFPROBE_TIMEOUT", lambda: inspect(InputSource.local_file(audio), roots))


def test_missing_ffprobe_and_malformed_json(roots, monkeypatch):
    _, local, _ = roots
    audio = pcm_wav(local / "test.wav")
    monkeypatch.setattr("modules.jobs.inputs.shutil.which", lambda _: None)
    check_error("FFPROBE_UNAVAILABLE", lambda: inspect(InputSource.local_file(audio), roots))
    monkeypatch.setattr("modules.jobs.inputs.shutil.which", lambda _: "/fake/ffprobe")
    class Result:
        returncode = 0
        stdout = "{broken json"
    monkeypatch.setattr("modules.jobs.inputs.subprocess.run", lambda *a, **k: Result())
    check_error("INVALID_MEDIA", lambda: inspect(InputSource.local_file(audio), roots))


def test_video_stream_and_no_audio_stream_are_rejected(roots, monkeypatch):
    _, local, _ = roots
    audio = pcm_wav(local / "valid.wav")
    monkeypatch.setattr("modules.jobs.inputs.shutil.which", lambda _: "/fake/ffprobe")
    class Result:
        returncode = 0
        stdout = ""
    result = Result()
    monkeypatch.setattr("modules.jobs.inputs.subprocess.run", lambda *a, **k: result)
    for data in (
        {"streams": [{"codec_type": "video", "codec_name": "h264"}], "format": {"duration": "1"}},
        {"streams": [], "format": {"duration": "1"}},
        {"streams": [{"codec_type": "audio", "codec_name": "opus"}], "format": {"duration": "NaN"}},
    ):
        result.stdout = json.dumps(data)
        check_error("INVALID_MEDIA", lambda: inspect(InputSource.local_file(audio), roots))


def test_audio_cover_art_is_not_mistaken_for_video(roots, monkeypatch):
    _, local, _ = roots
    audio = pcm_wav(local / "podcast.m4a")
    monkeypatch.setattr("modules.jobs.inputs.shutil.which", lambda _: "/fake/ffprobe")
    from types import SimpleNamespace
    response = {
        "format": {"duration": "15.0"},
        "streams": [
            {"codec_type": "audio", "codec_name": "aac"},
            {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}},
        ],
    }
    monkeypatch.setattr("modules.jobs.inputs.subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps(response),
    ))
    assert inspect(InputSource.local_file(audio), roots).codec_name == "aac"


def test_portable_safe_names_and_unique_storage_ids(roots, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("modules.jobs.inputs.shutil.which", lambda _: "/fake/ffprobe")
    monkeypatch.setattr("modules.jobs.inputs.subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps({
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}],
            "format": {"duration": "0.4"},
        }),
    ))
    upload, _, _ = roots
    wav = pcm_wav(upload / "data.wav")
    a = inspect(InputSource.upload(wav, original_name=r"C:\temp\CON.WAV"), roots)
    b = inspect(InputSource.upload(wav, original_name=r"../../../../very nice:meeting?.wav"), roots)
    assert a.safe_filename == "audio_CON.wav"
    assert b.safe_filename == "very nice_meeting.wav"
    assert a.storage_filename != b.storage_filename
    assert all(c not in b.safe_filename for c in "/\\:*?\"<>|")


def test_drive_is_declared_but_not_supported_yet():
    check_error("SOURCE_NOT_SUPPORTED", lambda: validate_input(
        InputSource.google_drive_url("https://drive.google.com/file/d/abc/view"),
    ))


def test_invalid_source_and_limits():
    check_error("INVALID_SOURCE", lambda: InputSource("unknown", "/tmp/test.wav"))
    check_error("INVALID_SOURCE", lambda: InputSource.local_file(""))
    with pytest.raises(ValueError):
        InputLimits(max_file_bytes=0)
    with pytest.raises(ValueError):
        InputLimits(max_duration_seconds=float("nan"))


@pytest.mark.parametrize("suffix", [".mp3", ".wav", ".m4a", ".ogg"])
def test_whitelisted_formats_with_mock_ffprobe(roots, monkeypatch, suffix):
    _, local, _ = roots
    candidate = local / ("sample" + suffix)
    candidate.write_bytes(b"placeholder audio for fake probe")
    monkeypatch.setattr("modules.jobs.inputs.shutil.which", lambda _: "/fake/ffprobe")
    from types import SimpleNamespace
    monkeypatch.setattr("modules.jobs.inputs.subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}],
                          "format": {"duration": "65.2"}}),
    ))
    item = inspect(InputSource.local_file(candidate), roots)
    assert item.safe_filename == "sample" + suffix
    assert item.duration_seconds == 65.2
