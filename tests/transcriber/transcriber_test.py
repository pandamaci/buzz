import pathlib
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import parse_qs

import pytest

from buzz.transcriber.file_transcriber import (
    FileTranscriber,
    extract_google_drive_file_id,
    write_output,
    to_timestamp,
)
from buzz.transcriber.transcriber import (
    OutputFormat,
    Segment,
)
from buzz.store import keyring_store
from buzz.google_drive_downloader import (
    DriveAuthenticationError,
    DriveDownloadDeniedError,
    DriveNotFoundError,
    DriveTransientError,
)


@pytest.mark.parametrize(
    "url,file_id",
    [
        ("https://drive.google.com/file/d/abc_123-xyz/view?usp=sharing", "abc_123-xyz"),
        ("https://drive.google.com/open?id=abc_123-xyz", "abc_123-xyz"),
        ("https://drive.google.com/uc?id=abc_123-xyz", None),
        ("https://example.com/file/d/abc_123-xyz/view", None),
        ("https://drive.google.com/file/d/not%20an%20id/view", None),
    ],
)
def test_extract_google_drive_file_id(url, file_id):
    assert extract_google_drive_file_id(url) == file_id


class TestFileTranscriberDownload:
    @staticmethod
    def transcriber(url):
        class ConcreteFileTranscriber(FileTranscriber):
            def transcribe(self):
                return []

            def stop(self):
                pass

        task = Mock(url=url)
        task.file_path = None
        transcriber = ConcreteFileTranscriber(task)
        # Existing fallback tests model the no-Drive-configuration path.
        transcriber._drive_oauth_service = Mock()
        transcriber._drive_oauth_service.state.return_value = Mock(
            status=keyring_store.DriveCredentialStatus.UNCONFIGURED
        )
        return transcriber

    def test_google_drive_fallback_after_ytdlp_failure(self, tmp_path):
        transcriber = self.transcriber("https://drive.google.com/file/d/file-id/view")
        response = MagicMock()
        response.getcode.return_value = 200
        response.geturl.return_value = "https://drive.usercontent.google.com/download?id=file-id&export=download"
        response.headers = {"Content-Length": "6", "Content-Type": "audio/mpeg"}
        response.read.side_effect = [b"abc", b"def", b""]

        ytdlp = MagicMock()
        ytdlp.__enter__.return_value = ytdlp
        ytdlp.extract_info.side_effect = RuntimeError("metadata unavailable")
        ytdlp.download.side_effect = RuntimeError("download unavailable")
        response.__enter__.return_value = response
        ytdlp_factory = MagicMock(return_value=ytdlp)
        ytdlp_factory.sanitize_info.return_value = {"title": "Google Drive file file-id"}

        with patch("buzz.transcriber.file_transcriber.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.build_opener") as build_opener, \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            opener = build_opener.return_value
            opener.open.return_value = response
            run.return_value.stderr = b""
            run.return_value.returncode = 0

            assert transcriber._download_from_url()

        assert opener.open.call_count == 1
        assert (tmp_path / "Google Drive file file-id").read_bytes() == b"abcdef"
        assert transcriber.transcription_task.file_path is not None
        assert transcriber.transcription_task.file_path.endswith(".wav")

    def test_authenticated_drive_is_first_and_terminal(self, tmp_path, caplog):
        url = "https://drive.google.com/file/d/file-id/view?resourcekey=capability-key"
        transcriber = self.transcriber(url)
        transcriber._drive_oauth_service.state.return_value = Mock(
            status=keyring_store.DriveCredentialStatus.AVAILABLE
        )
        transcriber._drive_oauth_service.credentials.return_value = Mock(
            token="access", valid=True
        )
        drive_downloader = MagicMock()
        drive_downloader.download.side_effect = lambda reference, path, **kwargs: Path(path).write_bytes(b"audio")
        ytdlp_factory = MagicMock()
        ytdlp_factory.sanitize_info.return_value = {"title": "Google Drive file file-id"}
        with patch("buzz.transcriber.file_transcriber.DriveDownloader", return_value=drive_downloader), \
             patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stderr = b""
            assert transcriber._download_from_url()
        ytdlp_factory.assert_not_called()
        drive_downloader.download.assert_called_once()
        assert "capability-key" not in caplog.text

    @pytest.mark.parametrize(
        "outcome",
        [
            DriveDownloadDeniedError("denied"),
            DriveAuthenticationError("auth"),
            DriveNotFoundError("missing"),
            DriveTransientError("temporary"),
        ],
    )
    def test_authenticated_drive_outcomes_never_use_fallback(self, tmp_path, outcome):
        transcriber = self.transcriber(
            "https://drive.google.com/open?id=file-id&resourcekey=secret-key"
        )
        transcriber._drive_oauth_service.state.return_value = Mock(
            status=keyring_store.DriveCredentialStatus.AVAILABLE
        )
        transcriber._drive_oauth_service.credentials.return_value = Mock(
            token="access", valid=True
        )
        drive_downloader = MagicMock()
        drive_downloader.download.side_effect = outcome
        ytdlp_factory = MagicMock()
        ytdlp_factory.sanitize_info.return_value = {"title": "Google Drive file file-id"}
        with patch("buzz.transcriber.file_transcriber.DriveDownloader", return_value=drive_downloader), \
             patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory):
            assert not transcriber._download_from_url()
        ytdlp_factory.assert_not_called()
        assert transcriber.transcription_task.file_path is None

    @pytest.mark.parametrize(
        "status",
        [
            keyring_store.DriveCredentialStatus.UNAVAILABLE,
            keyring_store.DriveCredentialStatus.UNCONFIGURED,
        ],
    )
    def test_drive_storage_or_credentials_route_as_required(self, tmp_path, status):
        transcriber = self.transcriber("https://drive.google.com/file/d/file-id/view")
        transcriber._drive_oauth_service.state.return_value = Mock(status=status)
        ytdlp_factory = MagicMock()
        ytdlp_factory.sanitize_info.return_value = {"title": "Google Drive file file-id"}
        with patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stderr = b""
            result = transcriber._download_from_url()
        if status is keyring_store.DriveCredentialStatus.UNAVAILABLE:
            assert not result
            ytdlp_factory.assert_not_called()
        else:
            # Unconfigured Drive keeps the pre-Phase-3 legacy routing path.
            assert result
            ytdlp_factory.assert_called()

    def test_configured_unreadable_drive_credentials_are_terminal(self):
        transcriber = self.transcriber("https://drive.google.com/file/d/file-id/view")
        transcriber._drive_oauth_service.state.return_value = Mock(
            status=keyring_store.DriveCredentialStatus.AVAILABLE
        )
        transcriber._drive_oauth_service.credentials.side_effect = ValueError("corrupt")
        ytdlp_factory = MagicMock()
        with patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory):
            assert not transcriber._download_from_url()
        ytdlp_factory.assert_not_called()

    def test_drive_log_does_not_include_full_url_or_resource_key(self, tmp_path, caplog):
        url = "https://drive.google.com/file/d/file-id/view?resourcekey=secret-key&usp=sharing"
        transcriber = self.transcriber(url)
        transcriber._drive_oauth_service.state.return_value = Mock(
            status=keyring_store.DriveCredentialStatus.AVAILABLE
        )
        transcriber._drive_oauth_service.credentials.return_value = Mock(
            token="access", valid=True
        )
        drive_downloader = MagicMock()
        drive_downloader.download.side_effect = DriveTransientError("temporary")
        with patch("buzz.transcriber.file_transcriber.DriveDownloader", return_value=drive_downloader):
            assert not transcriber._download_from_url()
        assert url not in caplog.text
        assert "secret-key" not in caplog.text

    @pytest.mark.parametrize(
        "url,secret",
        [
            (
                "http://drive.google.com/file/d/file-id/view?resourcekey=http-secret#http-fragment",
                "http-secret",
            ),
            (
                "https://drive.google.com:invalid-port/file/d/file-id/view?resourcekey=port-secret#port-fragment",
                "port-secret",
            ),
            (
                "https://user:password@drive.google.com/file/d/file-id/view?resourcekey=user-secret#user-fragment",
                "password",
            ),
        ],
    )
    def test_legacy_ytdlp_drive_logger_redacts_candidates(self, url, secret, tmp_path, caplog):
        transcriber = self.transcriber(url)
        ytdlp = MagicMock()
        ytdlp.__enter__.return_value = ytdlp
        ytdlp.extract_info.return_value = {"title": "audio"}
        ytdlp.download.return_value = None
        ytdlp_factory = MagicMock(return_value=ytdlp)
        ytdlp_factory.sanitize_info.return_value = {"title": "audio"}

        with patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stderr = b""
            assert transcriber._download_from_url()
            logger = ytdlp_factory.call_args.args[0]["logger"]
            logger.error("yt-dlp URL: %s", url)

        assert secret not in caplog.text
        assert "resourcekey=" not in caplog.text
        assert "#http-fragment" not in caplog.text
        assert "#port-fragment" not in caplog.text
        assert "#user-fragment" not in caplog.text
        assert "password@" not in caplog.text
        assert "http://" not in caplog.text
        assert "https://" not in caplog.text

    def test_non_drive_url_keeps_ytdlp_path(self, tmp_path):
        transcriber = self.transcriber("https://example.com/audio")
        ytdlp = MagicMock()
        ytdlp.__enter__.return_value = ytdlp
        ytdlp.extract_info.return_value = {"title": "audio"}
        ytdlp.download.return_value = None
        ytdlp_factory = MagicMock(return_value=ytdlp)
        with patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stderr = b""
            assert transcriber._download_from_url()
        ytdlp.extract_info.assert_called_once_with("https://example.com/audio", download=False)
        ytdlp.download.assert_called_once_with(["https://example.com/audio"])

    def test_google_drive_fallback_when_download_ytdlp_constructor_fails(self, tmp_path):
        transcriber = self.transcriber("https://drive.google.com/file/d/file-id/view")
        response = MagicMock()
        response.__enter__.return_value = response
        response.getcode.return_value = 200
        response.geturl.return_value = "https://drive.usercontent.google.com/download?id=file-id&export=download"
        response.headers = {"Content-Length": "3", "Content-Type": "audio/mpeg"}
        response.read.side_effect = [b"abc", b""]

        metadata_ydl = MagicMock()
        metadata_ydl.__enter__.return_value = metadata_ydl
        metadata_ydl.extract_info.return_value = {"title": "drive audio"}
        ytdlp_factory = MagicMock(side_effect=[metadata_ydl, RuntimeError("constructor failed")])
        ytdlp_factory.sanitize_info.return_value = {"title": "drive audio"}

        with patch("buzz.transcriber.file_transcriber.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.build_opener") as build_opener, \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            build_opener.return_value.open.return_value = response
            run.return_value.stderr = b""
            run.return_value.returncode = 0
            assert transcriber._download_from_url()

        assert (tmp_path / "drive audio").read_bytes() == b"abc"

    def test_google_drive_fallback_rejects_content_length_mismatch(self, tmp_path):
        transcriber = self.transcriber("https://drive.google.com/file/d/file-id/view")
        response = MagicMock()
        response.__enter__.return_value = response
        response.getcode.return_value = 200
        response.geturl.return_value = "https://drive.usercontent.google.com/download?id=file-id&export=download"
        response.headers = {"Content-Length": "6", "Content-Type": "audio/mpeg"}
        response.read.side_effect = [b"abc", b""]

        ytdlp = MagicMock()
        ytdlp.__enter__.return_value = ytdlp
        ytdlp.extract_info.side_effect = RuntimeError("metadata unavailable")
        ytdlp.download.side_effect = RuntimeError("download unavailable")
        ytdlp_factory = MagicMock(return_value=ytdlp)
        ytdlp_factory.sanitize_info.return_value = {"title": "drive audio"}

        with patch("buzz.transcriber.file_transcriber.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.build_opener") as build_opener, \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            build_opener.return_value.open.return_value = response
            assert not transcriber._download_from_url()

        run.assert_not_called()
        assert not (tmp_path / "drive audio").exists()
        assert transcriber.transcription_task.file_path is None

    def test_ffmpeg_nonzero_exit_without_stderr_fails(self, tmp_path):
        transcriber = self.transcriber("https://example.com/audio")
        ytdlp = MagicMock()
        ytdlp.__enter__.return_value = ytdlp
        ytdlp.extract_info.return_value = {"title": "audio"}
        ytdlp.download.return_value = None
        ytdlp_factory = MagicMock(return_value=ytdlp)
        ytdlp_factory.sanitize_info.return_value = {"title": "audio"}

        with patch("buzz.transcriber.file_transcriber.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            run.return_value.stderr = b""
            run.return_value.returncode = 1
            with pytest.raises(Exception, match="ffmpeg exited with status 1"):
                transcriber._download_from_url()

        assert transcriber.transcription_task.file_path is None

    def test_google_drive_confirmation_form_preserves_hidden_fields(self, tmp_path):
        transcriber = self.transcriber("https://drive.google.com/file/d/file-id/view")
        confirmation = MagicMock()
        confirmation.getcode.return_value = 200
        confirmation.geturl.return_value = "https://drive.usercontent.google.com/download?id=file-id&export=download"
        confirmation.headers = {"Content-Type": "text/html"}
        confirmation.read.return_value = b'''<form action="https://drive.google.com/uc" method="get">
          <input type="hidden" name="id" value="file-id">
          <input type="hidden" name="export" value="download">
          <input type="hidden" name="confirm" value="t">
          <input type="hidden" name="uuid" value="session-uuid">
        </form>'''
        media = MagicMock()
        media.getcode.return_value = 200
        media.geturl.return_value = "https://drive.google.com/uc"
        media.headers = {"Content-Length": "3", "Content-Type": "audio/mpeg"}
        media.read.side_effect = [b"abc", b""]

        ytdlp = MagicMock()
        ytdlp.__enter__.return_value = ytdlp
        ytdlp.extract_info.side_effect = RuntimeError("metadata unavailable")
        ytdlp.download.side_effect = RuntimeError("download unavailable")
        ytdlp_factory = MagicMock(return_value=ytdlp)
        ytdlp_factory.sanitize_info.return_value = {"title": "drive audio"}

        with patch("buzz.transcriber.file_transcriber.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.build_opener") as build_opener, \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            opener = build_opener.return_value
            opener.open.side_effect = [confirmation, media]
            run.return_value.stderr = b""
            run.return_value.returncode = 0
            assert transcriber._download_from_url()

        assert opener.open.call_count == 2
        followup_request = opener.open.call_args_list[1].args[0]
        query = parse_qs(followup_request.full_url.split("?", 1)[1])
        assert query == {
            "id": ["file-id"],
            "export": ["download"],
            "confirm": ["t"],
            "uuid": ["session-uuid"],
        }
        assert (tmp_path / "drive audio").read_bytes() == b"abc"

    @pytest.mark.parametrize(
        "action",
        [
            "https://evil.example/download?id=file-id",
            "not a valid confirmation form",
        ],
    )
    def test_google_drive_confirmation_rejects_unsafe_or_malformed_action(self, tmp_path, action):
        transcriber = self.transcriber("https://drive.google.com/file/d/file-id/view")
        confirmation = MagicMock()
        confirmation.getcode.return_value = 200
        confirmation.geturl.return_value = "https://drive.usercontent.google.com/download?id=file-id&export=download"
        confirmation.headers = {"Content-Type": "text/html"}
        confirmation.read.return_value = (
            action.encode()
            if action.startswith("not")
            else f'<form action="{action}" method="get"><input type="hidden" name="id" value="file-id"></form>'.encode()
        )

        ytdlp = MagicMock()
        ytdlp.__enter__.return_value = ytdlp
        ytdlp.extract_info.side_effect = RuntimeError("metadata unavailable")
        ytdlp.download.side_effect = RuntimeError("download unavailable")
        ytdlp_factory = MagicMock(return_value=ytdlp)
        ytdlp_factory.sanitize_info.return_value = {"title": "drive audio"}

        with patch("buzz.transcriber.file_transcriber.tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("buzz.transcriber.file_transcriber.YoutubeDL", ytdlp_factory), \
             patch("buzz.transcriber.file_transcriber.build_opener") as build_opener, \
             patch("buzz.transcriber.file_transcriber.subprocess.run") as run:
            build_opener.return_value.open.return_value = confirmation
            assert not transcriber._download_from_url()

        assert build_opener.return_value.open.call_count == 1
        run.assert_not_called()
        assert not (tmp_path / "drive audio").exists()


class TestToTimestamp:
    def test_to_timestamp(self):
        assert to_timestamp(0) == "00:00:00.000"
        assert to_timestamp(123456789) == "34:17:36.789"


@pytest.mark.parametrize(
    "output_format,output_text",
    [
        (OutputFormat.TXT, "Bien venue dans "),
        (
            OutputFormat.SRT,
            "1\n00:00:00,040 --> 00:00:00,299\nBien\n\n2\n00:00:00,299 --> 00:00:00,329\nvenue dans\n\n",
        ),
        (
            OutputFormat.VTT,
            "WEBVTT\n\n00:00:00.040 --> 00:00:00.299\nBien\n\n00:00:00.299 --> 00:00:00.329\nvenue dans\n\n",
        ),
    ],
)
def test_write_output(
    tmp_path: pathlib.Path, output_format: OutputFormat, output_text: str
):
    output_file_path = tmp_path / "whisper.txt"
    segments = [Segment(40, 299, "Bien"), Segment(299, 329, "venue dans")]

    write_output(
        path=str(output_file_path), segments=segments, output_format=output_format
    )

    with open(output_file_path, encoding="utf-8") as output_file:
        assert output_text == output_file.read()
