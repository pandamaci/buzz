import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from buzz.google_drive_downloader import (
    DriveAuthenticationError,
    DriveAuthorizationError,
    DriveCancelledError,
    DriveCredentialsUnconfiguredError,
    DriveDownloadDeniedError,
    DriveDownloader,
    DriveFileReference,
    DriveRedirectError,
    DriveNotFoundError,
    DriveStorageUnavailableError,
    DriveTransientError,
    DriveUnsupportedTypeError,
    DriveURLParseError,
    parse_drive_url,
)
from buzz.store import keyring_store


class Response:
    def __init__(self, status=200, payload=None, chunks=(), headers=None):
        self.status_code = status
        self._payload = payload
        self._chunks = chunks
        self.headers = headers or {}
        self.closed = False

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        return iter(self._chunks)

    def close(self):
        self.closed = True


class Service:
    def __init__(self, credentials=SimpleNamespace(token="access", valid=True), status=None):
        self._credentials = credentials
        self._status = status or keyring_store.DriveCredentialStatus.AVAILABLE

    def state(self):
        return SimpleNamespace(status=self._status)

    def credentials(self):
        return self._credentials


def metadata(**overrides):
    value = {
        "id": "file-id",
        "name": "audio.bin",
        "mimeType": "audio/mpeg",
        "size": "6",
        "resourceKey": "server-key",
        "capabilities": {"canDownload": True},
    }
    value.update(overrides)
    return value


def downloader(responses, **kwargs):
    session = Mock()
    session.get.side_effect = responses
    return DriveDownloader(Service(), session=session, chunk_size=2, **kwargs), session


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://drive.google.com/file/d/file-id/view", DriveFileReference("file-id")),
        (
            "https://www.drive.google.com/open?id=file_id-1&resourcekey=resource_key",
            DriveFileReference("file_id-1", "resource_key"),
        ),
    ],
)
def test_parse_trusted_drive_urls(url, expected):
    assert parse_drive_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://drive.google.com/file/d/file-id",
        "https://evil.example/file/d/file-id",
        "https://drive.google.com/file/d/bad.id",
        "https://drive.google.com/open?id=file-id&resourcekey=bad.key",
        "https://drive.google.com/open?id=file-id&id=other",
    ],
)
def test_parser_rejects_untrusted_or_invalid_urls_without_echoing_input(url):
    with pytest.raises(DriveURLParseError) as error:
        parse_drive_url(url)
    assert url not in str(error.value)
    assert "resourcekey" not in str(error.value)


@pytest.mark.parametrize("field", ["file_id", "resource_key"])
def test_direct_reference_rejects_oversize_and_crlf_components(field):
    value = "x" * 257 if field == "file_id" else "x\r\n"
    kwargs = {field: value}
    if field == "file_id":
        kwargs["resource_key"] = "valid-key"
    else:
        kwargs["file_id"] = "valid-id"
    with pytest.raises(Exception):
        DriveFileReference(**kwargs)


def test_metadata_and_media_send_resource_key_and_shared_drive_parameters(tmp_path):
    metadata_response = Response(payload=metadata())
    media_response = Response(chunks=[b"ab", b"cdef"], headers={"Content-Length": "6"})
    service, session = downloader([metadata_response, media_response])
    result = service.download(
        "https://drive.google.com/file/d/file-id?resourcekey=resource_key", tmp_path / "out.bin"
    )
    assert result.path.read_bytes() == b"abcdef"
    metadata_call, media_call = session.get.call_args_list
    assert metadata_call.kwargs["params"] == {
        "fields": "id,name,mimeType,size,resourceKey,capabilities(canDownload)",
        "supportsAllDrives": True,
    }
    assert media_call.kwargs["params"] == {"alt": "media", "supportsAllDrives": True}
    assert metadata_call.kwargs["headers"]["X-Goog-Drive-Resource-Keys"] == "file-id/resource_key"
    assert media_call.kwargs["headers"] == metadata_call.kwargs["headers"]
    assert metadata_response.closed and media_response.closed


def test_redirect_is_terminal_without_following_cross_host_location(tmp_path):
    redirect = Response(status=302, headers={"Location": "https://evil.example/leak"})
    service, session = downloader([redirect])
    with pytest.raises(DriveRedirectError):
        service.download(DriveFileReference("file-id"), tmp_path / "out")
    assert session.get.call_count == 1
    assert session.get.call_args.kwargs["allow_redirects"] is False
    assert "evil.example" not in repr(session.get.call_args.kwargs["headers"])


def test_can_download_false_is_terminal_and_does_not_start_media_request(tmp_path):
    response = Response(payload=metadata(capabilities={"canDownload": False}))
    service, session = downloader([response])
    with pytest.raises(DriveDownloadDeniedError):
        service.download(DriveFileReference("file-id"), tmp_path / "out")
    assert session.get.call_count == 1


@pytest.mark.parametrize(
    "file_metadata",
    [
        metadata(mimeType="application/vnd.google-apps.document"),
        metadata(mimeType="application/vnd.google-apps.shortcut"),
    ],
)
def test_workspace_types_and_shortcuts_are_terminal(tmp_path, file_metadata):
    response = Response(payload=file_metadata)
    service, session = downloader([response])
    with pytest.raises(DriveUnsupportedTypeError):
        service.download(DriveFileReference("file-id"), tmp_path / "out")
    assert session.get.call_count == 1


@pytest.mark.parametrize(
    ("status", "exception"),
    [
        (401, DriveAuthenticationError),
        (403, DriveAuthorizationError),
        (404, DriveNotFoundError),
        (429, DriveTransientError),
        (503, DriveTransientError),
    ],
)
def test_metadata_http_classes_are_typed(tmp_path, status, exception):
    service, _session = downloader([Response(status=status)])
    with pytest.raises(exception):
        service.download(DriveFileReference("file-id"), tmp_path / "out")


def test_media_http_failure_is_typed_and_partial_file_removed(tmp_path):
    metadata_response = Response(payload=metadata())
    media_response = Response(status=404)
    service, _session = downloader([metadata_response, media_response])
    target = tmp_path / "out.bin"
    with pytest.raises(DriveNotFoundError):
        service.download(DriveFileReference("file-id"), target)
    assert not target.exists()
    assert metadata_response.closed and media_response.closed


def test_streaming_reports_progress_and_rejects_size_mismatch(tmp_path):
    metadata_response = Response(payload=metadata(size="7"))
    media_response = Response(chunks=[b"abc", b"def"], headers={"Content-Length": "6"})
    service, _session = downloader([metadata_response, media_response])
    updates = []
    target = tmp_path / "out.bin"
    with pytest.raises(DriveTransientError):
        service.download(DriveFileReference("file-id"), target, lambda done, total: updates.append((done, total)))
    assert updates == [(3, 7), (6, 7)]
    assert not target.exists()


def test_cancellation_deletes_partial_output_and_closes_response(tmp_path):
    metadata_response = Response(payload=metadata(size="6"))
    media_response = Response(chunks=[b"abc", b"def"])
    service, _session = downloader([metadata_response, media_response])
    cancelled = threading.Event()
    updates = []

    def progress(done, _total):
        updates.append(done)
        cancelled.set()

    target = tmp_path / "out.bin"
    with pytest.raises(DriveCancelledError):
        service.download(DriveFileReference("file-id"), target, progress, cancelled)
    assert updates == [3]
    assert not target.exists()
    assert media_response.closed


@pytest.mark.parametrize(
    ("status", "exception"),
    [
        (keyring_store.DriveCredentialStatus.UNCONFIGURED, DriveCredentialsUnconfiguredError),
        (keyring_store.DriveCredentialStatus.UNAVAILABLE, DriveStorageUnavailableError),
    ],
)
def test_credential_state_is_typed(status, exception, tmp_path):
    service = DriveDownloader(Service(status=status), session=Mock())
    with pytest.raises(exception):
        service.download(DriveFileReference("file-id"), tmp_path / "out")


def test_network_errors_are_typed_without_sensitive_values(tmp_path):
    session = Mock()
    session.get.side_effect = requests.ConnectionError("access-token file-id/resource_key")
    service = DriveDownloader(Service(), session=session)
    with pytest.raises(DriveTransientError) as error:
        service.download(
            "https://drive.google.com/file/d/file-id?resourcekey=resource_key",
            tmp_path / "out",
        )
    message = str(error.value)
    assert "access-token" not in message
    assert "resource_key" not in message
    assert "file-id" not in message
