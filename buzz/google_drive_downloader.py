"""Authenticated, non-UI Google Drive v3 binary-file downloader.

This module intentionally owns no fallback policy.  Callers can use the typed
exceptions to implement the Phase 3 routing matrix.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, quote, urlsplit

import google.auth.transport.requests
import requests

from buzz.google_drive_oauth import DriveOAuthService
from buzz.store import keyring_store


DRIVE_FILES_ENDPOINT = "https://www.googleapis.com/drive/v3/files"
DRIVE_HOSTS = frozenset({"drive.google.com", "www.drive.google.com"})
METADATA_FIELDS = "id,name,mimeType,size,resourceKey,capabilities(canDownload)"
DEFAULT_TIMEOUT = (10.0, 60.0)
CHUNK_SIZE = 1024 * 1024
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


class DriveDownloadError(RuntimeError):
    """Base class for errors that are terminal to one Drive API attempt."""


class DriveURLParseError(DriveDownloadError):
    pass


class DriveRedirectError(DriveDownloadError):
    """Drive REST never follows redirects or forwards credentials to them."""


class DriveStorageUnavailableError(DriveDownloadError):
    pass


class DriveCredentialsUnconfiguredError(DriveDownloadError):
    pass


class DriveCredentialsUnavailableError(DriveDownloadError):
    pass


class DriveAuthenticationError(DriveDownloadError):
    pass


class DriveAuthorizationError(DriveDownloadError):
    pass


class DriveDownloadDeniedError(DriveDownloadError):
    pass


class DriveNotFoundError(DriveDownloadError):
    pass


class DriveUnsupportedTypeError(DriveDownloadError):
    pass


class DriveTransientError(DriveDownloadError):
    pass


class DriveCancelledError(DriveDownloadError):
    pass


@dataclass(frozen=True)
class DriveFileReference:
    file_id: str
    resource_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.file_id, str) or not _valid_component(self.file_id):
            raise DriveURLParseError("unsupported Google Drive file reference")
        if self.resource_key is not None and (
            not isinstance(self.resource_key, str) or not _valid_component(self.resource_key)
        ):
            raise DriveURLParseError("unsupported Google Drive file reference")


@dataclass(frozen=True)
class DriveFileMetadata:
    file_id: str
    name: str
    mime_type: str
    size: int | None
    resource_key: str | None
    can_download: bool


@dataclass(frozen=True)
class DriveDownloadResult:
    path: Path
    metadata: DriveFileMetadata
    bytes_downloaded: int


def _valid_component(value: str) -> bool:
    return _ID_PATTERN.fullmatch(value) is not None


def parse_drive_url(url: str) -> DriveFileReference:
    """Parse only trusted HTTPS Drive file links without retaining the URL."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if (
            parsed.scheme.lower() != "https"
            or host is None
            or host.lower() not in DRIVE_HOSTS
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError
        # Accessing port validates malformed port syntax.  Drive links do not
        # need a non-default port and accepting one would not be trusted HTTPS.
        if parsed.port is not None:
            raise ValueError
        query = parse_qs(parsed.query, keep_blank_values=True)
        resource_values = query.get("resourcekey", [])
        if len(resource_values) > 1 or (
            resource_values and not _valid_component(resource_values[0])
        ):
            raise ValueError
        resource_key = resource_values[0] if resource_values else None

        file_id: str | None = None
        match = re.fullmatch(r"/file/d/([^/]+)(?:/[^/]*)?", parsed.path)
        if match:
            file_id = match.group(1)
        elif parsed.path == "/open":
            id_values = query.get("id", [])
            if len(id_values) == 1:
                file_id = id_values[0]
        if file_id is None or not _valid_component(file_id):
            raise ValueError
        return DriveFileReference(file_id, resource_key)
    except (TypeError, ValueError):
        raise DriveURLParseError("unsupported Google Drive URL") from None


Cancellation = Callable[[], bool] | threading.Event | None
ProgressCallback = Callable[[int, int | None], None]


def _cancelled(cancel: Cancellation) -> bool:
    if cancel is None:
        return False
    if isinstance(cancel, threading.Event):
        return cancel.is_set()
    return bool(cancel())


class DriveDownloader:
    """Download an authorized ordinary Drive binary file to a local path."""

    def __init__(
        self,
        oauth_service: DriveOAuthService,
        session: requests.Session | None = None,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        chunk_size: int = CHUNK_SIZE,
    ):
        self._oauth_service = oauth_service
        self._session = session or requests.Session()
        self._timeout = timeout
        self._chunk_size = chunk_size

    def _headers(self, reference: DriveFileReference, token: str) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {token}"}
        if reference.resource_key is not None:
            headers["X-Goog-Drive-Resource-Keys"] = (
                f"{reference.file_id}/{reference.resource_key}"
            )
        return headers

    def _credentials(self) -> Any:
        try:
            state = self._oauth_service.state()
        except Exception:
            raise DriveStorageUnavailableError(
                "Google Drive credential storage is unavailable"
            ) from None
        if state.status is keyring_store.DriveCredentialStatus.UNAVAILABLE:
            raise DriveStorageUnavailableError(
                "Google Drive credential storage is unavailable"
            )
        if state.status is keyring_store.DriveCredentialStatus.UNCONFIGURED:
            raise DriveCredentialsUnconfiguredError(
                "Google Drive credentials are not configured"
            )
        try:
            credentials = self._oauth_service.credentials()
        except keyring_store.DriveCredentialStoreError:
            raise DriveCredentialsUnavailableError(
                "Google Drive credentials are unavailable"
            ) from None
        except Exception:
            raise DriveAuthenticationError(
                "Google Drive credentials could not be loaded"
            ) from None
        if credentials is None:
            raise DriveCredentialsUnconfiguredError(
                "Google Drive credentials are not configured"
            )
        if not getattr(credentials, "valid", bool(getattr(credentials, "token", None))):
            try:
                credentials = self._oauth_service.refresh(
                    google.auth.transport.requests.Request(session=self._session)
                )
            except Exception:
                raise DriveAuthenticationError(
                    "Google Drive credentials could not be refreshed"
                ) from None
        token = getattr(credentials, "token", None)
        if not isinstance(token, str) or not token:
            raise DriveAuthenticationError("Google Drive credentials are invalid")
        return credentials

    @staticmethod
    def _endpoint(reference: DriveFileReference) -> str:
        return f"{DRIVE_FILES_ENDPOINT}/{quote(reference.file_id, safe='')}"

    @staticmethod
    def _response_error(status_code: int, media: bool = False) -> DriveDownloadError | None:
        if 300 <= status_code < 400:
            return DriveRedirectError("Google Drive returned an unexpected redirect")
        if status_code in (401,):
            return DriveAuthenticationError("Google Drive authentication failed")
        if status_code == 403:
            return DriveAuthorizationError("Google Drive authorization was denied")
        if status_code == 404:
            return DriveNotFoundError("Google Drive file was not found")
        if status_code == 429 or status_code >= 500:
            return DriveTransientError("Google Drive service is temporarily unavailable")
        if 400 <= status_code < 500:
            return DriveAuthorizationError("Google Drive request was not authorized")
        return None

    def _metadata(
        self, reference: DriveFileReference, token: str, cancel: Cancellation
    ) -> DriveFileMetadata:
        if _cancelled(cancel):
            raise DriveCancelledError("Google Drive download was cancelled")
        params = {"fields": METADATA_FIELDS, "supportsAllDrives": True}
        response = None
        try:
            response = self._session.get(
                self._endpoint(reference),
                params=params,
                headers=self._headers(reference, token),
                timeout=self._timeout,
                allow_redirects=False,
            )
            error = self._response_error(response.status_code)
            if error:
                raise error
            try:
                payload = response.json()
            except (TypeError, ValueError):
                raise DriveTransientError("Google Drive metadata was invalid") from None
        except requests.RequestException:
            raise DriveTransientError("Google Drive metadata request failed") from None
        finally:
            if response is not None:
                response.close()
        if not isinstance(payload, Mapping):
            raise DriveTransientError("Google Drive metadata was invalid")
        file_id = payload.get("id")
        name = payload.get("name")
        mime_type = payload.get("mimeType")
        capabilities = payload.get("capabilities")
        can_download = isinstance(capabilities, Mapping) and capabilities.get("canDownload") is True
        if not isinstance(file_id, str) or not isinstance(name, str) or not isinstance(mime_type, str):
            raise DriveTransientError("Google Drive metadata was incomplete")
        if not can_download:
            raise DriveDownloadDeniedError("Google Drive does not allow this file to be downloaded")
        if mime_type == "application/vnd.google-apps.shortcut" or mime_type.startswith(
            "application/vnd.google-apps."
        ):
            raise DriveUnsupportedTypeError("Google Drive file type is not a downloadable binary")
        raw_size = payload.get("size")
        if raw_size is None:
            size = None
        else:
            try:
                size = int(raw_size)
                if size < 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise DriveTransientError("Google Drive metadata size was invalid") from None
        response_resource_key = payload.get("resourceKey")
        return DriveFileMetadata(
            file_id=file_id,
            name=name,
            mime_type=mime_type,
            size=size,
            resource_key=response_resource_key if isinstance(response_resource_key, str) else reference.resource_key,
            can_download=True,
        )

    def download(
        self,
        url: str | DriveFileReference,
        destination: str | os.PathLike[str],
        progress: ProgressCallback | None = None,
        cancel: Cancellation = None,
    ) -> DriveDownloadResult:
        reference = url if isinstance(url, DriveFileReference) else parse_drive_url(url)
        credentials = self._credentials()
        token = credentials.token
        metadata = self._metadata(reference, token, cancel)
        # Preserve the caller's validated capability-like key exactly for both
        # requests; the response's resourceKey is metadata, not a replacement.
        media_reference = reference
        target = Path(destination)
        temporary_path: str | None = None
        response = None
        downloaded = 0
        try:
            if _cancelled(cancel):
                raise DriveCancelledError("Google Drive download was cancelled")
            params = {"alt": "media", "supportsAllDrives": True}
            response = self._session.get(
                self._endpoint(media_reference),
                params=params,
                headers=self._headers(media_reference, token),
                timeout=self._timeout,
                stream=True,
                allow_redirects=False,
            )
            error = self._response_error(response.status_code, media=True)
            if error:
                raise error
            content_length = response.headers.get("Content-Length")
            expected_transport_size = None
            if content_length is not None:
                try:
                    expected_transport_size = int(content_length)
                    if expected_transport_size < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    raise DriveTransientError("Google Drive response size was invalid") from None
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".buzz-drive-", suffix=".part", dir=target.parent, delete=False
            ) as output:
                temporary_path = output.name
                for chunk in response.iter_content(chunk_size=self._chunk_size):
                    if _cancelled(cancel):
                        raise DriveCancelledError("Google Drive download was cancelled")
                    if not chunk:
                        continue
                    output.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, metadata.size)
            if _cancelled(cancel):
                raise DriveCancelledError("Google Drive download was cancelled")
            if (
                (metadata.size is not None and downloaded != metadata.size)
                or (
                    expected_transport_size is not None
                    and downloaded != expected_transport_size
                )
            ):
                raise DriveTransientError("Google Drive response size did not match metadata")
            os.replace(temporary_path, target)
            temporary_path = None
            return DriveDownloadResult(target, metadata, downloaded)
        except requests.RequestException:
            raise DriveTransientError("Google Drive media request failed") from None
        finally:
            if response is not None:
                response.close()
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
