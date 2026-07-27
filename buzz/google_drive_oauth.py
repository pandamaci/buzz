"""Secure, advanced BYO Google Desktop OAuth backend for Drive.

Phase 1 deliberately contains no Drive REST or UI code.  The service exposes
credential/configuration primitives for the later downloader and Preferences
integration while keeping refresh credentials out of generic secret storage.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlsplit

import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from buzz.store import keyring_store


DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
CALLBACK_PATH = "/oauth2callback"
DEFAULT_TIMEOUT_SECONDS = 300
MAX_CLIENT_JSON_BYTES = 64 * 1024
_OAUTH_LOGGERS = ("oauthlib", "requests_oauthlib", "urllib3")
_OAUTH_LOG_LOCK = threading.RLock()


def _quiet_oauth_http_logging() -> dict[str, int]:
    _OAUTH_LOG_LOCK.acquire()
    previous = {}
    try:
        for name in _OAUTH_LOGGERS:
            logger = logging.getLogger(name)
            previous[name] = logger.level
            logger.setLevel(logging.WARNING)
    except Exception:
        _OAUTH_LOG_LOCK.release()
        raise
    return previous


def _restore_oauth_http_logging(previous: dict[str, int]) -> None:
    try:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)
    finally:
        _OAUTH_LOG_LOCK.release()


class DriveOAuthError(RuntimeError):
    """Safe, user-displayable OAuth failure without secret material."""


class ClientConfigError(DriveOAuthError):
    pass


class AuthorizationCancelled(DriveOAuthError):
    pass


@dataclass(frozen=True)
class DesktopClient:
    client_id: str
    client_secret: str

    def as_installed_config(self) -> dict[str, dict[str, str]]:
        return {
            "installed": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": AUTH_URI,
                "token_uri": TOKEN_URI,
            }
        }


def parse_desktop_client_json(raw: str | bytes | Mapping[str, Any]) -> DesktopClient:
    """Accept only Google's installed-client shape; endpoints are pinned."""
    try:
        if isinstance(raw, bytes):
            if len(raw) > MAX_CLIENT_JSON_BYTES:
                raise ClientConfigError("the OAuth client file is too large")
            value = json.loads(raw)
        elif isinstance(raw, str):
            if len(raw.encode("utf-8")) > MAX_CLIENT_JSON_BYTES:
                raise ClientConfigError("the OAuth client file is too large")
            value = json.loads(raw)
        else:
            value = dict(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ClientConfigError("the OAuth client file is malformed") from None
    if not isinstance(value, dict) or set(value) != {"installed"}:
        raise ClientConfigError("the OAuth client file must contain an installed desktop client")
    config = value["installed"]
    if not isinstance(config, dict):
        raise ClientConfigError("the OAuth client file is malformed")
    client_id = config.get("client_id")
    client_secret = config.get("client_secret", "")
    if not isinstance(client_id, str) or not client_id.strip():
        raise ClientConfigError("the OAuth client file has no client id")
    if not isinstance(client_secret, str):
        raise ClientConfigError("the OAuth client file is malformed")
    # Do not pass user-provided endpoints to the OAuth library.
    return DesktopClient(client_id.strip(), client_secret)


def _is_invalid_grant(exc: BaseException) -> bool:
    """Recognize only an explicit OAuth invalid_grant response."""
    # google-auth places the structured token response in the second
    # positional argument of RefreshError. Text-only errors are not enough.
    details = exc.args[1] if len(exc.args) > 1 else None
    return isinstance(details, Mapping) and details.get("error") == "invalid_grant"


class DriveOAuthService:
    """Serialized credential lifecycle with disconnect-wins semantics."""

    def __init__(self, store=keyring_store):
        self._store = store
        self._lock = threading.RLock()
        self._generation = self._read_generation()

    def _read_generation(self) -> int:
        return getattr(self._store, "drive_credential_generation", lambda: 0)()

    def _current_generation(self) -> int:
        return self._read_generation()

    def _bump_generation(self) -> int:
        bump = getattr(self._store, "bump_drive_credential_generation", None)
        if bump is not None:
            return bump()
        self._generation += 1
        return self._generation

    def state(self) -> keyring_store.DriveCredentialStoreState:
        with self._lock:
            return self._store.drive_credential_store_state()

    def credentials(self) -> Credentials | None:
        with self._lock:
            record = self._store.get_drive_credentials()
            if record is None:
                return None
            try:
                info = json.loads(record.serialized_credentials)
                if not isinstance(info, dict) or info.get("token_uri") != TOKEN_URI:
                    raise ValueError
                return Credentials.from_authorized_user_info(info, scopes=[DRIVE_READONLY_SCOPE])
            except (ValueError, TypeError, json.JSONDecodeError):
                raise DriveOAuthError("saved Google Drive credentials are unreadable") from None

    def save(
        self,
        credentials: Credentials,
        expected_version: int | None = None,
        expected_generation: int | None = None,
    ) -> int:
        with self._lock:
            if expected_version is None:
                current = self._store.get_drive_credentials()
                expected_version = current.version if current else 0
            return self._store.set_drive_credentials(
                credentials.to_json(), expected_version, expected_generation
            )

    def disconnect(self) -> None:
        with self._lock:
            self._bump_generation()
            self._store.delete_drive_credentials()

    def cancel_authorization(self) -> None:
        """Invalidate an in-flight authorization before it can persist tokens."""
        # Do not take the service lock: this method is intentionally callable
        # while authorize is between its final check and compare-and-set write.
        # The store generation lock makes the invalidation and the write atomic.
        self._bump_generation()
        self._store.delete_drive_credentials()

    def refresh(self, request: Any = None) -> Credentials:
        """Refresh and persist credentials; a disconnect invalidates the result."""
        with self._lock:
            generation = self._current_generation()
            record = self._store.get_drive_credentials()
            if record is None:
                raise DriveOAuthError("Google Drive is not connected")
            try:
                info = json.loads(record.serialized_credentials)
                if not isinstance(info, dict) or info.get("token_uri") != TOKEN_URI:
                    raise ValueError
                credentials = Credentials.from_authorized_user_info(
                    info, scopes=[DRIVE_READONLY_SCOPE]
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                raise DriveOAuthError("saved Google Drive credentials are unreadable") from None
            loaded_version = record.version
        try:
            previous_logging = _quiet_oauth_http_logging()
            try:
                credentials.refresh(request or google.auth.transport.requests.Request())
            finally:
                _restore_oauth_http_logging(previous_logging)
        except Exception as exc:
            if _is_invalid_grant(exc):
                with self._lock:
                    if generation == self._current_generation():
                        try:
                            self._store.delete_drive_credentials(loaded_version)
                        except keyring_store.DriveCredentialVersionConflict:
                            pass
                raise DriveOAuthError("Google Drive authorization has been revoked") from None
            raise DriveOAuthError("Google Drive authorization could not be refreshed") from None
        with self._lock:
            if generation != self._current_generation():
                raise DriveOAuthError("Google Drive authorization was disconnected")
            try:
                self.save(credentials, loaded_version)
            except keyring_store.DriveCredentialVersionConflict:
                raise DriveOAuthError("Google Drive authorization was changed") from None
            return credentials

    def authorize(
        self,
        client: DesktopClient,
        open_browser: Callable[[str], bool] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cancel: threading.Event | None = None,
        flow_factory: Callable[..., InstalledAppFlow] | None = None,
    ) -> Credentials:
        """Run one fixed-endpoint loopback flow and save the resulting tokens."""
        with self._lock:
            if self._store.drive_credential_store_state().status is keyring_store.DriveCredentialStatus.UNAVAILABLE:
                raise DriveOAuthError("an approved OS-backed keyring is unavailable")
            # Starting a new authorization supersedes any refresh/cleanup that
            # loaded the previous credential.  The shared tombstone also
            # coordinates separate service instances.
            generation = self._bump_generation()
        flow_factory = flow_factory or InstalledAppFlow.from_client_config
        flow = flow_factory(
            client.as_installed_config(),
            [DRIVE_READONLY_SCOPE],
            redirect_uri=f"http://127.0.0.1:0{CALLBACK_PATH}",
            autogenerate_code_verifier=True,
        )
        callback = _CallbackServer(timeout_seconds, cancel)
        try:
            callback.start()
            flow.redirect_uri = f"http://127.0.0.1:{callback.port}{CALLBACK_PATH}"
            previous_logging = _quiet_oauth_http_logging()
            try:
                auth_url, state = flow.authorization_url(
                    access_type="offline", prompt="consent", code_challenge_method="S256"
                )
                try:
                    browser_opened = (open_browser or webbrowser.open)(auth_url)
                except Exception:
                    browser_opened = False
                if not browser_opened:
                    raise AuthorizationCancelled("unable to open the system browser")
                callback.wait()
                if callback.error:
                    raise AuthorizationCancelled("Google authorization was cancelled")
                if not callback.code or callback.state != state:
                    raise DriveOAuthError("Google authorization response was invalid")
                response_url = callback.response_url
                if response_url is None:
                    raise DriveOAuthError("Google authorization response was invalid")
                if cancel and cancel.is_set():
                    raise AuthorizationCancelled("Google authorization was cancelled")
                try:
                    flow.fetch_token(
                    # oauthlib requires a secure transport for token exchange;
                    # the library's own loopback helper applies this conversion.
                    authorization_response=response_url.replace("http://", "https://", 1),
                    timeout=timeout_seconds,
                    )
                except Exception:
                    raise DriveOAuthError("Google authorization could not be completed") from None
            finally:
                _restore_oauth_http_logging(previous_logging)
            credentials = flow.credentials
            with self._lock:
                if cancel and cancel.is_set():
                    raise AuthorizationCancelled("Google authorization was cancelled")
                if generation != self._current_generation():
                    raise DriveOAuthError("Google Drive authorization was disconnected")
                try:
                    current = self._store.get_drive_credentials()
                    loaded_version = current.version if current else 0
                    self.save(credentials, loaded_version, generation)
                except keyring_store.DriveCredentialVersionConflict:
                    raise DriveOAuthError("Google Drive authorization was changed") from None
            return credentials
        finally:
            callback.close()


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        callback = getattr(self.server, "callback")
        if urlsplit(self.path).path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        if callback.path_seen:
            self.send_response(404)
            self.end_headers()
            return
        callback.path_seen = True
        query = parse_qs(urlsplit(self.path).query)
        callback.code = query.get("code", [None])[0]
        callback.state = query.get("state", [None])[0]
        callback.error = query.get("error", [None])[0]
        callback.response_url = callback.base_url + self.path
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Authorization complete. You may close this window.")

    def log_message(self, format, *args):  # noqa: A002, ARG002
        return


class _CallbackServer:
    def __init__(self, timeout: float, cancel: threading.Event | None):
        self.timeout, self.cancel = timeout, cancel
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.path_seen = False
        self.code = self.state = self.error = self.response_url = None

    @property
    def port(self):
        assert self.httpd is not None
        return self.httpd.server_port

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.httpd = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
        setattr(self.httpd, "callback", self)
        self.httpd.timeout = 0.2
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        assert self.httpd is not None
        deadline = time.monotonic() + self.timeout
        while not self.path_seen and time.monotonic() < deadline:
            if self.cancel and self.cancel.is_set():
                return
            self.httpd.handle_request()

    def wait(self):
        assert self.thread is not None
        self.thread.join(self.timeout + 1)
        if self.cancel and self.cancel.is_set():
            raise AuthorizationCancelled("Google authorization was cancelled")
        if not self.path_seen:
            raise AuthorizationCancelled("Google authorization timed out")

    def close(self):
        if self.httpd:
            self.httpd.server_close()
        if self.thread and self.thread.is_alive():
            self.thread.join(1)
