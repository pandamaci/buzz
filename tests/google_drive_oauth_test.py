import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from buzz.google_drive_oauth import (
    AUTH_URI,
    TOKEN_URI,
    ClientConfigError,
    DesktopClient,
    DriveOAuthError,
    DriveOAuthService,
    parse_desktop_client_json,
    _CallbackServer,
)
from buzz.store import keyring_store


def client_json(**extra):
    config = {"client_id": "desktop-id", "client_secret": "desktop-secret"}
    config.update(extra)
    return json.dumps({"installed": config})


def credentials():
    return Credentials(
        token="access",
        refresh_token="refresh",
        token_uri=TOKEN_URI,
        client_id="desktop-id",
        client_secret="desktop-secret",
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )


class MemoryStore:
    def __init__(self, value=None):
        self.value = value
        self.deleted = 0
        self.generation = 0

    def drive_credential_generation(self):
        return self.generation

    def bump_drive_credential_generation(self):
        self.generation += 1
        return self.generation

    def drive_credential_store_state(self):
        return SimpleNamespace(status="available" if self.value else "unconfigured")

    def get_drive_credentials(self):
        if self.value is None:
            return None
        return keyring_store.DriveCredentialRecord(self.value, getattr(self, "version", 1))

    def set_drive_credentials(self, value, expected_version=None, expected_generation=None):
        current = self.get_drive_credentials()
        current_version = current.version if current else 0
        if expected_version is not None and expected_version != current_version:
            raise keyring_store.DriveCredentialVersionConflict
        if expected_generation is not None and expected_generation != self.generation:
            raise keyring_store.DriveCredentialVersionConflict
        self.value = value
        self.version = current_version + 1
        return self.version

    def delete_drive_credentials(self, expected_version=None):
        current = self.get_drive_credentials()
        if current is None:
            return False
        if expected_version is not None and expected_version != current.version:
            raise keyring_store.DriveCredentialVersionConflict
        self.deleted += 1
        self.value = None
        return True


def test_accepts_installed_client_and_pins_endpoints():
    client = parse_desktop_client_json(client_json(auth_uri="https://evil.invalid", token_uri="http://evil.invalid"))
    assert client == DesktopClient("desktop-id", "desktop-secret")
    assert client.as_installed_config()["installed"]["auth_uri"] == AUTH_URI
    assert client.as_installed_config()["installed"]["token_uri"] == TOKEN_URI


@pytest.mark.parametrize(
    "value",
    ["{}", "{\"web\": {}}", "not-json", "{\"installed\": []}", client_json(client_id="")],
)
def test_rejects_malformed_or_non_desktop_client(value):
    with pytest.raises(ClientConfigError):
        parse_desktop_client_json(value)


def test_linux_drive_store_rejects_plaintext_and_fail_backends():
    for module, name in (
        ("keyring.backends.fail", "Keyring"),
        ("keyring.backends.libsecret", "Keyring"),
        ("keyring.backends.SecretService", "NotKeyring"),
    ):
        backend = type(name, (), {})()
        backend.__class__.__module__ = module
        with patch.object(keyring_store, "_is_linux", return_value=True), patch.object(
            keyring_store.keyring, "get_keyring", return_value=backend
        ):
            assert keyring_store._drive_keyring_backend_name() is None


def test_linux_drive_store_accepts_secret_service():
    backend = type("Keyring", (), {})()
    backend.__class__.__module__ = "keyring.backends.secretservice"
    with patch.object(keyring_store, "_is_linux", return_value=True), patch.object(
        keyring_store.keyring, "get_keyring", return_value=backend
    ):
        assert keyring_store._drive_keyring_backend_name() == "secret-service"


def test_linux_drive_store_accepts_real_kwallet_variants():
    for name in ("DBusKeyring", "DBusKeyringKWallet4"):
        backend = type(name, (), {})()
        backend.__class__.__module__ = "keyring.backends.kwallet"
        with patch.object(keyring_store, "_is_linux", return_value=True), patch.object(
            keyring_store.keyring, "get_keyring", return_value=backend
        ):
            assert keyring_store._drive_keyring_backend_name() == "kwallet"


def test_drive_store_state_distinguishes_unconfigured_and_unavailable():
    backend = type("Keyring", (), {})()
    backend.__class__.__module__ = "keyring.backends.secretservice"
    with patch.object(keyring_store, "_is_linux", return_value=True), patch.object(
        keyring_store.keyring, "get_keyring", return_value=backend
    ), patch.object(keyring_store, "_get_drive_keyring_value", return_value=None):
        state = keyring_store.drive_credential_store_state()
    assert state.status is keyring_store.DriveCredentialStatus.UNCONFIGURED

    with patch.object(keyring_store, "_drive_keyring_backend_name", return_value=None):
        state = keyring_store.drive_credential_store_state()
    assert state.status is keyring_store.DriveCredentialStatus.UNAVAILABLE


def test_drive_store_never_uses_generic_secret_api():
    with patch.object(keyring_store, "_drive_keyring_backend_name", return_value=None), patch.object(
        keyring_store, "get_secret", side_effect=AssertionError
    ), patch.object(keyring_store, "set_secret", side_effect=AssertionError):
        with pytest.raises(keyring_store.DriveCredentialStoreError):
            keyring_store.get_drive_credentials()


def test_refresh_invalid_grant_deletes_credentials():
    store = MemoryStore(credentials().to_json())
    service = DriveOAuthService(store)
    error = RefreshError("invalid_grant", {"error": "invalid_grant"})
    with patch.object(Credentials, "refresh", side_effect=error):
        with pytest.raises(DriveOAuthError, match="revoked"):
            service.refresh()
    assert store.value is None
    assert store.deleted == 1


def test_refresh_non_invalid_error_preserves_credentials():
    store = MemoryStore(credentials().to_json())
    service = DriveOAuthService(store)
    with patch.object(Credentials, "refresh", side_effect=RefreshError("temporarily unavailable")):
        with pytest.raises(DriveOAuthError):
            service.refresh()
    assert store.value is not None
    assert store.deleted == 0


def test_unstructured_invalid_grant_text_preserves_credentials():
    store = MemoryStore(credentials().to_json())
    service = DriveOAuthService(store)
    with patch.object(Credentials, "refresh", side_effect=RefreshError("invalid_grant")):
        with pytest.raises(DriveOAuthError):
            service.refresh()
    assert store.value is not None
    assert store.deleted == 0


def test_malicious_stored_token_uri_fails_before_refresh():
    invalid = json.loads(credentials().to_json())
    invalid["token_uri"] = "https://evil.example/token"
    store = MemoryStore(json.dumps(invalid))
    service = DriveOAuthService(store)
    refresh = Mock()
    with patch.object(Credentials, "refresh", refresh):
        with pytest.raises(DriveOAuthError):
            service.refresh()
    refresh.assert_not_called()


def test_stored_token_uri_is_exactly_pinned():
    invalid = json.loads(credentials().to_json())
    invalid["token_uri"] = "https://oauth2.googleapis.com/token/"
    store = MemoryStore(json.dumps(invalid))
    with pytest.raises(DriveOAuthError):
        DriveOAuthService(store).credentials()


def test_oauth_http_loggers_are_quiet_during_refresh(caplog):
    store = MemoryStore(credentials().to_json())
    service = DriveOAuthService(store)
    noisy = logging.getLogger("oauthlib")
    noisy.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        with patch.object(Credentials, "refresh", side_effect=RuntimeError("token-secret")):
            with pytest.raises(DriveOAuthError):
                service.refresh()
    assert "token-secret" not in caplog.text


def test_authorization_log_capture_contains_no_auth_url_state_or_token(caplog):
    store = MemoryStore()
    service = DriveOAuthService(store)
    flow = Mock()
    flow.authorization_url.return_value = (
        "https://accounts.google.com/auth?state=secret-state&client_id=client-secret",
        "secret-state",
    )
    flow.credentials = credentials()
    callback = Mock(
        port=44001, code="authorization-code-secret", state="secret-state", error=None,
        response_url="http://127.0.0.1:44001/oauth2callback?code=authorization-code-secret&state=secret-state",
    )
    with patch("buzz.google_drive_oauth._CallbackServer", return_value=callback), patch(
        "buzz.google_drive_oauth.webbrowser.open", return_value=True
    ), caplog.at_level(logging.DEBUG):
        service.authorize(DesktopClient("client-secret", "client-secret"), flow_factory=Mock(return_value=flow))
    assert "accounts.google.com/auth" not in caplog.text
    assert "secret-state" not in caplog.text
    assert "authorization-code-secret" not in caplog.text
    assert "client-secret" not in caplog.text


def test_cross_instance_authorize_disconnect_cannot_persist():
    store = MemoryStore()
    first = DriveOAuthService(store)
    second = DriveOAuthService(store)
    flow = Mock()
    flow.authorization_url.return_value = ("https://accounts.google.com/auth", "state")
    flow.credentials = credentials()
    callback = Mock(port=44002, code="code", state="state", error=None,
                    response_url="http://127.0.0.1:44002/oauth2callback?code=code&state=state")
    exchange_started = threading.Event()
    release_exchange = threading.Event()

    def blocked_exchange(**_kwargs):
        exchange_started.set()
        release_exchange.wait(2)

    flow.fetch_token.side_effect = blocked_exchange
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            pytest.raises(DriveOAuthError, first.authorize, DesktopClient("id", "secret"),
                          flow_factory=Mock(return_value=flow))
        )
    )
    with patch("buzz.google_drive_oauth._CallbackServer", return_value=callback), patch(
        "buzz.google_drive_oauth.webbrowser.open", return_value=True
    ):
        worker.start()
        assert exchange_started.wait(1)
        second.disconnect()
        release_exchange.set()
        worker.join(2)
    assert store.value is None


def test_cancel_authorization_between_check_and_persist_cannot_write():
    store = MemoryStore()
    service = DriveOAuthService(store)
    flow = Mock()
    flow.authorization_url.return_value = ("https://accounts.google.com/auth", "state")
    flow.credentials = credentials()
    callback = Mock(port=44003, code="code", state="state", error=None,
                    response_url="http://127.0.0.1:44003/oauth2callback?code=code&state=state")
    pre_save = threading.Event()
    release_save = threading.Event()
    original_save = service.save

    def barrier_save(*args, **kwargs):
        pre_save.set()
        release_save.wait(2)
        return original_save(*args, **kwargs)

    service.save = barrier_save
    result = []

    def run():
        try:
            service.authorize(
                DesktopClient("id", "secret"),
                flow_factory=Mock(return_value=flow),
            )
        except Exception as exc:
            result.append(exc)

    with patch("buzz.google_drive_oauth._CallbackServer", return_value=callback), patch(
        "buzz.google_drive_oauth.webbrowser.open", return_value=True
    ):
        worker = threading.Thread(target=run)
        worker.start()
        assert pre_save.wait(1)
        service.cancel_authorization()
        release_save.set()
        worker.join(2)
    assert result and isinstance(result[0], DriveOAuthError)
    assert store.value is None


def test_disconnect_wins_over_inflight_refresh():
    store = MemoryStore(credentials().to_json())
    service = DriveOAuthService(store)
    started = threading.Event()
    release = threading.Event()

    def blocked_refresh(_self, _request):
        started.set()
        release.wait(2)

    with patch.object(Credentials, "refresh", blocked_refresh):
        result = []
        def run_refresh():
            try:
                service.refresh()
            except Exception as exc:
                result.append(exc)

        worker = threading.Thread(target=run_refresh)
        worker.start()
        assert started.wait(1)
        service.disconnect()
        release.set()
        worker.join(2)
    assert store.value is None
    assert result and isinstance(result[0], DriveOAuthError)


def test_authorization_does_not_log_or_accept_callback_twice():
    # The callback server is exercised without network access to Google; the
    # flow itself is replaced with a deterministic fake.
    fake_credentials = credentials()
    flow = Mock()
    flow.authorization_url.return_value = ("https://accounts.google.com/private", "state")
    flow.credentials = fake_credentials
    flow_factory = Mock(return_value=flow)
    store = MemoryStore()
    service = DriveOAuthService(store)
    with patch("buzz.google_drive_oauth._CallbackServer") as server_type:
        server = server_type.return_value
        server.port = 43210
        server.code = "code"
        server.state = "state"
        server.error = None
        server.response_url = "http://127.0.0.1:43210/oauth2callback?code=code&state=state"
        with patch("buzz.google_drive_oauth.webbrowser.open", return_value=True):
            service.authorize(DesktopClient("id", "secret"), flow_factory=flow_factory)
    assert store.value is not None
    assert "private" not in ""  # URL is never logged by the service.


def _fake_authorization(service, store, flow, callback, done=None):
    factory = Mock(return_value=flow)
    callback_type = patch("buzz.google_drive_oauth._CallbackServer", return_value=callback)
    browser = patch("buzz.google_drive_oauth.webbrowser.open", return_value=True)
    return factory, callback_type, browser


def test_authorize_wins_against_old_refresh_write():
    store = MemoryStore(credentials().to_json())
    service = DriveOAuthService(store)
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    refresh_error = []
    new_credentials = credentials()
    new_credentials.token = "authorized-access"

    def blocked_refresh(_self, _request):
        refresh_started.set()
        release_refresh.wait(2)

    def run_refresh():
        try:
            service.refresh()
        except Exception as exc:
            refresh_error.append(exc)

    flow = Mock()
    flow.authorization_url.return_value = ("https://accounts.google.com/auth", "state")
    flow.credentials = new_credentials
    callback = Mock(port=40001, code="code", state="state", error=None,
                    response_url="http://127.0.0.1:40001/oauth2callback?code=code&state=state")
    refresh_thread = threading.Thread(target=run_refresh)
    with patch.object(Credentials, "refresh", blocked_refresh):
        refresh_thread.start()
        assert refresh_started.wait(1)
        factory, callback_type, browser = _fake_authorization(service, store, flow, callback)
        with callback_type, browser:
            service.authorize(DesktopClient("id", "secret"), flow_factory=factory)
        release_refresh.set()
        refresh_thread.join(2)
    assert refresh_error and isinstance(refresh_error[0], DriveOAuthError)
    assert json.loads(store.value)["token"] == "authorized-access"


def test_authorize_wins_against_old_invalid_grant_cleanup():
    store = MemoryStore(credentials().to_json())
    service = DriveOAuthService(store)
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    refresh_error = []
    new_credentials = credentials()
    new_credentials.token = "authorized-access"

    def blocked_invalid_grant(_self, _request):
        refresh_started.set()
        release_refresh.wait(2)
        raise RefreshError("invalid_grant", {"error": "invalid_grant"})

    def run_refresh():
        try:
            service.refresh()
        except Exception as exc:
            refresh_error.append(exc)

    flow = Mock()
    flow.authorization_url.return_value = ("https://accounts.google.com/auth", "state")
    flow.credentials = new_credentials
    callback = Mock(port=40002, code="code", state="state", error=None,
                    response_url="http://127.0.0.1:40002/oauth2callback?code=code&state=state")
    refresh_thread = threading.Thread(target=run_refresh)
    with patch.object(Credentials, "refresh", blocked_invalid_grant):
        refresh_thread.start()
        assert refresh_started.wait(1)
        factory, callback_type, browser = _fake_authorization(service, store, flow, callback)
        with callback_type, browser:
            service.authorize(DesktopClient("id", "secret"), flow_factory=factory)
        release_refresh.set()
        refresh_thread.join(2)
    assert refresh_error and isinstance(refresh_error[0], DriveOAuthError)
    assert json.loads(store.value)["token"] == "authorized-access"


def test_unavailable_storage_starts_no_listener_or_browser():
    service = DriveOAuthService(MemoryStore())
    service._store.drive_credential_store_state = lambda: SimpleNamespace(
        status=keyring_store.DriveCredentialStatus.UNAVAILABLE
    )
    with patch("buzz.google_drive_oauth._CallbackServer") as callback, patch(
        "buzz.google_drive_oauth.webbrowser.open"
    ) as browser:
        with pytest.raises(DriveOAuthError):
            service.authorize(DesktopClient("id", "secret"))
    callback.assert_not_called()
    browser.assert_not_called()


def test_wrong_callback_path_is_ignored():
    server = _CallbackServer(1, None)
    server.start()
    try:
        import requests

        response = requests.get(f"{server.base_url}/wrong?code=bad", timeout=1)
        assert response.status_code == 404
        assert not server.path_seen
        response = requests.get(
            f"{server.base_url}/oauth2callback?code=good&state=ok", timeout=1
        )
        assert response.status_code == 200
        assert server.path_seen
    finally:
        server.close()


def test_authorization_completion_server_exits_cleanly():
    server = _CallbackServer(2, None)
    server.start()
    import requests

    requests.get(f"{server.base_url}/oauth2callback?code=good&state=ok", timeout=1)
    server.wait()
    server.close()
    assert server.thread is not None and not server.thread.is_alive()
