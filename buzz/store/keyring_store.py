import base64
import enum
import hashlib
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass

import keyring

from buzz.settings.settings import APP_NAME


# Some Secret Service backends can leave a keyring read blocked indefinitely.
# Keep the fallback bounded without making callers wait for the backend thread
# to finish after the timeout.
_KEYRING_READ_TIMEOUT_SECONDS = 5.0
_KEYRING_READ_LOCK = threading.Lock()
_KEYRING_READ_THREAD: threading.Thread | None = None


class Key(enum.Enum):
    OPENAI_API_KEY = "OpenAI API key"


class DriveCredentialStatus(enum.Enum):
    """The deliberately small state machine exposed by the Drive store."""

    UNCONFIGURED = "unconfigured"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class DriveCredentialStoreError(RuntimeError):
    """The OS keyring cannot safely be used for the Drive refresh token."""


class DriveCredentialVersionConflict(DriveCredentialStoreError):
    """A credential write was based on an older keyring value."""


DRIVE_CREDENTIAL_USERNAME = "Google Drive OAuth refresh credentials"
_DRIVE_KEYRING_TIMEOUT_SECONDS = 5.0
_DRIVE_KEYRING_LOCK = threading.RLock()
_DRIVE_CREDENTIAL_GENERATION = 0


@dataclass(frozen=True)
class DriveCredentialStoreState:
    status: DriveCredentialStatus
    reason: str | None = None


@dataclass(frozen=True)
class DriveCredentialRecord:
    serialized_credentials: str
    version: int


def drive_credential_generation() -> int:
    with _DRIVE_KEYRING_LOCK:
        return _DRIVE_CREDENTIAL_GENERATION


def bump_drive_credential_generation() -> int:
    global _DRIVE_CREDENTIAL_GENERATION
    with _DRIVE_KEYRING_LOCK:
        _DRIVE_CREDENTIAL_GENERATION += 1
        return _DRIVE_CREDENTIAL_GENERATION


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _drive_keyring_backend_name() -> str | None:
    """Return an explicitly approved OS-backed backend, or ``None``.

    This check is intentionally separate from the legacy secret API.  In
    particular, Linux's fail/plaintext backends and the portal/XOR store must
    never be considered suitable for an OAuth refresh credential.
    """
    try:
        backend = keyring.get_keyring()
        module = type(backend).__module__.lower()
        name = type(backend).__name__.lower()
        if _is_linux():
            if module == "keyring.backends.secretservice" and name == "keyring":
                return "secret-service"
            if module == "keyring.backends.kwallet" and name in {
                "dbuskeyring",
                "dbuskeyringkwallet4",
            }:
                return "kwallet"
            return None
        # Native OS keyrings are accepted on non-Linux platforms. Never accept
        # the explicit plaintext/fail implementations there either.
        if "fail" in module or "plaintext" in module:
            return None
        return f"{module}.{name}"
    except Exception:
        return None


def drive_credential_store_state() -> DriveCredentialStoreState:
    backend = _drive_keyring_backend_name()
    if backend is None:
        return DriveCredentialStoreState(
            DriveCredentialStatus.UNAVAILABLE,
            "an approved OS-backed keyring is unavailable",
        )
    try:
        value = _get_drive_keyring_value()
    except DriveCredentialStoreError:
        return DriveCredentialStoreState(DriveCredentialStatus.UNAVAILABLE, "keyring error")
    if value:
        return DriveCredentialStoreState(DriveCredentialStatus.AVAILABLE)
    return DriveCredentialStoreState(DriveCredentialStatus.UNCONFIGURED)


def get_drive_credentials() -> DriveCredentialRecord | None:
    """Read Drive credentials directly from the approved OS keyring."""
    if _drive_keyring_backend_name() is None:
        raise DriveCredentialStoreError("an approved OS-backed keyring is unavailable")
    value = _get_drive_keyring_value()
    if not value:
        return None
    try:
        envelope = json.loads(value)
        if (
            not isinstance(envelope, dict)
            or not isinstance(envelope.get("version"), int)
            or envelope["version"] < 1
            or not isinstance(envelope.get("credentials"), str)
        ):
            raise ValueError
        return DriveCredentialRecord(envelope["credentials"], envelope["version"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DriveCredentialStoreError("stored Drive credentials are unreadable") from exc


def _get_drive_keyring_value() -> str | None:
    """Read directly, with a bounded wait and errors distinguishable from unset."""
    result: list[str | None] = []
    error: list[BaseException] = []

    def read() -> None:
        try:
            result.append(keyring.get_password(APP_NAME, username=DRIVE_CREDENTIAL_USERNAME))
        except BaseException as exc:  # backend exceptions are intentionally opaque
            error.append(exc)

    with _DRIVE_KEYRING_LOCK:
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(_DRIVE_KEYRING_TIMEOUT_SECONDS)
    if reader.is_alive() or error:
        raise DriveCredentialStoreError("unable to read the OS keyring")
    return result[0] if result else None


def set_drive_credentials(
    serialized_credentials: str,
    expected_version: int | None = None,
    expected_generation: int | None = None,
) -> int:
    if _drive_keyring_backend_name() is None:
        raise DriveCredentialStoreError("an approved OS-backed keyring is unavailable")
    try:
        with _DRIVE_KEYRING_LOCK:
            current = get_drive_credentials()
            current_version = current.version if current else 0
            if expected_version is not None and current_version != expected_version:
                raise DriveCredentialVersionConflict("Drive credentials changed")
            if (
                expected_generation is not None
                and _DRIVE_CREDENTIAL_GENERATION != expected_generation
            ):
                raise DriveCredentialVersionConflict("Drive credentials were invalidated")
            version = current_version + 1
            envelope = json.dumps(
                {"version": version, "credentials": serialized_credentials},
                separators=(",", ":"),
            )
            keyring.set_password(APP_NAME, DRIVE_CREDENTIAL_USERNAME, envelope)
            return version
    except DriveCredentialVersionConflict:
        raise
    except Exception as exc:
        raise DriveCredentialStoreError("unable to write the OS keyring") from exc


def delete_drive_credentials(expected_version: int | None = None) -> bool:
    if _drive_keyring_backend_name() is None:
        raise DriveCredentialStoreError("an approved OS-backed keyring is unavailable")
    try:
        with _DRIVE_KEYRING_LOCK:
            current = get_drive_credentials()
            if current is None:
                return False
            if expected_version is not None and current.version != expected_version:
                raise DriveCredentialVersionConflict("Drive credentials changed")
            keyring.delete_password(APP_NAME, DRIVE_CREDENTIAL_USERNAME)
            return True
    except keyring.errors.PasswordDeleteError:
        return False
    except DriveCredentialVersionConflict:
        raise
    except Exception as exc:
        raise DriveCredentialStoreError("unable to delete from the OS keyring") from exc


def _get_secrets_file_path() -> str:
    """Get the path to the local encrypted secrets file."""
    from platformdirs import user_data_dir

    data_dir = user_data_dir(APP_NAME)
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, ".secrets.json")


def _get_portal_secret() -> bytes | None:
    """Get the application secret from XDG Desktop Portal.

    The portal provides a per-application secret that can be used
    for encrypting application-specific data. This works in sandboxed
    environments (Snap/Flatpak) via the desktop plug.
    """
    if not _is_linux():
        return None

    try:
        from jeepney import DBusAddress, new_method_call
        from jeepney.io.blocking import open_dbus_connection
        import socket

        # Open connection with file descriptor support enabled
        conn = open_dbus_connection(bus="SESSION", enable_fds=True)

        portal = DBusAddress(
            "/org/freedesktop/portal/desktop",
            bus_name="org.freedesktop.portal.Desktop",
            interface="org.freedesktop.portal.Secret",
        )

        # Create a socket pair for receiving the secret
        sock_read, sock_write = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            # Build the method call with file descriptor
            # RetrieveSecret(fd: h, options: a{sv}) -> (handle: o)
            # Pass the socket object directly - jeepney handles fd passing
            msg = new_method_call(portal, "RetrieveSecret", "ha{sv}", (sock_write, {}))

            # Send message and get reply
            conn.send_and_get_reply(msg, timeout=10)

            # Close the write end - portal has it now
            sock_write.close()
            sock_write = None

            # Read the secret from the read end
            # The portal writes the secret and closes its end
            sock_read.settimeout(5.0)
            secret_data = b""
            while True:
                try:
                    chunk = sock_read.recv(4096)
                    if not chunk:
                        break
                    secret_data += chunk
                except socket.timeout:
                    break

            if secret_data:
                return secret_data

            return None

        finally:
            sock_read.close()
            if sock_write is not None:
                sock_write.close()

    except Exception as exc:
        logging.debug("XDG Portal secret not available: %s", exc)
        return None


def _derive_key(master_secret: bytes, key_name: str) -> bytes:
    """Derive a key-specific encryption key from the master secret."""
    # Use PBKDF2 to derive a key for this specific secret
    return hashlib.pbkdf2_hmac(
        "sha256",
        master_secret,
        f"{APP_NAME}:{key_name}".encode(),
        100000,
        dklen=32,
    )


def _encrypt_value(value: str, key: bytes) -> str:
    """Encrypt a value using XOR with the derived key (simple encryption)."""
    # For a more secure implementation, use cryptography library with AES
    # This is a simple XOR-based encryption suitable for the use case
    value_bytes = value.encode("utf-8")
    key_extended = (key * ((len(value_bytes) // len(key)) + 1))[: len(value_bytes)]
    encrypted = bytes(a ^ b for a, b in zip(value_bytes, key_extended))
    return base64.b64encode(encrypted).decode("ascii")


def _decrypt_value(encrypted: str, key: bytes) -> str:
    """Decrypt a value using XOR with the derived key."""
    encrypted_bytes = base64.b64decode(encrypted.encode("ascii"))
    key_extended = (key * ((len(encrypted_bytes) // len(key)) + 1))[: len(encrypted_bytes)]
    decrypted = bytes(a ^ b for a, b in zip(encrypted_bytes, key_extended))
    return decrypted.decode("utf-8")


def _load_local_secrets() -> dict:
    """Load the local secrets file."""
    secrets_file = _get_secrets_file_path()
    if os.path.exists(secrets_file):
        try:
            with open(secrets_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logging.debug("Failed to load secrets file: %s", exc)
    return {}


def _save_local_secrets(secrets: dict) -> None:
    """Save secrets to the local file."""
    secrets_file = _get_secrets_file_path()
    try:
        with open(secrets_file, "w") as f:
            json.dump(secrets, f)
        # Set restrictive permissions
        os.chmod(secrets_file, 0o600)
    except IOError as exc:
        logging.warning("Failed to save secrets file: %s", exc)


def _get_portal_password(key: Key) -> str | None:
    """Get a password using the XDG Desktop Portal Secret."""
    portal_secret = _get_portal_secret()
    if portal_secret is None:
        return None

    secrets = _load_local_secrets()
    encrypted_value = secrets.get(key.value)
    if encrypted_value is None:
        return None

    try:
        derived_key = _derive_key(portal_secret, key.value)
        return _decrypt_value(encrypted_value, derived_key)
    except Exception as exc:
        logging.debug("Failed to decrypt portal secret: %s", exc)
        return None


def _set_portal_password(key: Key, password: str) -> bool:
    """Set a password using the XDG Desktop Portal Secret."""
    portal_secret = _get_portal_secret()
    if portal_secret is None:
        return False

    try:
        derived_key = _derive_key(portal_secret, key.value)
        encrypted_value = _encrypt_value(password, derived_key)

        secrets = _load_local_secrets()
        secrets[key.value] = encrypted_value
        _save_local_secrets(secrets)
        return True
    except Exception as exc:
        logging.debug("Failed to set portal secret: %s", exc)
        return False


def _delete_portal_password(key: Key) -> bool:
    """Delete a password from the portal-based local storage."""
    secrets = _load_local_secrets()
    if key.value in secrets:
        del secrets[key.value]
        _save_local_secrets(secrets)
        return True
    return False


def _get_keyring_password(username: str) -> str | None:
    """Read a keyring password without allowing a backend stall to block us."""
    global _KEYRING_READ_THREAD

    with _KEYRING_READ_LOCK:
        if _KEYRING_READ_THREAD is not None:
            if _KEYRING_READ_THREAD.is_alive():
                logging.warning(
                    "Keyring password read already in progress; returning empty fallback"
                )
                return None
            _KEYRING_READ_THREAD = None

    result: list[str | None] = []
    error: list[Exception] = []

    def read_password() -> None:
        global _KEYRING_READ_THREAD
        try:
            result.append(keyring.get_password(APP_NAME, username=username))
        except Exception as exc:
            error.append(exc)
        finally:
            with _KEYRING_READ_LOCK:
                if _KEYRING_READ_THREAD is threading.current_thread():
                    _KEYRING_READ_THREAD = None

    reader = threading.Thread(target=read_password, daemon=True)
    with _KEYRING_READ_LOCK:
        # Re-check under the lock so concurrent callers cannot start another
        # backend operation between the initial check and thread creation.
        if _KEYRING_READ_THREAD is not None:
            logging.warning(
                "Keyring password read already in progress; returning empty fallback"
            )
            return None
        _KEYRING_READ_THREAD = reader
        reader.start()
    reader.join(_KEYRING_READ_TIMEOUT_SECONDS)

    if reader.is_alive():
        logging.warning(
            "Keyring password read timed out after %.1f seconds; returning empty fallback",
            _KEYRING_READ_TIMEOUT_SECONDS,
        )
        return None

    if error:
        logging.warning(
            "Unable to read from keyring (%s); returning empty fallback",
            type(error[0]).__name__,
        )
        return None

    return result[0] if result else None


def get_password(key: Key) -> str | None:
    # On Linux, try XDG Desktop Portal first (works in sandboxed environments)
    if _is_linux():
        result = _get_portal_password(key)


        if result is not None:
            return result

    # Fall back to keyring (cross-platform, uses Secret Service on Linux)
    password = _get_keyring_password(key.value)
    return password if password is not None else ""


def set_password(username: Key, password: str) -> None:
    # On Linux, try XDG Desktop Portal first (works in sandboxed environments)
    if _is_linux():
        if _set_portal_password(username, password):
            return

    # Fall back to keyring (cross-platform, uses Secret Service on Linux)
    keyring.set_password(APP_NAME, username.value, password)


def delete_password(key: Key) -> None:
    """Delete a password from the secret store."""
    # On Linux, also delete from portal storage
    if _is_linux():
        _delete_portal_password(key)

    # Delete from keyring
    try:
        keyring.delete_password(APP_NAME, key.value)
    except keyring.errors.PasswordDeleteError:
        pass  # Password doesn't exist, ignore
    except Exception as exc:
        logging.warning("Unable to delete from keyring: %s", exc)


# --- String-keyed secrets -------------------------------------------------
# The Key enum above is for built-in, statically-known secrets. Plugins need
# secrets under dynamic names (e.g. "plugin:<id>:<field>"), so the following
# helpers accept a raw string name while reusing the same portal + keyring
# fallback storage as the enum-based API.


def _get_portal_secret_by_name(name: str) -> str | None:
    portal_secret = _get_portal_secret()
    if portal_secret is None:
        return None
    secrets = _load_local_secrets()
    encrypted_value = secrets.get(name)
    if encrypted_value is None:
        return None
    try:
        derived_key = _derive_key(portal_secret, name)
        return _decrypt_value(encrypted_value, derived_key)
    except Exception as exc:
        logging.debug("Failed to decrypt portal secret: %s", exc)
        return None


def _set_portal_secret_by_name(name: str, password: str) -> bool:
    portal_secret = _get_portal_secret()
    if portal_secret is None:
        return False
    try:
        derived_key = _derive_key(portal_secret, name)
        secrets = _load_local_secrets()
        secrets[name] = _encrypt_value(password, derived_key)
        _save_local_secrets(secrets)
        return True
    except Exception as exc:
        logging.debug("Failed to set portal secret: %s", exc)
        return False


def get_secret(name: str) -> str:
    """Get a secret stored under an arbitrary string name. Returns "" if unset."""
    if _is_linux():
        result = _get_portal_secret_by_name(name)
        if result is not None:
            return result
    password = _get_keyring_password(name)
    return password if password is not None else ""


def set_secret(name: str, password: str) -> None:
    """Set a secret stored under an arbitrary string name."""
    if _is_linux():
        if _set_portal_secret_by_name(name, password):
            return
    keyring.set_password(APP_NAME, name, password)


def delete_secret(name: str) -> None:
    """Delete a secret stored under an arbitrary string name."""
    if _is_linux():
        secrets = _load_local_secrets()
        if name in secrets:
            del secrets[name]
            _save_local_secrets(secrets)
    try:
        keyring.delete_password(APP_NAME, name)
    except keyring.errors.PasswordDeleteError:
        pass
    except Exception as exc:
        logging.warning("Unable to delete from keyring: %s", exc)
