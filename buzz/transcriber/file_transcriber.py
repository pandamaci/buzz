import logging
import os
import re
import sys
import subprocess
import shutil
import tempfile
import threading
from abc import abstractmethod
from html.parser import HTMLParser
from typing import Optional, List
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, unquote, urlencode, urljoin, urlparse, urlsplit, urlunsplit
from urllib.request import HTTPCookieProcessor, HTTPRedirectHandler, Request, build_opener
from http.cookiejar import CookieJar

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from yt_dlp import YoutubeDL

from buzz import whisper_audio
from buzz.assets import APP_BASE_DIR
from buzz.google_drive_downloader import (
    DriveAuthenticationError,
    DriveAuthorizationError,
    DriveCancelledError,
    DriveCredentialsUnavailableError,
    DriveCredentialsUnconfiguredError,
    DriveDownloadDeniedError,
    DriveDownloadError,
    DriveDownloader,
    DriveNotFoundError,
    DriveRedirectError,
    DriveStorageUnavailableError,
    DriveTransientError,
    DriveUnsupportedTypeError,
    parse_drive_url,
)
from buzz.google_drive_oauth import DriveOAuthService
from buzz.store import keyring_store
from buzz.transcriber.transcriber import (
    FileTranscriptionTask,
    get_output_file_path,
    Segment,
    OutputFormat,
)

app_env = os.environ.copy()
app_env['PATH'] = os.pathsep.join([os.path.join(APP_BASE_DIR, "_internal")] + [app_env['PATH']])

GOOGLE_DRIVE_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?id={file_id}&export=download"
)
GOOGLE_DRIVE_DOWNLOAD_HOSTS = {
    "drive.google.com",
    "www.drive.google.com",
    "drive.usercontent.google.com",
}
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_CONFIRMATION_PAGE_SIZE = 1024 * 1024


def _redact_drive_log_text(value) -> str:
    text = str(value)
    # Parse URL-looking portions rather than matching one particular URL
    # spelling.  In particular, urlsplit().hostname still identifies a Drive
    # host when the URL has userinfo or a malformed port; parsed.netloc must
    # not be copied because it may contain either of those secrets.
    result = []
    position = 0
    lower_text = text.lower()
    while position < len(text):
        starts = [
            index
            for index in (
                lower_text.find("http://", position),
                lower_text.find("https://", position),
            )
            if index >= 0
        ]
        if not starts:
            result.append(text[position:])
            break

        start = min(starts)
        end = start
        while end < len(text) and text[end] not in " \t\r\n'\"<>":
            end += 1
        candidate = text[start:end]
        parsed = None
        try:
            parsed = urlsplit(candidate)
            hostname = (parsed.hostname or "").lower()
        except ValueError:
            hostname = ""

        result.append(text[position:start])
        if parsed is not None and hostname in GOOGLE_DRIVE_DOWNLOAD_HOSTS:
            # Deliberately reconstruct a scheme-free identifier from the
            # hostname and path: query, fragment, userinfo, ports, and scheme
            # are never retained.
            result.append(hostname + parsed.path)
        else:
            result.append(candidate)
        position = end

    return "".join(result)


class _DriveRedactingLogger:
    def _log(self, level, message, *args, **kwargs):
        # yt-dlp commonly supplies URLs as %-format arguments.  Redacting only
        # the format string would leave those arguments visible to the root
        # logger, so render first and redact the complete message.
        if args:
            try:
                message = str(message) % args
            except Exception:
                message = " ".join(str(value) for value in (message, *args))
        # A traceback can contain the original exception (and its URL), which
        # cannot be safely redacted by the URL parser.  Drive diagnostics keep
        # the message but omit that optional traceback.
        kwargs.pop("exc_info", None)
        logging.getLogger().log(level, _redact_drive_log_text(message), **kwargs)

    def debug(self, message, *args, **kwargs):
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._log(logging.ERROR, message, *args, **kwargs)


class _GoogleDriveConfirmationParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms = []
        self.links = []
        self._form = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag.lower() == "form":
            if self._form is not None:
                self._form = None
                return
            self._form = {
                "action": attributes.get("action"),
                "method": (attributes.get("method") or "get").lower(),
                "hidden": [],
            }
        elif tag.lower() == "input" and self._form is not None:
            if (attributes.get("type") or "text").lower() == "hidden":
                name = attributes.get("name")
                if name:
                    self._form["hidden"].append((name, attributes.get("value", "")))
        elif tag.lower() == "a" and attributes.get("href"):
            self.links.append(attributes["href"])

    def handle_endtag(self, tag):
        if tag.lower() == "form" and self._form is not None:
            self.forms.append(self._form)
            self._form = None

    @property
    def malformed(self):
        return self._form is not None


def _is_allowed_google_drive_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() in GOOGLE_DRIVE_DOWNLOAD_HOSTS


class _GoogleDriveRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirect_url = urljoin(req.full_url, newurl)
        if not _is_allowed_google_drive_url(redirect_url):
            raise ValueError("Google Drive download redirected to an untrusted host")
        return super().redirect_request(req, fp, code, msg, headers, redirect_url)


def extract_google_drive_file_id(url: Optional[str]) -> Optional[str]:
    """Return the file ID from a supported public Google Drive URL."""
    if not url:
        return None

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        return None

    hostname = (parsed_url.hostname or "").lower()
    if hostname not in {"drive.google.com", "www.drive.google.com"}:
        return None

    file_match = re.search(r"/file/d/([^/?#]+)", parsed_url.path)
    file_id = file_match.group(1) if file_match else None
    if file_id is None and parsed_url.path.rstrip("/") == "/open":
        file_id = parse_qs(parsed_url.query).get("id", [None])[0]
    if file_id is None:
        return None

    file_id = unquote(file_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        return None
    return file_id


class FileTranscriber(QObject):
    transcription_task: FileTranscriptionTask
    progress = pyqtSignal(tuple)  # (current, total)
    download_progress = pyqtSignal(float)
    completed = pyqtSignal(list)  # List[Segment]
    error = pyqtSignal(str)

    def __init__(self, task: FileTranscriptionTask, parent: Optional["QObject"] = None):
        super().__init__(parent)
        self.transcription_task = task
        self._drive_oauth_service = DriveOAuthService()
        self._cancel_event = threading.Event()

    @pyqtSlot()
    def run(self):
        if self.transcription_task.source == FileTranscriptionTask.Source.URL_IMPORT:
            if not self._download_from_url():
                return

        try:
            segments = self.transcribe()
        except Exception as exc:
            logging.exception("")
            self.error.emit(str(exc))
            return

        for segment in segments:
            segment.text = segment.text.strip()

        self.completed.emit(segments)

        for (
            output_format
        ) in self.transcription_task.file_transcription_options.output_formats:
            default_path = get_output_file_path(
                file_path=self.transcription_task.file_path,
                output_format=output_format,
                language=self.transcription_task.transcription_options.language,
                output_directory=self.transcription_task.output_directory,
                model=self.transcription_task.transcription_options.model,
                task=self.transcription_task.transcription_options.task,
            )

            write_output(
                path=default_path, segments=segments, output_format=output_format
            )

        if self.transcription_task.source == FileTranscriptionTask.Source.FOLDER_WATCH:
            self._handle_folder_watch()

    def _download_from_url(self) -> bool:
        cookiefile = os.getenv("BUZZ_DOWNLOAD_COOKIEFILE")
        drive_reference = None
        drive_candidate = bool(
            self.transcription_task.url
            and self._is_drive_host_candidate(self.transcription_task.url)
        )
        try:
            drive_reference = parse_drive_url(self.transcription_task.url or "")
        except DriveDownloadError:
            if drive_candidate:
                try:
                    if self._drive_credentials_are_configured():
                        self.error.emit("Google Drive URL is invalid.")
                        return False
                except DriveDownloadError as exc:
                    self.error.emit(self._drive_download_error_message(exc))
                    return False
        drive_file_id = drive_reference.file_id if drive_reference else extract_google_drive_file_id(
            self.transcription_task.url
        )

        if drive_reference:
            try:
                configured = self._drive_credentials_are_configured()
            except DriveDownloadError as exc:
                self.error.emit(self._drive_download_error_message(exc))
                return False
            if configured:
                if drive_file_id is None:
                    self.error.emit("Google Drive URL is invalid.")
                    return False
                return self._download_authenticated_drive(drive_reference, drive_file_id)

        extract_options = {
            "logger": _DriveRedactingLogger() if drive_reference or drive_candidate else logging.getLogger(),
        }
        if cookiefile:
            extract_options["cookiefile"] = cookiefile

        try:
            with YoutubeDL(extract_options) as ydl_info:
                info = ydl_info.extract_info(self.transcription_task.url, download=False)
                video_title = info.get("title", "audio")
        except Exception as exc:
            if drive_file_id or drive_candidate:
                logging.debug("Error extracting Google Drive metadata (%s)", type(exc).__name__)
            else:
                logging.debug(f"Error extracting video info: {exc}")
            video_title = (
                f"Google Drive file {drive_file_id}"
                if drive_file_id
                else "audio"
            )

        video_title = YoutubeDL.sanitize_info({"title": video_title})["title"]
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            video_title = video_title.replace(char, '_')

        temp_dir = tempfile.mkdtemp()
        temp_output_path = os.path.join(temp_dir, video_title)
        wav_file = temp_output_path + ".wav"
        wav_file = str(Path(wav_file).resolve())

        options = {
            "format": "bestaudio/best",
            "progress_hooks": [self.on_download_progress],
            "outtmpl": temp_output_path,
            "logger": _DriveRedactingLogger() if drive_reference or drive_candidate else logging.getLogger(),
        }

        if cookiefile:
            options["cookiefile"] = cookiefile

        try:
            with YoutubeDL(options) as ydl:
                if drive_file_id:
                    logging.debug("Downloading Google Drive file %s", drive_file_id)
                else:
                    logging.debug(f"Downloading audio file from URL: {self.transcription_task.url}")
                ydl.download([self.transcription_task.url])
        except Exception as exc:
            if drive_reference or drive_candidate:
                logging.debug("Error downloading Google Drive file (%s)", type(exc).__name__)
            else:
                logging.debug(f"Error downloading audio: {exc}")
            if not drive_file_id or not self._download_google_drive_file(
                drive_file_id, temp_output_path
            ):
                error_message = "Google Drive download failed." if drive_file_id or drive_candidate else getattr(exc, "msg", str(exc))
                self.error.emit(error_message)
                return False

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", temp_output_path,
            "-ac", "1",
            "-ar", str(whisper_audio.SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            "-loglevel", "panic",
            wav_file
        ]

        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run(
                cmd,
                capture_output=True,
                startupinfo=si,
                env=app_env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace") if result.stderr else ""
            diagnostic = stderr or f"ffmpeg exited with status {result.returncode}"
            logging.warning(f"Error processing downloaded audio. Error: {diagnostic}")
            raise Exception(f"Error processing downloaded audio: {diagnostic}")

        self.transcription_task.file_path = wav_file
        logging.debug(f"Downloaded audio to file: {self.transcription_task.file_path}")
        return True

    @staticmethod
    def _is_drive_host_candidate(url: str) -> bool:
        try:
            parsed = urlsplit(url)
            return (parsed.hostname or "").lower() in GOOGLE_DRIVE_DOWNLOAD_HOSTS
        except ValueError:
            return any(host in url.lower() for host in GOOGLE_DRIVE_DOWNLOAD_HOSTS)

    def _drive_credentials_are_configured(self) -> bool:
        """Return whether Drive should take the terminal authenticated path.

        ``False`` is reserved for genuinely unconfigured storage.  An
        unavailable store or unreadable configured credentials is terminal and
        must not silently enter the anonymous compatibility paths.
        """
        try:
            state = self._drive_oauth_service.state()
            if state.status is keyring_store.DriveCredentialStatus.UNAVAILABLE:
                raise DriveStorageUnavailableError(
                    "Google Drive credential storage is unavailable"
                )
            if state.status is keyring_store.DriveCredentialStatus.UNCONFIGURED:
                return False
            if self._drive_oauth_service.credentials() is None:
                raise DriveCredentialsUnavailableError(
                    "Google Drive credentials could not be loaded"
                )
            return True
        except Exception as exc:
            if isinstance(exc, DriveStorageUnavailableError):
                raise
            logging.debug("Google Drive credential configuration failed (%s)", type(exc).__name__)
            raise DriveAuthenticationError(
                "Google Drive authorization is unavailable"
            ) from None

    def _download_authenticated_drive(self, drive_reference, drive_file_id: str) -> bool:
        temp_dir = tempfile.mkdtemp()
        video_title = YoutubeDL.sanitize_info(
            {"title": f"Google Drive file {drive_file_id}"}
        )["title"]
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            video_title = video_title.replace(char, "_")
        temp_output_path = os.path.join(temp_dir, video_title)
        try:
            downloader = DriveDownloader(self._drive_oauth_service)
            downloader.download(
                drive_reference,
                temp_output_path,
                progress=self._on_drive_download_progress,
                cancel=self._cancel_event,
            )
            self.download_progress.emit(1.0)
            return self._convert_downloaded_audio(temp_output_path)
        except Exception as exc:
            message = self._drive_download_error_message(exc)
            logging.debug("Authenticated Google Drive download failed (%s)", type(exc).__name__)
            self.error.emit(message)
            return False

    def _on_drive_download_progress(self, downloaded: int, total: int | None) -> None:
        if total and total > 0:
            self.download_progress.emit(downloaded / total)

    @staticmethod
    def _drive_download_error_message(error: Exception) -> str:
        messages = {
            DriveStorageUnavailableError: "Google Drive credential storage is unavailable.",
            DriveCredentialsUnavailableError: "Google Drive credentials are unavailable.",
            DriveCredentialsUnconfiguredError: "Google Drive credentials are not configured.",
            DriveAuthenticationError: "Google Drive authorization is unavailable.",
            DriveAuthorizationError: "Google Drive authorization was denied.",
            DriveDownloadDeniedError: "Google Drive does not allow this file to be downloaded.",
            DriveNotFoundError: "Google Drive file was not found.",
            DriveUnsupportedTypeError: "This Google Drive file type cannot be downloaded.",
            DriveCancelledError: "Google Drive download was canceled.",
            DriveRedirectError: "Google Drive returned an unexpected redirect.",
            DriveTransientError: "Google Drive download failed; please try again.",
        }
        for error_type, message in messages.items():
            if isinstance(error, error_type):
                return message
        return "Google Drive download failed."

    def _convert_downloaded_audio(self, temp_output_path: str) -> bool:
        """Convert an already downloaded file using the existing task contract."""
        wav_file = str(Path(temp_output_path + ".wav").resolve())
        cmd = [
            "ffmpeg", "-nostdin", "-threads", "0", "-i", temp_output_path,
            "-ac", "1", "-ar", str(whisper_audio.SAMPLE_RATE),
            "-acodec", "pcm_s16le", "-loglevel", "panic", wav_file,
        ]
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run(
                cmd, capture_output=True, startupinfo=si, env=app_env,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace") if result.stderr else ""
            diagnostic = stderr or f"ffmpeg exited with status {result.returncode}"
            logging.warning(f"Error processing downloaded audio. Error: {diagnostic}")
            raise Exception(f"Error processing downloaded audio: {diagnostic}")
        self.transcription_task.file_path = wav_file
        logging.debug("Downloaded Google Drive audio to file: %s", wav_file)
        return True

    def _download_google_drive_file(self, file_id: str, output_path: str) -> bool:
        """Download a public Drive file without attempting any access workaround."""
        download_url = GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=file_id)
        try:
            cookie_jar = CookieJar()
            opener = build_opener(
                HTTPCookieProcessor(cookie_jar), _GoogleDriveRedirectHandler
            )
            response = opener.open(Request(download_url, method="GET"), timeout=60)
            response = self._follow_google_drive_confirmation(opener, response)
            if response is None:
                raise ValueError("Google Drive returned an invalid confirmation page")

            headers = getattr(response, "headers", None)
            content_length = self._content_length(headers)
            downloaded_bytes = 0
            with response:
                with open(output_path, "wb") as output_file:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        output_file.write(chunk)
                        downloaded_bytes += len(chunk)
                        if content_length:
                            self.download_progress.emit(downloaded_bytes / content_length)

            if content_length is not None and downloaded_bytes != content_length:
                raise ValueError(
                    f"Google Drive download size mismatch: expected {content_length} bytes, "
                    f"received {downloaded_bytes}"
                )

            self.download_progress.emit(1.0)
            logging.debug(f"Downloaded Google Drive file to {output_path}")
            return True
        except Exception as exc:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                logging.debug("Error removing incomplete Google Drive file")
            logging.debug(
                "Error downloading Google Drive file (%s)", type(exc).__name__
            )
            return False

    @staticmethod
    def _content_length(headers):
        if headers is None:
            return None
        try:
            return int(headers.get("Content-Length", ""))
        except (TypeError, ValueError):
            return None

    def _follow_google_drive_confirmation(self, opener, response):
        if not _is_allowed_google_drive_url(response.geturl()):
            response.close()
            raise ValueError("Google Drive response came from an untrusted host")
        headers = getattr(response, "headers", None)
        content_type = ""
        if headers is not None:
            if hasattr(headers, "get_content_type"):
                content_type = headers.get_content_type()
            else:
                content_type = headers.get("Content-Type", "").split(";", 1)[0]
        status = response.getcode() if hasattr(response, "getcode") else getattr(response, "status", 200)
        if status is not None and status >= 400:
            raise ValueError(f"Google Drive returned HTTP status {status}")
        if content_type.lower() != "text/html":
            return response

        page = response.read(MAX_CONFIRMATION_PAGE_SIZE + 1)
        if len(page) > MAX_CONFIRMATION_PAGE_SIZE:
            raise ValueError("Google Drive confirmation page is too large")
        response.close()

        parser = _GoogleDriveConfirmationParser()
        try:
            parser.feed(page.decode("utf-8"))
            parser.close()
        except (UnicodeDecodeError, ValueError):
            raise ValueError("Google Drive confirmation page is malformed")
        if parser.malformed or len(parser.forms) + len(parser.links) != 1:
            raise ValueError("Google Drive confirmation page is malformed")

        if parser.forms:
            form = parser.forms[0]
            if form["method"] != "get" or not form["action"]:
                raise ValueError("Google Drive confirmation form is invalid")
            action = urljoin(response.geturl(), form["action"])
            query = parse_qsl(urlsplit(action).query, keep_blank_values=True)
            query.extend(form["hidden"])
            parts = urlsplit(action)
            followup_url = urlunsplit(
                (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
            )
        else:
            followup_url = urljoin(response.geturl(), parser.links[0])

        if not _is_allowed_google_drive_url(followup_url):
            raise ValueError("Google Drive confirmation action is not trusted")
        followup = opener.open(Request(followup_url, method="GET"), timeout=60)
        if not _is_allowed_google_drive_url(followup.geturl()):
            followup.close()
            raise ValueError("Google Drive confirmation redirected to an untrusted host")
        followup_headers = getattr(followup, "headers", None)
        followup_type = ""
        if followup_headers is not None:
            if hasattr(followup_headers, "get_content_type"):
                followup_type = followup_headers.get_content_type()
            else:
                followup_type = followup_headers.get("Content-Type", "").split(";", 1)[0]
        followup_status = followup.getcode() if hasattr(followup, "getcode") else getattr(followup, "status", 200)
        if followup_status is not None and followup_status >= 400:
            followup.close()
            raise ValueError(f"Google Drive confirmation returned HTTP status {followup_status}")
        if followup_type.lower() == "text/html":
            followup.close()
            raise ValueError("Google Drive confirmation did not return media")
        return followup

    def _handle_folder_watch(self):
        source_path = (
            self.transcription_task.original_file_path
            or self.transcription_task.file_path
        )
        if source_path and os.path.exists(source_path):
            if self.transcription_task.delete_source_file:
                os.remove(source_path)
            else:
                shutil.move(
                    source_path,
                    os.path.join(
                        self.transcription_task.output_directory,
                        os.path.basename(source_path),
                    ),
                )

    def on_download_progress(self, data: dict):
        if data["status"] == "downloading":
            self.download_progress.emit(data["downloaded_bytes"] / data["total_bytes"])

    @abstractmethod
    def transcribe(self) -> List[Segment]:
        ...

    @abstractmethod
    def stop(self):
        self._cancel_event.set()


def write_output(
    path: str,
    segments: List[Segment],
    output_format: OutputFormat,
    segment_key: str = 'text'
):
    logging.debug(
        "Writing transcription output, path = %s, output format = %s, number of segments = %s",
        path,
        output_format,
        len(segments),
    )

    with open(os.fsencode(path), "w", encoding="utf-8") as file:
        if output_format == OutputFormat.TXT:
            combined_text = ""
            previous_end_time = None

            paragraph_split_time = int(os.getenv("BUZZ_PARAGRAPH_SPLIT_TIME", "2000"))
            
            for segment in segments:
                if previous_end_time is not None and (segment.start - previous_end_time) >= paragraph_split_time:
                    combined_text += "\n\n"
                combined_text += getattr(segment, segment_key).strip() + " "
                previous_end_time = segment.end

            file.write(combined_text)

        elif output_format == OutputFormat.VTT:
            file.write("WEBVTT\n\n")
            for segment in segments:
                file.write(
                    f"{to_timestamp(segment.start)} --> {to_timestamp(segment.end)}\n"
                )
                file.write(f"{getattr(segment, segment_key)}\n\n")

        elif output_format == OutputFormat.SRT:
            for i, segment in enumerate(segments):
                file.write(f"{i + 1}\n")
                file.write(
                    f'{to_timestamp(segment.start, ms_separator=",")} --> {to_timestamp(segment.end, ms_separator=",")}\n'
                )
                file.write(f"{getattr(segment, segment_key)}\n\n")

    logging.debug("Written transcription output")


def to_timestamp(ms: float, ms_separator=".") -> str:
    hr = int(ms / (1000 * 60 * 60))
    ms -= hr * (1000 * 60 * 60)
    min = int(ms / (1000 * 60))
    ms -= min * (1000 * 60)
    sec = int(ms / 1000)
    ms = int(ms - sec * 1000)
    return f"{hr:02d}:{min:02d}:{sec:02d}{ms_separator}{ms:03d}"

# To detect when transcription source is a video
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".ogm", ".wmv"}

def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS
