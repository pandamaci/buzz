import threading
from PyQt6.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QGroupBox
)

from buzz.locale import _
from buzz.widgets.line_edit import LineEdit
from buzz.google_drive_oauth import DriveOAuthService, parse_desktop_client_json, AuthorizationCancelled
from buzz.store.keyring_store import DriveCredentialStatus, DriveCredentialStoreState

class GetStateJob(QRunnable):
    class Signals(QObject):
        success = pyqtSignal(object)  # DriveCredentialStoreState

    def __init__(self, service: DriveOAuthService):
        super().__init__()
        self.service = service
        self.signals = self.Signals()

    def run(self):
        state = self.service.state()
        self.signals.success.emit(state)

class DisconnectJob(QRunnable):
    class Signals(QObject):
        success = pyqtSignal()
        error = pyqtSignal(str)

    def __init__(self, service: DriveOAuthService):
        super().__init__()
        self.service = service
        self.signals = self.Signals()

    def run(self):
        try:
            self.service.disconnect()
            self.signals.success.emit()
        except Exception as e:
            self.signals.error.emit(str(e))

class AuthorizeJob(QRunnable):
    class Signals(QObject):
        success = pyqtSignal()
        error = pyqtSignal(str)

    def __init__(self, client_json: str, service: DriveOAuthService, cancel_event: threading.Event):
        super().__init__()
        self.client_json = client_json
        self.service = service
        self.cancel_event = cancel_event
        self.signals = self.Signals()

    def run(self):
        try:
            client = parse_desktop_client_json(self.client_json)
            self.service.authorize(client, cancel=self.cancel_event)
            self.signals.success.emit()
        except AuthorizationCancelled:
            self.signals.error.emit(_("Authorization cancelled."))
        except Exception as e:
            self.signals.error.emit(str(e))

class DriveOAuthWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(_("Advanced Google Drive OAuth"), parent)
        self.service = DriveOAuthService()
        self._cancel_event = threading.Event()
        self._authorizing = False
        self._disconnecting = False
        self._init_ui()
        self.destroyed.connect(self._cancel_event.set)
        self._update_state()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        info_text = _(
            "<p>Bring Your Own (BYO) Desktop OAuth client JSON. You must configure your own "
            "Google Cloud project with the broad <b>drive.readonly</b> scope. A secure OS keyring "
            "is required on Linux. Tokens are not displayed. Disconnecting deletes the local saved "
            "authorization. Full Drive URL/resource-key may remain in local history for retries until "
            "history is deleted.</p>"
            "<p><b>Setup Instructions:</b><br/>"
            "1. Go to the <a href=\"https://console.cloud.google.com/\">Google Cloud Console</a> and create a project.<br/>"
            "2. Enable the <b>Google Drive API</b> for your project.<br/>"
            "3. Configure the <b>OAuth consent screen</b> (add yourself as a Test User).<br/>"
            "4. Open <b>Clients</b> and create an <b>OAuth client ID</b> (Application type: <b>Desktop app</b>).<br/>"
            "5. Download the client JSON file and select it below.</p>"
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setOpenExternalLinks(True)
        layout.addWidget(info_label)

        self.state_label = QLabel()
        layout.addWidget(self.state_label)

        file_layout = QHBoxLayout()
        self.file_line_edit = LineEdit()
        self.file_line_edit.setPlaceholderText(_("Select Client JSON file..."))
        browse_button = QPushButton(_("Browse..."))
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_line_edit)
        file_layout.addWidget(browse_button)
        layout.addLayout(file_layout)

        button_layout = QHBoxLayout()
        self.authorize_button = QPushButton(_("Authorize"))
        self.authorize_button.clicked.connect(self._authorize)
        
        self.cancel_button = QPushButton(_("Cancel"))
        self.cancel_button.clicked.connect(self._cancel_authorization)
        self.cancel_button.setVisible(False)
        
        self.disconnect_button = QPushButton(_("Disconnect"))
        self.disconnect_button.clicked.connect(self._disconnect)
        
        button_layout.addWidget(self.authorize_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.disconnect_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

    def _update_state(self):
        if self._authorizing or self._disconnecting:
            return

        self.authorize_button.setEnabled(False)
        self.disconnect_button.setEnabled(False)
        self.cancel_button.setVisible(False)

        job = GetStateJob(self.service)
        job.signals.success.connect(self._on_state_received)
        QThreadPool.globalInstance().start(job)

    def _on_state_received(self, state: DriveCredentialStoreState):
        if self._authorizing or self._disconnecting:
            return

        if state.status == DriveCredentialStatus.UNAVAILABLE:
            self.state_label.setText(_("Status: <b>Unavailable</b> (Secure OS keyring required)"))
            self.authorize_button.setEnabled(False)
            self.disconnect_button.setEnabled(False)
            self.cancel_button.setVisible(False)
        elif state.status == DriveCredentialStatus.AVAILABLE:
            self.state_label.setText(_("Status: <b>Authorization saved</b>"))
            self.authorize_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            self.cancel_button.setVisible(False)
        else:
            self.state_label.setText(_("Status: <b>Unconfigured</b>"))
            self.authorize_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.cancel_button.setVisible(False)

    def _browse_file(self):
        path, _filter = QFileDialog.getOpenFileName(self, _("Select Client JSON"), "", "JSON Files (*.json);;All Files (*)")
        if path:
            self.file_line_edit.setText(path)

    def _authorize(self):
        path = self.file_line_edit.text().strip()
        if not path:
            QMessageBox.warning(self, _("Error"), _("Please select a client JSON file."))
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                client_json = f.read()
        except Exception as e:
            QMessageBox.warning(self, _("Error"), _(f"Failed to read file: {e}"))
            return

        self._authorizing = True
        self._cancel_event.clear()

        self.authorize_button.setEnabled(False)
        self.disconnect_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.state_label.setText(_("Status: <b>Authorizing in browser...</b>"))
        
        job = AuthorizeJob(client_json, self.service, self._cancel_event)
        job.signals.success.connect(self._on_authorize_success)
        job.signals.error.connect(self._on_authorize_error)
        QThreadPool.globalInstance().start(job)

    def cancel(self):
        if self._authorizing:
            self._cancel_event.set()
            try:
                self.service.cancel_authorization()
            except Exception:
                pass
            self._authorizing = False
            self.cancel_button.setEnabled(False)
            self.state_label.setText(_("Status: <b>Cancelling...</b>"))

    def _cancel_authorization(self):
        self.cancel()

    def _on_authorize_success(self):
        self._authorizing = False
        self._update_state()
        if not self.isVisible() or self._cancel_event.is_set():
            return
        QMessageBox.information(self, _("Success"), _("Google Drive authorization saved successfully."))

    def _on_authorize_error(self, error_msg):
        self._authorizing = False
        self._update_state()
        if not self.isVisible() or (self._cancel_event.is_set() and error_msg == _("Authorization cancelled.")):
            return
        QMessageBox.warning(self, _("Authorization Failed"), error_msg)

    def _disconnect(self):
        self._disconnecting = True
        self.authorize_button.setEnabled(False)
        self.disconnect_button.setEnabled(False)
        self.state_label.setText(_("Status: <b>Disconnecting...</b>"))
        
        job = DisconnectJob(self.service)
        job.signals.success.connect(self._on_disconnect_success)
        job.signals.error.connect(self._on_disconnect_error)
        QThreadPool.globalInstance().start(job)

    def _on_disconnect_success(self):
        self._disconnecting = False
        self._update_state()

    def _on_disconnect_error(self, error_msg):
        self._disconnecting = False
        self._update_state()
        if not self.isVisible():
            return
        QMessageBox.warning(self, _("Error"), _(f"Failed to disconnect: {error_msg}"))
