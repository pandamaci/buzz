from PyQt6.QtCore import Qt, QThreadPool
from unittest.mock import patch

from PyQt6.QtWidgets import QFileDialog, QLabel

from buzz.widgets.preferences_dialog.drive_oauth_widget import DriveOAuthWidget
from buzz.store.keyring_store import DriveCredentialStatus, DriveCredentialStoreState

class TestDriveOAuthWidget:
    def test_init_state_unconfigured(self, qapp, qtbot):
        with patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.state') as mock_state:
            mock_state.return_value = DriveCredentialStoreState(status=DriveCredentialStatus.UNCONFIGURED)
            widget = DriveOAuthWidget()
            qtbot.addWidget(widget)
            
            def check():
                assert "Unconfigured" in widget.state_label.text()
                assert widget.authorize_button.isEnabled()
                assert not widget.disconnect_button.isEnabled()
                assert widget.cancel_button.isHidden()

            qtbot.waitUntil(check)
            
            # Verify instructions and hyperlink behavior
            info_label = widget.findChild(QLabel)
            assert "Google Cloud Console" in info_label.text()
            assert "Setup Instructions" in info_label.text()
            assert info_label.openExternalLinks() is True

    def test_init_state_available(self, qapp, qtbot):
        with patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.state') as mock_state:
            mock_state.return_value = DriveCredentialStoreState(status=DriveCredentialStatus.AVAILABLE)
            widget = DriveOAuthWidget()
            qtbot.addWidget(widget)
            
            def check():
                assert "Authorization saved" in widget.state_label.text()
                assert not widget.authorize_button.isEnabled()
                assert widget.disconnect_button.isEnabled()
                assert widget.cancel_button.isHidden()

            qtbot.waitUntil(check)

    def test_init_state_unavailable(self, qapp, qtbot):
        with patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.state') as mock_state:
            mock_state.return_value = DriveCredentialStoreState(status=DriveCredentialStatus.UNAVAILABLE)
            widget = DriveOAuthWidget()
            qtbot.addWidget(widget)
            
            def check():
                assert "Unavailable" in widget.state_label.text()
                assert not widget.authorize_button.isEnabled()
                assert not widget.disconnect_button.isEnabled()
                assert widget.cancel_button.isHidden()

            qtbot.waitUntil(check)

    def test_browse_file(self, qapp, qtbot):
        with patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.state') as mock_state:
            mock_state.return_value = DriveCredentialStoreState(status=DriveCredentialStatus.UNCONFIGURED)
            widget = DriveOAuthWidget()
            qtbot.addWidget(widget)

            with patch.object(QFileDialog, 'getOpenFileName', return_value=('/path/to/client.json', '')):
                widget._browse_file()
                
            assert widget.file_line_edit.text() == '/path/to/client.json'

    def test_disconnect(self, qapp, qtbot):
        with patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.state') as mock_state, \
             patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.disconnect') as mock_disconnect:
            
            mock_state.return_value = DriveCredentialStoreState(status=DriveCredentialStatus.AVAILABLE)
            widget = DriveOAuthWidget()
            qtbot.addWidget(widget)
            
            # wait for available
            def check_available():
                assert widget.disconnect_button.isEnabled()
            qtbot.waitUntil(check_available)

            mock_state.return_value = DriveCredentialStoreState(status=DriveCredentialStatus.UNCONFIGURED)
            
            qtbot.mouseClick(widget.disconnect_button, Qt.MouseButton.LeftButton)
            
            def check_disconnected():
                mock_disconnect.assert_called_once()
                assert "Unconfigured" in widget.state_label.text()
            qtbot.waitUntil(check_disconnected)

    def test_cancel_authorization_flow(self, qapp, qtbot, tmp_path):
        with patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.state') as mock_state, \
             patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.authorize') as mock_authorize, \
             patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.cancel_authorization') as mock_cancel_auth:
            
            mock_state.return_value = DriveCredentialStoreState(status=DriveCredentialStatus.UNCONFIGURED)
            widget = DriveOAuthWidget()
            qtbot.addWidget(widget)
            
            def check_unconfigured():
                assert widget.authorize_button.isEnabled()
            qtbot.waitUntil(check_unconfigured)

            p = tmp_path / "client.json"
            p.write_text('{"installed": {"client_id": "test"}}')
            widget.file_line_edit.setText(str(p))
            
            # Authorize blocks in mock
            import threading
            import time
            from buzz.google_drive_oauth import AuthorizationCancelled
            
            authorize_called = threading.Event()
            
            def mock_auth(client, cancel=None):
                authorize_called.set()
                # Wait until cancel is set
                while cancel and not cancel.is_set():
                    time.sleep(0.01)
                raise AuthorizationCancelled("cancel")

            mock_authorize.side_effect = mock_auth

            qtbot.mouseClick(widget.authorize_button, Qt.MouseButton.LeftButton)
            
            def check_authorizing():
                assert "Authorizing in browser..." in widget.state_label.text()
                assert not widget.cancel_button.isHidden()
                assert widget.cancel_button.isEnabled()
            qtbot.waitUntil(check_authorizing)
            
            assert authorize_called.wait(1.0)
            
            qtbot.mouseClick(widget.cancel_button, Qt.MouseButton.LeftButton)
            
            def check_cancelled():
                assert "Unconfigured" in widget.state_label.text()
                assert widget.cancel_button.isHidden()
                mock_cancel_auth.assert_called_once()
            qtbot.waitUntil(check_cancelled)
            
            QThreadPool.globalInstance().waitForDone()

    def test_destroyed_cancels_authorization(self, qapp, qtbot):
        widget = DriveOAuthWidget()
        assert not widget._cancel_event.is_set()
        widget.deleteLater()
        
        def check_destroyed():
            assert widget._cancel_event.is_set()
        qtbot.waitUntil(check_destroyed)
        
        QThreadPool.globalInstance().waitForDone()
