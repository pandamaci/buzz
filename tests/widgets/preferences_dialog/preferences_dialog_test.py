import os

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QTabWidget, QMessageBox
from pytestqt.qtbot import QtBot
from unittest.mock import patch
import threading
import time

from buzz.locale import _
from buzz.widgets.preferences_dialog.models.preferences import Preferences
from buzz.widgets.preferences_dialog.preferences_dialog import PreferencesDialog
from buzz.store.keyring_store import DriveCredentialStatus, DriveCredentialStoreState


class TestPreferencesDialog:
    locale_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../buzz/locale/lv_LV/LC_MESSAGES/buzz.mo")
    )

    def test_create(self, qtbot: QtBot, shortcuts):
        dialog = PreferencesDialog(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(dialog)

        assert dialog.windowTitle() == _("Preferences")

        tab_widget = dialog.findChild(QTabWidget)
        assert isinstance(tab_widget, QTabWidget)
        assert tab_widget.count() == 4
        assert tab_widget.tabText(0) == _("General")
        assert tab_widget.tabText(1) == _("Models")
        assert tab_widget.tabText(2) == _("Shortcuts")
        assert tab_widget.tabText(3) == _("Folder Watch")

    def test_create_localized(self, qtbot: QtBot, shortcuts, mocker):
        mocker.patch(
            "PyQt6.QtCore.QLocale.name",
            return_value='lv_LV',
        )

        # Reload the module after the patch
        from importlib import reload
        import buzz.locale
        import buzz.widgets.preferences_dialog.models.preferences
        import buzz.widgets.preferences_dialog.preferences_dialog

        reload(buzz.locale)
        reload(buzz.widgets.preferences_dialog.models.preferences)
        reload(buzz.widgets.preferences_dialog.preferences_dialog)

        from buzz.locale import _
        from buzz.widgets.preferences_dialog.models.preferences import Preferences
        from buzz.widgets.preferences_dialog.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog(
            shortcuts=shortcuts, preferences=Preferences.load(QSettings())
        )
        qtbot.add_widget(dialog)

        assert os.path.isfile(self.locale_file_path), "File .mo file does not exist"
        assert _("Preferences") == "Iestatījumi"
        assert dialog.windowTitle() == "Iestatījumi"

        tab_widget = dialog.findChild(QTabWidget)
        assert isinstance(tab_widget, QTabWidget)
        assert tab_widget.count() == 4
        assert tab_widget.tabText(0) == "Vispārīgi"
        assert tab_widget.tabText(1) == "Modeļi"
        assert tab_widget.tabText(2) == "Īsinājumi"
        assert tab_widget.tabText(3) == "Mapes vērošana"

    def test_reject_cancels_authorization_and_prevents_late_ui(self, qapp, qtbot: QtBot, shortcuts, tmp_path, mocker):
        with patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.state') as mock_state, \
             patch('buzz.widgets.preferences_dialog.drive_oauth_widget.DriveOAuthService.authorize') as mock_authorize:

            mock_state.return_value = DriveCredentialStoreState(status=DriveCredentialStatus.UNCONFIGURED)

            dialog = PreferencesDialog(
                shortcuts=shortcuts, preferences=Preferences.load(QSettings())
            )
            qtbot.add_widget(dialog)

            # Wait for it to become idle
            general_tab = dialog.general_tab_widget
            oauth_widget = general_tab.drive_oauth_widget
            
            def check_unconfigured():
                assert oauth_widget.authorize_button.isEnabled()
            qtbot.waitUntil(check_unconfigured)

            p = tmp_path / "client.json"
            p.write_text('{"installed": {"client_id": "test"}}')
            oauth_widget.file_line_edit.setText(str(p))

            from buzz.google_drive_oauth import AuthorizationCancelled

            authorize_called = threading.Event()
            
            def mock_auth(client, cancel=None):
                authorize_called.set()
                # Wait until cancel is set
                while cancel and not cancel.is_set():
                    time.sleep(0.01)
                raise AuthorizationCancelled("cancel")

            mock_authorize.side_effect = mock_auth

            dialog.show()
            oauth_widget.authorize_button.click()

            def check_authorizing():
                assert "Authorizing in browser..." in oauth_widget.state_label.text()
            qtbot.waitUntil(check_authorizing)

            assert authorize_called.wait(1.0)

            # mock message box to ensure it is not called
            mock_warning = mocker.patch.object(QMessageBox, 'warning')
            mock_information = mocker.patch.object(QMessageBox, 'information')

            # Now close/reject the dialog
            dialog.reject()

            def check_cancelled():
                assert oauth_widget._cancel_event.is_set()
                # The widget is hidden now, so the error from AuthorizationCancelled will be swallowed
            qtbot.waitUntil(check_cancelled)
            
            from PyQt6.QtCore import QThreadPool
            QThreadPool.globalInstance().waitForDone()

            # Ensure no UI popups occurred
            mock_warning.assert_not_called()
            mock_information.assert_not_called()
