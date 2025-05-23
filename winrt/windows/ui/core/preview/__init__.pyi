# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.ui.windowmanagement
except Exception:
    pass

class CoreAppWindowPreview(_winrt.winrt_base):
    ...
    def get_id_from_window(window: winrt.windows.ui.windowmanagement.AppWindow) -> int:
        ...

class SystemNavigationCloseRequestedPreviewEventArgs(_winrt.winrt_base):
    ...
    handled: bool
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...

class SystemNavigationManagerPreview(_winrt.winrt_base):
    ...
    def get_for_current_view() -> SystemNavigationManagerPreview:
        ...
    def add_close_requested(handler: winrt.windows.foundation.EventHandler[SystemNavigationCloseRequestedPreviewEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_close_requested(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

