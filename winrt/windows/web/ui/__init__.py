# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Web.UI")

try:
    import winrt.windows.applicationmodel.datatransfer
except Exception:
    pass

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

try:
    import winrt.windows.ui
except Exception:
    pass

try:
    import winrt.windows.web
except Exception:
    pass

try:
    import winrt.windows.web.http
except Exception:
    pass

class WebViewControlPermissionState(enum.IntEnum):
    UNKNOWN = 0
    DEFER = 1
    ALLOW = 2
    DENY = 3

class WebViewControlPermissionType(enum.IntEnum):
    GEOLOCATION = 0
    UNLIMITED_INDEXED_D_B_QUOTA = 1
    MEDIA = 2
    POINTER_LOCK = 3
    WEB_NOTIFICATIONS = 4
    SCREEN = 5
    IMMERSIVE_VIEW = 6

WebViewControlContentLoadingEventArgs = _ns_module.WebViewControlContentLoadingEventArgs
WebViewControlDOMContentLoadedEventArgs = _ns_module.WebViewControlDOMContentLoadedEventArgs
WebViewControlDeferredPermissionRequest = _ns_module.WebViewControlDeferredPermissionRequest
WebViewControlLongRunningScriptDetectedEventArgs = _ns_module.WebViewControlLongRunningScriptDetectedEventArgs
WebViewControlNavigationCompletedEventArgs = _ns_module.WebViewControlNavigationCompletedEventArgs
WebViewControlNavigationStartingEventArgs = _ns_module.WebViewControlNavigationStartingEventArgs
WebViewControlNewWindowRequestedEventArgs = _ns_module.WebViewControlNewWindowRequestedEventArgs
WebViewControlPermissionRequest = _ns_module.WebViewControlPermissionRequest
WebViewControlPermissionRequestedEventArgs = _ns_module.WebViewControlPermissionRequestedEventArgs
WebViewControlScriptNotifyEventArgs = _ns_module.WebViewControlScriptNotifyEventArgs
WebViewControlSettings = _ns_module.WebViewControlSettings
WebViewControlUnsupportedUriSchemeIdentifiedEventArgs = _ns_module.WebViewControlUnsupportedUriSchemeIdentifiedEventArgs
WebViewControlUnviewableContentIdentifiedEventArgs = _ns_module.WebViewControlUnviewableContentIdentifiedEventArgs
WebViewControlWebResourceRequestedEventArgs = _ns_module.WebViewControlWebResourceRequestedEventArgs
IWebViewControl = _ns_module.IWebViewControl
IWebViewControl2 = _ns_module.IWebViewControl2
