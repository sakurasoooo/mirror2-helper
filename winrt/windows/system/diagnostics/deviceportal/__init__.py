# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.System.Diagnostics.DevicePortal")

try:
    import winrt.windows.applicationmodel.appservice
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
    import winrt.windows.networking.sockets
except Exception:
    pass

try:
    import winrt.windows.web.http
except Exception:
    pass

class DevicePortalConnectionClosedReason(enum.IntEnum):
    UNKNOWN = 0
    RESOURCE_LIMITS_EXCEEDED = 1
    PROTOCOL_ERROR = 2
    NOT_AUTHORIZED = 3
    USER_NOT_PRESENT = 4
    SERVICE_TERMINATED = 5

DevicePortalConnection = _ns_module.DevicePortalConnection
DevicePortalConnectionClosedEventArgs = _ns_module.DevicePortalConnectionClosedEventArgs
DevicePortalConnectionRequestReceivedEventArgs = _ns_module.DevicePortalConnectionRequestReceivedEventArgs
