# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Security.Authentication.Web")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class TokenBindingKeyType(enum.IntEnum):
    RSA2048 = 0
    ECDSA_P256 = 1
    ANY_EXISTING = 2

class WebAuthenticationOptions(enum.IntFlag):
    NONE = 0
    SILENT_MODE = 0x1
    USE_TITLE = 0x2
    USE_HTTP_POST = 0x4
    USE_CORPORATE_NETWORK = 0x8

class WebAuthenticationStatus(enum.IntEnum):
    SUCCESS = 0
    USER_CANCEL = 1
    ERROR_HTTP = 2

WebAuthenticationBroker = _ns_module.WebAuthenticationBroker
WebAuthenticationResult = _ns_module.WebAuthenticationResult
