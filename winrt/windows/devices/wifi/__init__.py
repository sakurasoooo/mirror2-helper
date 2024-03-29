# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Devices.WiFi")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.networking.connectivity
except Exception:
    pass

try:
    import winrt.windows.security.credentials
except Exception:
    pass

class WiFiAccessStatus(enum.IntEnum):
    UNSPECIFIED = 0
    ALLOWED = 1
    DENIED_BY_USER = 2
    DENIED_BY_SYSTEM = 3

class WiFiConnectionMethod(enum.IntEnum):
    DEFAULT = 0
    WPS_PIN = 1
    WPS_PUSH_BUTTON = 2

class WiFiConnectionStatus(enum.IntEnum):
    UNSPECIFIED_FAILURE = 0
    SUCCESS = 1
    ACCESS_REVOKED = 2
    INVALID_CREDENTIAL = 3
    NETWORK_NOT_AVAILABLE = 4
    TIMEOUT = 5
    UNSUPPORTED_AUTHENTICATION_PROTOCOL = 6

class WiFiNetworkKind(enum.IntEnum):
    ANY = 0
    INFRASTRUCTURE = 1
    ADHOC = 2

class WiFiPhyKind(enum.IntEnum):
    UNKNOWN = 0
    FHSS = 1
    DSSS = 2
    I_R_BASEBAND = 3
    OFDM = 4
    HRDSSS = 5
    ERP = 6
    H_T = 7
    VHT = 8
    DMG = 9
    H_E = 10

class WiFiReconnectionKind(enum.IntEnum):
    AUTOMATIC = 0
    MANUAL = 1

class WiFiWpsConfigurationStatus(enum.IntEnum):
    UNSPECIFIED_FAILURE = 0
    SUCCESS = 1
    TIMEOUT = 2

class WiFiWpsKind(enum.IntEnum):
    UNKNOWN = 0
    PIN = 1
    PUSH_BUTTON = 2
    NFC = 3
    ETHERNET = 4
    USB = 5

WiFiAdapter = _ns_module.WiFiAdapter
WiFiAvailableNetwork = _ns_module.WiFiAvailableNetwork
WiFiConnectionResult = _ns_module.WiFiConnectionResult
WiFiNetworkReport = _ns_module.WiFiNetworkReport
WiFiWpsConfigurationResult = _ns_module.WiFiWpsConfigurationResult
