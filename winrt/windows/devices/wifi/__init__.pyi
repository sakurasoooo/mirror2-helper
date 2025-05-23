# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
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

class WiFiAdapter(_winrt.winrt_base):
    ...
    network_adapter: winrt.windows.networking.connectivity.NetworkAdapter
    network_report: WiFiNetworkReport
    def connect_async(available_network: WiFiAvailableNetwork, reconnection_kind: WiFiReconnectionKind) -> winrt.windows.foundation.IAsyncOperation[WiFiConnectionResult]:
        ...
    def connect_async(available_network: WiFiAvailableNetwork, reconnection_kind: WiFiReconnectionKind, password_credential: winrt.windows.security.credentials.PasswordCredential) -> winrt.windows.foundation.IAsyncOperation[WiFiConnectionResult]:
        ...
    def connect_async(available_network: WiFiAvailableNetwork, reconnection_kind: WiFiReconnectionKind, password_credential: winrt.windows.security.credentials.PasswordCredential, ssid: str) -> winrt.windows.foundation.IAsyncOperation[WiFiConnectionResult]:
        ...
    def connect_async(available_network: WiFiAvailableNetwork, reconnection_kind: WiFiReconnectionKind, password_credential: winrt.windows.security.credentials.PasswordCredential, ssid: str, connection_method: WiFiConnectionMethod) -> winrt.windows.foundation.IAsyncOperation[WiFiConnectionResult]:
        ...
    def disconnect() -> None:
        ...
    def find_all_adapters_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[WiFiAdapter]]:
        ...
    def from_id_async(device_id: str) -> winrt.windows.foundation.IAsyncOperation[WiFiAdapter]:
        ...
    def get_device_selector() -> str:
        ...
    def get_wps_configuration_async(available_network: WiFiAvailableNetwork) -> winrt.windows.foundation.IAsyncOperation[WiFiWpsConfigurationResult]:
        ...
    def request_access_async() -> winrt.windows.foundation.IAsyncOperation[WiFiAccessStatus]:
        ...
    def scan_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def add_available_networks_changed(args: winrt.windows.foundation.TypedEventHandler[WiFiAdapter, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_available_networks_changed(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class WiFiAvailableNetwork(_winrt.winrt_base):
    ...
    beacon_interval: winrt.windows.foundation.TimeSpan
    bssid: str
    channel_center_frequency_in_kilohertz: int
    is_wi_fi_direct: bool
    network_kind: WiFiNetworkKind
    network_rssi_in_decibel_milliwatts: float
    phy_kind: WiFiPhyKind
    security_settings: winrt.windows.networking.connectivity.NetworkSecuritySettings
    signal_bars: int
    ssid: str
    uptime: winrt.windows.foundation.TimeSpan

class WiFiConnectionResult(_winrt.winrt_base):
    ...
    connection_status: WiFiConnectionStatus

class WiFiNetworkReport(_winrt.winrt_base):
    ...
    available_networks: winrt.windows.foundation.collections.IVectorView[WiFiAvailableNetwork]
    timestamp: winrt.windows.foundation.DateTime

class WiFiWpsConfigurationResult(_winrt.winrt_base):
    ...
    status: WiFiWpsConfigurationStatus
    supported_wps_kinds: winrt.windows.foundation.collections.IVectorView[WiFiWpsKind]

