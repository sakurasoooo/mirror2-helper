# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Networking.ServiceDiscovery.Dnssd")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.networking
except Exception:
    pass

try:
    import winrt.windows.networking.connectivity
except Exception:
    pass

try:
    import winrt.windows.networking.sockets
except Exception:
    pass

class DnssdRegistrationStatus(enum.IntEnum):
    SUCCESS = 0
    INVALID_SERVICE_NAME = 1
    SERVER_ERROR = 2
    SECURITY_ERROR = 3

class DnssdServiceWatcherStatus(enum.IntEnum):
    CREATED = 0
    STARTED = 1
    ENUMERATION_COMPLETED = 2
    STOPPING = 3
    STOPPED = 4
    ABORTED = 5

DnssdRegistrationResult = _ns_module.DnssdRegistrationResult
DnssdServiceInstance = _ns_module.DnssdServiceInstance
DnssdServiceInstanceCollection = _ns_module.DnssdServiceInstanceCollection
DnssdServiceWatcher = _ns_module.DnssdServiceWatcher
