# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Media.Protection")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.media.playback
except Exception:
    pass

class GraphicsTrustStatus(enum.IntEnum):
    TRUST_NOT_REQUIRED = 0
    TRUST_ESTABLISHED = 1
    ENVIRONMENT_NOT_SUPPORTED = 2
    DRIVER_NOT_SUPPORTED = 3
    DRIVER_SIGNING_FAILURE = 4
    UNKNOWN_FAILURE = 5

class HdcpProtection(enum.IntEnum):
    OFF = 0
    ON = 1
    ON_WITH_TYPE_ENFORCEMENT = 2

class HdcpSetProtectionResult(enum.IntEnum):
    SUCCESS = 0
    TIMED_OUT = 1
    NOT_SUPPORTED = 2
    UNKNOWN_FAILURE = 3

class ProtectionCapabilityResult(enum.IntEnum):
    NOT_SUPPORTED = 0
    MAYBE = 1
    PROBABLY = 2

class RevocationAndRenewalReasons(enum.IntFlag):
    USER_MODE_COMPONENT_LOAD = 0x1
    KERNEL_MODE_COMPONENT_LOAD = 0x2
    APP_COMPONENT = 0x4
    GLOBAL_REVOCATION_LIST_LOAD_FAILED = 0x10
    INVALID_GLOBAL_REVOCATION_LIST_SIGNATURE = 0x20
    GLOBAL_REVOCATION_LIST_ABSENT = 0x1000
    COMPONENT_REVOKED = 0x2000
    INVALID_COMPONENT_CERTIFICATE_EXTENDED_KEY_USE = 0x4000
    COMPONENT_CERTIFICATE_REVOKED = 0x8000
    INVALID_COMPONENT_CERTIFICATE_ROOT = 0x10000
    COMPONENT_HIGH_SECURITY_CERTIFICATE_REVOKED = 0x20000
    COMPONENT_LOW_SECURITY_CERTIFICATE_REVOKED = 0x40000
    BOOT_DRIVER_VERIFICATION_FAILED = 0x100000
    COMPONENT_SIGNED_WITH_TEST_CERTIFICATE = 0x1000000
    ENCRYPTION_FAILURE = 0x10000000

ComponentLoadFailedEventArgs = _ns_module.ComponentLoadFailedEventArgs
HdcpSession = _ns_module.HdcpSession
MediaProtectionManager = _ns_module.MediaProtectionManager
MediaProtectionPMPServer = _ns_module.MediaProtectionPMPServer
MediaProtectionServiceCompletion = _ns_module.MediaProtectionServiceCompletion
ProtectionCapabilities = _ns_module.ProtectionCapabilities
RevocationAndRenewalInformation = _ns_module.RevocationAndRenewalInformation
RevocationAndRenewalItem = _ns_module.RevocationAndRenewalItem
ServiceRequestedEventArgs = _ns_module.ServiceRequestedEventArgs
IMediaProtectionServiceRequest = _ns_module.IMediaProtectionServiceRequest