# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.System.Diagnostics.TraceReporting")

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class PlatformDiagnosticActionState(enum.IntEnum):
    SUCCESS = 0
    FREE_NETWORK_NOT_AVAILABLE = 1
    A_C_POWER_NOT_AVAILABLE = 2

class PlatformDiagnosticEscalationType(enum.IntEnum):
    ON_COMPLETION = 0
    ON_FAILURE = 1

class PlatformDiagnosticEventBufferLatencies(enum.IntFlag):
    NORMAL = 0x1
    COST_DEFERRED = 0x2
    REALTIME = 0x4

class PlatformDiagnosticTracePriority(enum.IntEnum):
    NORMAL = 0
    USER_ELEVATED = 1

class PlatformDiagnosticTraceSlotState(enum.IntEnum):
    NOT_RUNNING = 0
    RUNNING = 1
    THROTTLED = 2

class PlatformDiagnosticTraceSlotType(enum.IntEnum):
    ALTERNATIVE = 0
    ALWAYS_ON = 1
    MINI = 2

PlatformDiagnosticActions = _ns_module.PlatformDiagnosticActions
PlatformDiagnosticTraceInfo = _ns_module.PlatformDiagnosticTraceInfo
PlatformDiagnosticTraceRuntimeInfo = _ns_module.PlatformDiagnosticTraceRuntimeInfo
