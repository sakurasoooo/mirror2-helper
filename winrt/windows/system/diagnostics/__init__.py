# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.System.Diagnostics")

try:
    import winrt.windows.data.json
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
    import winrt.windows.system
except Exception:
    pass

class DiagnosticActionState(enum.IntEnum):
    INITIALIZING = 0
    DOWNLOADING = 1
    VERIFYING_TRUST = 2
    DETECTING = 3
    RESOLVING = 4
    VERIFYING_RESOLUTION = 5
    EXECUTING = 6

DiagnosticActionResult = _ns_module.DiagnosticActionResult
DiagnosticInvoker = _ns_module.DiagnosticInvoker
ProcessCpuUsage = _ns_module.ProcessCpuUsage
ProcessCpuUsageReport = _ns_module.ProcessCpuUsageReport
ProcessDiagnosticInfo = _ns_module.ProcessDiagnosticInfo
ProcessDiskUsage = _ns_module.ProcessDiskUsage
ProcessDiskUsageReport = _ns_module.ProcessDiskUsageReport
ProcessMemoryUsage = _ns_module.ProcessMemoryUsage
ProcessMemoryUsageReport = _ns_module.ProcessMemoryUsageReport
SystemCpuUsage = _ns_module.SystemCpuUsage
SystemCpuUsageReport = _ns_module.SystemCpuUsageReport
SystemDiagnosticInfo = _ns_module.SystemDiagnosticInfo
SystemMemoryUsage = _ns_module.SystemMemoryUsage
SystemMemoryUsageReport = _ns_module.SystemMemoryUsageReport