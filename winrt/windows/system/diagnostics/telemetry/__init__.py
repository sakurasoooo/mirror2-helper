# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.System.Diagnostics.Telemetry")

class PlatformTelemetryRegistrationStatus(enum.IntEnum):
    SUCCESS = 0
    SETTINGS_OUT_OF_RANGE = 1
    UNKNOWN_FAILURE = 2

PlatformTelemetryClient = _ns_module.PlatformTelemetryClient
PlatformTelemetryRegistrationResult = _ns_module.PlatformTelemetryRegistrationResult
PlatformTelemetryRegistrationSettings = _ns_module.PlatformTelemetryRegistrationSettings