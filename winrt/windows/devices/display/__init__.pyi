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
    import winrt.windows.graphics
except Exception:
    pass

class DisplayMonitorConnectionKind(enum.IntEnum):
    INTERNAL = 0
    WIRED = 1
    WIRELESS = 2
    VIRTUAL = 3

class DisplayMonitorDescriptorKind(enum.IntEnum):
    EDID = 0
    DISPLAY_ID = 1

class DisplayMonitorPhysicalConnectorKind(enum.IntEnum):
    UNKNOWN = 0
    H_D15 = 1
    ANALOG_T_V = 2
    DVI = 3
    HDMI = 4
    LVDS = 5
    SDI = 6
    DISPLAY_PORT = 7

class DisplayMonitorUsageKind(enum.IntEnum):
    STANDARD = 0
    HEAD_MOUNTED = 1
    SPECIAL_PURPOSE = 2

class DisplayMonitor(_winrt.winrt_base):
    ...
    blue_primary: winrt.windows.foundation.Point
    connection_kind: DisplayMonitorConnectionKind
    device_id: str
    display_adapter_device_id: str
    display_adapter_id: winrt.windows.graphics.DisplayAdapterId
    display_adapter_target_id: int
    display_name: str
    green_primary: winrt.windows.foundation.Point
    max_average_full_frame_luminance_in_nits: float
    max_luminance_in_nits: float
    min_luminance_in_nits: float
    native_resolution_in_raw_pixels: winrt.windows.graphics.SizeInt32
    physical_connector: DisplayMonitorPhysicalConnectorKind
    physical_size_in_inches: typing.Optional[winrt.windows.foundation.Size]
    raw_dpi_x: float
    raw_dpi_y: float
    red_primary: winrt.windows.foundation.Point
    usage_kind: DisplayMonitorUsageKind
    white_point: winrt.windows.foundation.Point
    is_dolby_vision_supported_in_hdr_mode: bool
    def from_id_async(device_id: str) -> winrt.windows.foundation.IAsyncOperation[DisplayMonitor]:
        ...
    def from_interface_id_async(device_interface_id: str) -> winrt.windows.foundation.IAsyncOperation[DisplayMonitor]:
        ...
    def get_descriptor(descriptor_kind: DisplayMonitorDescriptorKind) -> int:
        ...
    def get_device_selector() -> str:
        ...

