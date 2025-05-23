# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Graphics.Capture")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.graphics
except Exception:
    pass

try:
    import winrt.windows.graphics.directx
except Exception:
    pass

try:
    import winrt.windows.graphics.directx.direct3d11
except Exception:
    pass

try:
    import winrt.windows.security.authorization.appcapabilityaccess
except Exception:
    pass

try:
    import winrt.windows.system
except Exception:
    pass

try:
    import winrt.windows.ui
except Exception:
    pass

try:
    import winrt.windows.ui.composition
except Exception:
    pass

class GraphicsCaptureAccessKind(enum.IntEnum):
    BORDERLESS = 0
    PROGRAMMATIC = 1

Direct3D11CaptureFrame = _ns_module.Direct3D11CaptureFrame
Direct3D11CaptureFramePool = _ns_module.Direct3D11CaptureFramePool
GraphicsCaptureAccess = _ns_module.GraphicsCaptureAccess
GraphicsCaptureItem = _ns_module.GraphicsCaptureItem
GraphicsCapturePicker = _ns_module.GraphicsCapturePicker
GraphicsCaptureSession = _ns_module.GraphicsCaptureSession
