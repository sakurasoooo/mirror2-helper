# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Graphics.Holographic")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.foundation.numerics
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
    import winrt.windows.perception
except Exception:
    pass

try:
    import winrt.windows.perception.spatial
except Exception:
    pass

try:
    import winrt.windows.ui.core
except Exception:
    pass

class HolographicDepthReprojectionMethod(enum.IntEnum):
    DEPTH_REPROJECTION = 0
    AUTO_PLANAR = 1

class HolographicFramePresentResult(enum.IntEnum):
    SUCCESS = 0
    DEVICE_REMOVED = 1

class HolographicFramePresentWaitBehavior(enum.IntEnum):
    WAIT_FOR_FRAME_TO_FINISH = 0
    DO_NOT_WAIT_FOR_FRAME_TO_FINISH = 1

class HolographicReprojectionMode(enum.IntEnum):
    POSITION_AND_ORIENTATION = 0
    ORIENTATION_ONLY = 1
    DISABLED = 2

class HolographicSpaceUserPresence(enum.IntEnum):
    ABSENT = 0
    PRESENT_PASSIVE = 1
    PRESENT_ACTIVE = 2

class HolographicViewConfigurationKind(enum.IntEnum):
    DISPLAY = 0
    PHOTO_VIDEO_CAMERA = 1

HolographicAdapterId = _ns_module.HolographicAdapterId
HolographicFrameId = _ns_module.HolographicFrameId
HolographicStereoTransform = _ns_module.HolographicStereoTransform
HolographicCamera = _ns_module.HolographicCamera
HolographicCameraPose = _ns_module.HolographicCameraPose
HolographicCameraRenderingParameters = _ns_module.HolographicCameraRenderingParameters
HolographicCameraViewportParameters = _ns_module.HolographicCameraViewportParameters
HolographicDisplay = _ns_module.HolographicDisplay
HolographicFrame = _ns_module.HolographicFrame
HolographicFramePrediction = _ns_module.HolographicFramePrediction
HolographicFramePresentationMonitor = _ns_module.HolographicFramePresentationMonitor
HolographicFramePresentationReport = _ns_module.HolographicFramePresentationReport
HolographicFrameRenderingReport = _ns_module.HolographicFrameRenderingReport
HolographicFrameScanoutMonitor = _ns_module.HolographicFrameScanoutMonitor
HolographicFrameScanoutReport = _ns_module.HolographicFrameScanoutReport
HolographicQuadLayer = _ns_module.HolographicQuadLayer
HolographicQuadLayerUpdateParameters = _ns_module.HolographicQuadLayerUpdateParameters
HolographicSpace = _ns_module.HolographicSpace
HolographicSpaceCameraAddedEventArgs = _ns_module.HolographicSpaceCameraAddedEventArgs
HolographicSpaceCameraRemovedEventArgs = _ns_module.HolographicSpaceCameraRemovedEventArgs
HolographicViewConfiguration = _ns_module.HolographicViewConfiguration