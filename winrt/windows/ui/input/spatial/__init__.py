# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.UI.Input.Spatial")

try:
    import winrt.windows.devices.haptics
except Exception:
    pass

try:
    import winrt.windows.devices.power
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
    import winrt.windows.foundation.numerics
except Exception:
    pass

try:
    import winrt.windows.perception
except Exception:
    pass

try:
    import winrt.windows.perception.people
except Exception:
    pass

try:
    import winrt.windows.perception.spatial
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class SpatialGestureSettings(enum.IntFlag):
    NONE = 0
    TAP = 0x1
    DOUBLE_TAP = 0x2
    HOLD = 0x4
    MANIPULATION_TRANSLATE = 0x8
    NAVIGATION_X = 0x10
    NAVIGATION_Y = 0x20
    NAVIGATION_Z = 0x40
    NAVIGATION_RAILS_X = 0x80
    NAVIGATION_RAILS_Y = 0x100
    NAVIGATION_RAILS_Z = 0x200

class SpatialInteractionPressKind(enum.IntEnum):
    NONE = 0
    SELECT = 1
    MENU = 2
    GRASP = 3
    TOUCHPAD = 4
    THUMBSTICK = 5

class SpatialInteractionSourceHandedness(enum.IntEnum):
    UNSPECIFIED = 0
    LEFT = 1
    RIGHT = 2

class SpatialInteractionSourceKind(enum.IntEnum):
    OTHER = 0
    HAND = 1
    VOICE = 2
    CONTROLLER = 3

class SpatialInteractionSourcePositionAccuracy(enum.IntEnum):
    HIGH = 0
    APPROXIMATE = 1

SpatialGestureRecognizer = _ns_module.SpatialGestureRecognizer
SpatialHoldCanceledEventArgs = _ns_module.SpatialHoldCanceledEventArgs
SpatialHoldCompletedEventArgs = _ns_module.SpatialHoldCompletedEventArgs
SpatialHoldStartedEventArgs = _ns_module.SpatialHoldStartedEventArgs
SpatialInteraction = _ns_module.SpatialInteraction
SpatialInteractionController = _ns_module.SpatialInteractionController
SpatialInteractionControllerProperties = _ns_module.SpatialInteractionControllerProperties
SpatialInteractionDetectedEventArgs = _ns_module.SpatialInteractionDetectedEventArgs
SpatialInteractionManager = _ns_module.SpatialInteractionManager
SpatialInteractionSource = _ns_module.SpatialInteractionSource
SpatialInteractionSourceEventArgs = _ns_module.SpatialInteractionSourceEventArgs
SpatialInteractionSourceLocation = _ns_module.SpatialInteractionSourceLocation
SpatialInteractionSourceProperties = _ns_module.SpatialInteractionSourceProperties
SpatialInteractionSourceState = _ns_module.SpatialInteractionSourceState
SpatialManipulationCanceledEventArgs = _ns_module.SpatialManipulationCanceledEventArgs
SpatialManipulationCompletedEventArgs = _ns_module.SpatialManipulationCompletedEventArgs
SpatialManipulationDelta = _ns_module.SpatialManipulationDelta
SpatialManipulationStartedEventArgs = _ns_module.SpatialManipulationStartedEventArgs
SpatialManipulationUpdatedEventArgs = _ns_module.SpatialManipulationUpdatedEventArgs
SpatialNavigationCanceledEventArgs = _ns_module.SpatialNavigationCanceledEventArgs
SpatialNavigationCompletedEventArgs = _ns_module.SpatialNavigationCompletedEventArgs
SpatialNavigationStartedEventArgs = _ns_module.SpatialNavigationStartedEventArgs
SpatialNavigationUpdatedEventArgs = _ns_module.SpatialNavigationUpdatedEventArgs
SpatialPointerInteractionSourcePose = _ns_module.SpatialPointerInteractionSourcePose
SpatialPointerPose = _ns_module.SpatialPointerPose
SpatialRecognitionEndedEventArgs = _ns_module.SpatialRecognitionEndedEventArgs
SpatialRecognitionStartedEventArgs = _ns_module.SpatialRecognitionStartedEventArgs
SpatialTappedEventArgs = _ns_module.SpatialTappedEventArgs
