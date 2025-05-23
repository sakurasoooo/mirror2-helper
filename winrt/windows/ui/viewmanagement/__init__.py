# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.UI.ViewManagement")

try:
    import winrt.windows.devices.enumeration
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
    import winrt.windows.ui
except Exception:
    pass

try:
    import winrt.windows.ui.core
except Exception:
    pass

try:
    import winrt.windows.ui.popups
except Exception:
    pass

try:
    import winrt.windows.ui.windowmanagement
except Exception:
    pass

class ApplicationViewBoundsMode(enum.IntEnum):
    USE_VISIBLE = 0
    USE_CORE_WINDOW = 1

class ApplicationViewMode(enum.IntEnum):
    DEFAULT = 0
    COMPACT_OVERLAY = 1

class ApplicationViewOrientation(enum.IntEnum):
    LANDSCAPE = 0
    PORTRAIT = 1

class ApplicationViewState(enum.IntEnum):
    FULL_SCREEN_LANDSCAPE = 0
    FILLED = 1
    SNAPPED = 2
    FULL_SCREEN_PORTRAIT = 3

class ApplicationViewSwitchingOptions(enum.IntFlag):
    DEFAULT = 0
    SKIP_ANIMATION = 0x1
    CONSOLIDATE_VIEWS = 0x2

class ApplicationViewWindowingMode(enum.IntEnum):
    AUTO = 0
    PREFERRED_LAUNCH_VIEW_SIZE = 1
    FULL_SCREEN = 2
    COMPACT_OVERLAY = 3
    MAXIMIZED = 4

class FullScreenSystemOverlayMode(enum.IntEnum):
    STANDARD = 0
    MINIMAL = 1

class HandPreference(enum.IntEnum):
    LEFT_HANDED = 0
    RIGHT_HANDED = 1

class UIColorType(enum.IntEnum):
    BACKGROUND = 0
    FOREGROUND = 1
    ACCENT_DARK3 = 2
    ACCENT_DARK2 = 3
    ACCENT_DARK1 = 4
    ACCENT = 5
    ACCENT_LIGHT1 = 6
    ACCENT_LIGHT2 = 7
    ACCENT_LIGHT3 = 8
    COMPLEMENT = 9

class UIElementType(enum.IntEnum):
    ACTIVE_CAPTION = 0
    BACKGROUND = 1
    BUTTON_FACE = 2
    BUTTON_TEXT = 3
    CAPTION_TEXT = 4
    GRAY_TEXT = 5
    HIGHLIGHT = 6
    HIGHLIGHT_TEXT = 7
    HOTLIGHT = 8
    INACTIVE_CAPTION = 9
    INACTIVE_CAPTION_TEXT = 10
    WINDOW = 11
    WINDOW_TEXT = 12
    ACCENT_COLOR = 1000
    TEXT_HIGH = 1001
    TEXT_MEDIUM = 1002
    TEXT_LOW = 1003
    TEXT_CONTRAST_WITH_HIGH = 1004
    NON_TEXT_HIGH = 1005
    NON_TEXT_MEDIUM_HIGH = 1006
    NON_TEXT_MEDIUM = 1007
    NON_TEXT_MEDIUM_LOW = 1008
    NON_TEXT_LOW = 1009
    PAGE_BACKGROUND = 1010
    POPUP_BACKGROUND = 1011
    OVERLAY_OUTSIDE_POPUP = 1012

class UserInteractionMode(enum.IntEnum):
    MOUSE = 0
    TOUCH = 1

class ViewSizePreference(enum.IntEnum):
    DEFAULT = 0
    USE_LESS = 1
    USE_HALF = 2
    USE_MORE = 3
    USE_MINIMUM = 4
    USE_NONE = 5
    CUSTOM = 6

AccessibilitySettings = _ns_module.AccessibilitySettings
ActivationViewSwitcher = _ns_module.ActivationViewSwitcher
ApplicationView = _ns_module.ApplicationView
ApplicationViewConsolidatedEventArgs = _ns_module.ApplicationViewConsolidatedEventArgs
ApplicationViewScaling = _ns_module.ApplicationViewScaling
ApplicationViewSwitcher = _ns_module.ApplicationViewSwitcher
ApplicationViewTitleBar = _ns_module.ApplicationViewTitleBar
ApplicationViewTransferContext = _ns_module.ApplicationViewTransferContext
InputPane = _ns_module.InputPane
InputPaneVisibilityEventArgs = _ns_module.InputPaneVisibilityEventArgs
ProjectionManager = _ns_module.ProjectionManager
UISettings = _ns_module.UISettings
UISettingsAnimationsEnabledChangedEventArgs = _ns_module.UISettingsAnimationsEnabledChangedEventArgs
UISettingsAutoHideScrollBarsChangedEventArgs = _ns_module.UISettingsAutoHideScrollBarsChangedEventArgs
UISettingsMessageDurationChangedEventArgs = _ns_module.UISettingsMessageDurationChangedEventArgs
UIViewSettings = _ns_module.UIViewSettings
ViewModePreferences = _ns_module.ViewModePreferences
