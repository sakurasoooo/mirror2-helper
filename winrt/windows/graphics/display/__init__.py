# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Graphics.Display")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.graphics
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class AdvancedColorKind(enum.IntEnum):
    STANDARD_DYNAMIC_RANGE = 0
    WIDE_COLOR_GAMUT = 1
    HIGH_DYNAMIC_RANGE = 2

class DisplayBrightnessOverrideOptions(enum.IntFlag):
    NONE = 0
    USE_DIMMED_POLICY_WHEN_BATTERY_IS_LOW = 0x1

class DisplayBrightnessOverrideScenario(enum.IntEnum):
    IDLE_BRIGHTNESS = 0
    BARCODE_READING_BRIGHTNESS = 1
    FULL_BRIGHTNESS = 2

class DisplayBrightnessScenario(enum.IntEnum):
    DEFAULT_BRIGHTNESS = 0
    IDLE_BRIGHTNESS = 1
    BARCODE_READING_BRIGHTNESS = 2
    FULL_BRIGHTNESS = 3

class DisplayColorOverrideScenario(enum.IntEnum):
    ACCURATE = 0

class DisplayOrientations(enum.IntFlag):
    NONE = 0
    LANDSCAPE = 0x1
    PORTRAIT = 0x2
    LANDSCAPE_FLIPPED = 0x4
    PORTRAIT_FLIPPED = 0x8

class HdrMetadataFormat(enum.IntEnum):
    HDR10 = 0
    HDR10_PLUS = 1

class ResolutionScale(enum.IntEnum):
    INVALID = 0
    SCALE100_PERCENT = 100
    SCALE120_PERCENT = 120
    SCALE125_PERCENT = 125
    SCALE140_PERCENT = 140
    SCALE150_PERCENT = 150
    SCALE160_PERCENT = 160
    SCALE175_PERCENT = 175
    SCALE180_PERCENT = 180
    SCALE200_PERCENT = 200
    SCALE225_PERCENT = 225
    SCALE250_PERCENT = 250
    SCALE300_PERCENT = 300
    SCALE350_PERCENT = 350
    SCALE400_PERCENT = 400
    SCALE450_PERCENT = 450
    SCALE500_PERCENT = 500

NitRange = _ns_module.NitRange
AdvancedColorInfo = _ns_module.AdvancedColorInfo
BrightnessOverride = _ns_module.BrightnessOverride
BrightnessOverrideSettings = _ns_module.BrightnessOverrideSettings
ColorOverrideSettings = _ns_module.ColorOverrideSettings
DisplayEnhancementOverride = _ns_module.DisplayEnhancementOverride
DisplayEnhancementOverrideCapabilities = _ns_module.DisplayEnhancementOverrideCapabilities
DisplayEnhancementOverrideCapabilitiesChangedEventArgs = _ns_module.DisplayEnhancementOverrideCapabilitiesChangedEventArgs
DisplayInformation = _ns_module.DisplayInformation
DisplayProperties = _ns_module.DisplayProperties
DisplayServices = _ns_module.DisplayServices
