# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Devices.Printers")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class IppAttributeErrorReason(enum.IntEnum):
    REQUEST_ENTITY_TOO_LARGE = 0
    ATTRIBUTE_NOT_SUPPORTED = 1
    ATTRIBUTE_VALUES_NOT_SUPPORTED = 2
    ATTRIBUTE_NOT_SETTABLE = 3
    CONFLICTING_ATTRIBUTES = 4

class IppAttributeValueKind(enum.IntEnum):
    UNSUPPORTED = 0
    UNKNOWN = 1
    NO_VALUE = 2
    INTEGER = 3
    BOOLEAN = 4
    ENUM = 5
    OCTET_STRING = 6
    DATE_TIME = 7
    RESOLUTION = 8
    RANGE_OF_INTEGER = 9
    COLLECTION = 10
    TEXT_WITH_LANGUAGE = 11
    NAME_WITH_LANGUAGE = 12
    TEXT_WITHOUT_LANGUAGE = 13
    NAME_WITHOUT_LANGUAGE = 14
    KEYWORD = 15
    URI = 16
    URI_SCHEMA = 17
    CHARSET = 18
    NATURAL_LANGUAGE = 19
    MIME_MEDIA_TYPE = 20

class IppResolutionUnit(enum.IntEnum):
    DOTS_PER_INCH = 0
    DOTS_PER_CENTIMETER = 1

IppAttributeError = _ns_module.IppAttributeError
IppAttributeValue = _ns_module.IppAttributeValue
IppIntegerRange = _ns_module.IppIntegerRange
IppPrintDevice = _ns_module.IppPrintDevice
IppResolution = _ns_module.IppResolution
IppSetAttributesResult = _ns_module.IppSetAttributesResult
IppTextWithLanguage = _ns_module.IppTextWithLanguage
Print3DDevice = _ns_module.Print3DDevice
PrintSchema = _ns_module.PrintSchema
