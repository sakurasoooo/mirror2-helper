# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Storage.Pickers.Provider")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.storage
except Exception:
    pass

class AddFileResult(enum.IntEnum):
    ADDED = 0
    ALREADY_ADDED = 1
    NOT_ALLOWED = 2
    UNAVAILABLE = 3

class FileSelectionMode(enum.IntEnum):
    SINGLE = 0
    MULTIPLE = 1

class SetFileNameResult(enum.IntEnum):
    SUCCEEDED = 0
    NOT_ALLOWED = 1
    UNAVAILABLE = 2

FileOpenPickerUI = _ns_module.FileOpenPickerUI
FileRemovedEventArgs = _ns_module.FileRemovedEventArgs
FileSavePickerUI = _ns_module.FileSavePickerUI
PickerClosingDeferral = _ns_module.PickerClosingDeferral
PickerClosingEventArgs = _ns_module.PickerClosingEventArgs
PickerClosingOperation = _ns_module.PickerClosingOperation
TargetFileRequest = _ns_module.TargetFileRequest
TargetFileRequestDeferral = _ns_module.TargetFileRequestDeferral
TargetFileRequestedEventArgs = _ns_module.TargetFileRequestedEventArgs