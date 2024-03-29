# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.System.Update")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

class SystemUpdateAttentionRequiredReason(enum.IntEnum):
    NONE = 0
    NETWORK_REQUIRED = 1
    INSUFFICIENT_DISK_SPACE = 2
    INSUFFICIENT_BATTERY = 3
    UPDATE_BLOCKED = 4

class SystemUpdateItemState(enum.IntEnum):
    NOT_STARTED = 0
    INITIALIZING = 1
    PREPARING = 2
    CALCULATING = 3
    DOWNLOADING = 4
    INSTALLING = 5
    COMPLETED = 6
    REBOOT_REQUIRED = 7
    ERROR = 8

class SystemUpdateManagerState(enum.IntEnum):
    IDLE = 0
    DETECTING = 1
    READY_TO_DOWNLOAD = 2
    DOWNLOADING = 3
    READY_TO_INSTALL = 4
    INSTALLING = 5
    REBOOT_REQUIRED = 6
    READY_TO_FINALIZE = 7
    FINALIZING = 8
    COMPLETED = 9
    ATTENTION_REQUIRED = 10
    ERROR = 11

class SystemUpdateStartInstallAction(enum.IntEnum):
    UP_TO_REBOOT = 0
    ALLOW_REBOOT = 1

SystemUpdateItem = _ns_module.SystemUpdateItem
SystemUpdateLastErrorInfo = _ns_module.SystemUpdateLastErrorInfo
SystemUpdateManager = _ns_module.SystemUpdateManager
