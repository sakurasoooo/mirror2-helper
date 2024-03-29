# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.devices.geolocation
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

class OfflineMapPackageQueryStatus(enum.IntEnum):
    SUCCESS = 0
    UNKNOWN_ERROR = 1
    INVALID_CREDENTIALS = 2
    NETWORK_FAILURE = 3

class OfflineMapPackageStartDownloadStatus(enum.IntEnum):
    SUCCESS = 0
    UNKNOWN_ERROR = 1
    INVALID_CREDENTIALS = 2
    DENIED_WITHOUT_CAPABILITY = 3

class OfflineMapPackageStatus(enum.IntEnum):
    NOT_DOWNLOADED = 0
    DOWNLOADING = 1
    DOWNLOADED = 2
    DELETING = 3

class OfflineMapPackage(_winrt.winrt_base):
    ...
    display_name: str
    enclosing_region_name: str
    estimated_size_in_bytes: int
    status: OfflineMapPackageStatus
    def find_packages_async(query_point: winrt.windows.devices.geolocation.Geopoint) -> winrt.windows.foundation.IAsyncOperation[OfflineMapPackageQueryResult]:
        ...
    def find_packages_in_bounding_box_async(query_bounding_box: winrt.windows.devices.geolocation.GeoboundingBox) -> winrt.windows.foundation.IAsyncOperation[OfflineMapPackageQueryResult]:
        ...
    def find_packages_in_geocircle_async(query_circle: winrt.windows.devices.geolocation.Geocircle) -> winrt.windows.foundation.IAsyncOperation[OfflineMapPackageQueryResult]:
        ...
    def request_start_download_async() -> winrt.windows.foundation.IAsyncOperation[OfflineMapPackageStartDownloadResult]:
        ...
    def add_status_changed(value: winrt.windows.foundation.TypedEventHandler[OfflineMapPackage, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_status_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class OfflineMapPackageQueryResult(_winrt.winrt_base):
    ...
    packages: winrt.windows.foundation.collections.IVectorView[OfflineMapPackage]
    status: OfflineMapPackageQueryStatus

class OfflineMapPackageStartDownloadResult(_winrt.winrt_base):
    ...
    status: OfflineMapPackageStartDownloadStatus

