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
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class LicenseRefreshOption(enum.IntEnum):
    RUNNING_LICENSES = 0
    ALL_LICENSES = 1

class LicenseManager(_winrt.winrt_base):
    ...
    def add_license_async(license: winrt.windows.storage.streams.IBuffer) -> winrt.windows.foundation.IAsyncAction:
        ...
    def get_satisfaction_infos_async(content_ids: typing.Iterable[str], key_ids: typing.Iterable[str]) -> winrt.windows.foundation.IAsyncOperation[LicenseSatisfactionResult]:
        ...
    def refresh_licenses_async(refresh_option: LicenseRefreshOption) -> winrt.windows.foundation.IAsyncAction:
        ...

class LicenseSatisfactionInfo(_winrt.winrt_base):
    ...
    is_satisfied: bool
    satisfied_by_device: bool
    satisfied_by_install_media: bool
    satisfied_by_open_license: bool
    satisfied_by_pass: bool
    satisfied_by_signed_in_user: bool
    satisfied_by_trial: bool

class LicenseSatisfactionResult(_winrt.winrt_base):
    ...
    extended_error: winrt.windows.foundation.HResult
    license_satisfaction_infos: winrt.windows.foundation.collections.IMapView[str, LicenseSatisfactionInfo]
