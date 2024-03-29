# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

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
    import winrt.windows.globalization
except Exception:
    pass

try:
    import winrt.windows.storage
except Exception:
    pass

try:
    import winrt.windows.system
except Exception:
    pass

class AdvertisingManager(_winrt.winrt_base):
    ...
    advertising_id: str
    def get_for_user(user: winrt.windows.system.User) -> AdvertisingManagerForUser:
        ...

class AdvertisingManagerForUser(_winrt.winrt_base):
    ...
    advertising_id: str
    user: winrt.windows.system.User

class AssignedAccessSettings(_winrt.winrt_base):
    ...
    is_enabled: bool
    is_single_app_kiosk_mode: bool
    user: winrt.windows.system.User
    def get_default() -> AssignedAccessSettings:
        ...
    def get_for_user(user: winrt.windows.system.User) -> AssignedAccessSettings:
        ...

class DiagnosticsSettings(_winrt.winrt_base):
    ...
    can_use_diagnostics_to_tailor_experiences: bool
    user: winrt.windows.system.User
    def get_default() -> DiagnosticsSettings:
        ...
    def get_for_user(user: winrt.windows.system.User) -> DiagnosticsSettings:
        ...

class FirstSignInSettings(winrt.windows.foundation.collections.IMapView[str, _winrt.winrt_base], winrt.windows.foundation.collections.IIterable[winrt.windows.foundation.collections.IKeyValuePair[str, _winrt.winrt_base]], _winrt.winrt_base):
    ...
    size: int
    def first() -> winrt.windows.foundation.collections.IIterator[winrt.windows.foundation.collections.IKeyValuePair[str, _winrt.winrt_base]]:
        ...
    def get_default() -> FirstSignInSettings:
        ...
    def has_key(key: str) -> bool:
        ...
    def lookup(key: str) -> _winrt.winrt_base:
        ...
    def split() -> typing.Tuple[winrt.windows.foundation.collections.IMapView[str, _winrt.winrt_base], winrt.windows.foundation.collections.IMapView[str, _winrt.winrt_base]]:
        ...

class GlobalizationPreferences(_winrt.winrt_base):
    ...
    calendars: winrt.windows.foundation.collections.IVectorView[str]
    clocks: winrt.windows.foundation.collections.IVectorView[str]
    currencies: winrt.windows.foundation.collections.IVectorView[str]
    home_geographic_region: str
    languages: winrt.windows.foundation.collections.IVectorView[str]
    week_starts_on: winrt.windows.globalization.DayOfWeek
    def get_for_user(user: winrt.windows.system.User) -> GlobalizationPreferencesForUser:
        ...
    def try_set_home_geographic_region(region: str) -> bool:
        ...
    def try_set_languages(language_tags: typing.Iterable[str]) -> bool:
        ...

class GlobalizationPreferencesForUser(_winrt.winrt_base):
    ...
    calendars: winrt.windows.foundation.collections.IVectorView[str]
    clocks: winrt.windows.foundation.collections.IVectorView[str]
    currencies: winrt.windows.foundation.collections.IVectorView[str]
    home_geographic_region: str
    languages: winrt.windows.foundation.collections.IVectorView[str]
    user: winrt.windows.system.User
    week_starts_on: winrt.windows.globalization.DayOfWeek

class UserProfilePersonalizationSettings(_winrt.winrt_base):
    ...
    current: UserProfilePersonalizationSettings
    def is_supported() -> bool:
        ...
    def try_set_lock_screen_image_async(image_file: winrt.windows.storage.StorageFile) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_set_wallpaper_image_async(image_file: winrt.windows.storage.StorageFile) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...

