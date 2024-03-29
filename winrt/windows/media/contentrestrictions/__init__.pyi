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

class ContentAccessRestrictionLevel(enum.IntEnum):
    ALLOW = 0
    WARN = 1
    BLOCK = 2
    HIDE = 3

class RatedContentCategory(enum.IntEnum):
    GENERAL = 0
    APPLICATION = 1
    GAME = 2
    MOVIE = 3
    TELEVISION = 4
    MUSIC = 5

class ContentRestrictionsBrowsePolicy(_winrt.winrt_base):
    ...
    geographic_region: str
    max_browsable_age_rating: typing.Optional[int]
    preferred_age_rating: typing.Optional[int]

class RatedContentDescription(_winrt.winrt_base):
    ...
    title: str
    ratings: winrt.windows.foundation.collections.IVector[str]
    image: winrt.windows.storage.streams.IRandomAccessStreamReference
    id: str
    category: RatedContentCategory

class RatedContentRestrictions(_winrt.winrt_base):
    ...
    def get_browse_policy_async() -> winrt.windows.foundation.IAsyncOperation[ContentRestrictionsBrowsePolicy]:
        ...
    def get_restriction_level_async(rated_content_description: RatedContentDescription) -> winrt.windows.foundation.IAsyncOperation[ContentAccessRestrictionLevel]:
        ...
    def request_content_access_async(rated_content_description: RatedContentDescription) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def add_restrictions_changed(handler: winrt.windows.foundation.EventHandler[_winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_restrictions_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

