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
    import winrt.windows.media
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class GlobalSystemMediaTransportControlsSessionPlaybackStatus(enum.IntEnum):
    CLOSED = 0
    OPENED = 1
    CHANGING = 2
    STOPPED = 3
    PLAYING = 4
    PAUSED = 5

class CurrentSessionChangedEventArgs(_winrt.winrt_base):
    ...

class GlobalSystemMediaTransportControlsSession(_winrt.winrt_base):
    ...
    source_app_user_model_id: str
    def get_playback_info() -> GlobalSystemMediaTransportControlsSessionPlaybackInfo:
        ...
    def get_timeline_properties() -> GlobalSystemMediaTransportControlsSessionTimelineProperties:
        ...
    def try_change_auto_repeat_mode_async(requested_auto_repeat_mode: winrt.windows.media.MediaPlaybackAutoRepeatMode) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_change_channel_down_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_change_channel_up_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_change_playback_position_async(requested_playback_position: int) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_change_playback_rate_async(requested_playback_rate: float) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_change_shuffle_active_async(requested_shuffle_state: bool) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_fast_forward_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_get_media_properties_async() -> winrt.windows.foundation.IAsyncOperation[GlobalSystemMediaTransportControlsSessionMediaProperties]:
        ...
    def try_pause_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_play_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_record_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_rewind_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_skip_next_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_skip_previous_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_stop_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def try_toggle_play_pause_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def add_media_properties_changed(handler: winrt.windows.foundation.TypedEventHandler[GlobalSystemMediaTransportControlsSession, MediaPropertiesChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_media_properties_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_playback_info_changed(handler: winrt.windows.foundation.TypedEventHandler[GlobalSystemMediaTransportControlsSession, PlaybackInfoChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_playback_info_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_timeline_properties_changed(handler: winrt.windows.foundation.TypedEventHandler[GlobalSystemMediaTransportControlsSession, TimelinePropertiesChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_timeline_properties_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GlobalSystemMediaTransportControlsSessionManager(_winrt.winrt_base):
    ...
    def get_current_session() -> GlobalSystemMediaTransportControlsSession:
        ...
    def get_sessions() -> winrt.windows.foundation.collections.IVectorView[GlobalSystemMediaTransportControlsSession]:
        ...
    def request_async() -> winrt.windows.foundation.IAsyncOperation[GlobalSystemMediaTransportControlsSessionManager]:
        ...
    def add_current_session_changed(handler: winrt.windows.foundation.TypedEventHandler[GlobalSystemMediaTransportControlsSessionManager, CurrentSessionChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_current_session_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_sessions_changed(handler: winrt.windows.foundation.TypedEventHandler[GlobalSystemMediaTransportControlsSessionManager, SessionsChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_sessions_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class GlobalSystemMediaTransportControlsSessionMediaProperties(_winrt.winrt_base):
    ...
    album_artist: str
    album_title: str
    album_track_count: int
    artist: str
    genres: winrt.windows.foundation.collections.IVectorView[str]
    playback_type: typing.Optional[winrt.windows.media.MediaPlaybackType]
    subtitle: str
    thumbnail: winrt.windows.storage.streams.IRandomAccessStreamReference
    title: str
    track_number: int

class GlobalSystemMediaTransportControlsSessionPlaybackControls(_winrt.winrt_base):
    ...
    is_channel_down_enabled: bool
    is_channel_up_enabled: bool
    is_fast_forward_enabled: bool
    is_next_enabled: bool
    is_pause_enabled: bool
    is_play_enabled: bool
    is_play_pause_toggle_enabled: bool
    is_playback_position_enabled: bool
    is_playback_rate_enabled: bool
    is_previous_enabled: bool
    is_record_enabled: bool
    is_repeat_enabled: bool
    is_rewind_enabled: bool
    is_shuffle_enabled: bool
    is_stop_enabled: bool

class GlobalSystemMediaTransportControlsSessionPlaybackInfo(_winrt.winrt_base):
    ...
    auto_repeat_mode: typing.Optional[winrt.windows.media.MediaPlaybackAutoRepeatMode]
    controls: GlobalSystemMediaTransportControlsSessionPlaybackControls
    is_shuffle_active: typing.Optional[bool]
    playback_rate: typing.Optional[float]
    playback_status: GlobalSystemMediaTransportControlsSessionPlaybackStatus
    playback_type: typing.Optional[winrt.windows.media.MediaPlaybackType]

class GlobalSystemMediaTransportControlsSessionTimelineProperties(_winrt.winrt_base):
    ...
    end_time: winrt.windows.foundation.TimeSpan
    last_updated_time: winrt.windows.foundation.DateTime
    max_seek_time: winrt.windows.foundation.TimeSpan
    min_seek_time: winrt.windows.foundation.TimeSpan
    position: winrt.windows.foundation.TimeSpan
    start_time: winrt.windows.foundation.TimeSpan

class MediaPropertiesChangedEventArgs(_winrt.winrt_base):
    ...

class PlaybackInfoChangedEventArgs(_winrt.winrt_base):
    ...

class SessionsChangedEventArgs(_winrt.winrt_base):
    ...

class TimelinePropertiesChangedEventArgs(_winrt.winrt_base):
    ...
