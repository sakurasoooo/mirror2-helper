# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.applicationmodel.core
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
    import winrt.windows.graphics
except Exception:
    pass

try:
    import winrt.windows.media.core
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class MiracastReceiverApplySettingsStatus(enum.IntEnum):
    SUCCESS = 0
    UNKNOWN_FAILURE = 1
    MIRACAST_NOT_SUPPORTED = 2
    ACCESS_DENIED = 3
    FRIENDLY_NAME_TOO_LONG = 4
    MODEL_NAME_TOO_LONG = 5
    MODEL_NUMBER_TOO_LONG = 6
    INVALID_SETTINGS = 7

class MiracastReceiverAuthorizationMethod(enum.IntEnum):
    NONE = 0
    CONFIRM_CONNECTION = 1
    PIN_DISPLAY_IF_REQUESTED = 2
    PIN_DISPLAY_REQUIRED = 3

class MiracastReceiverDisconnectReason(enum.IntEnum):
    FINISHED = 0
    APP_SPECIFIC_ERROR = 1
    CONNECTION_NOT_ACCEPTED = 2
    DISCONNECTED_BY_USER = 3
    FAILED_TO_START_STREAMING = 4
    MEDIA_DECODING_ERROR = 5
    MEDIA_STREAMING_ERROR = 6
    MEDIA_DECRYPTION_ERROR = 7

class MiracastReceiverGameControllerDeviceUsageMode(enum.IntEnum):
    AS_GAME_CONTROLLER = 0
    AS_MOUSE_AND_KEYBOARD = 1

class MiracastReceiverListeningStatus(enum.IntEnum):
    NOT_LISTENING = 0
    LISTENING = 1
    CONNECTION_PENDING = 2
    CONNECTED = 3
    DISABLED_BY_POLICY = 4
    TEMPORARILY_DISABLED = 5

class MiracastReceiverSessionStartStatus(enum.IntEnum):
    SUCCESS = 0
    UNKNOWN_FAILURE = 1
    MIRACAST_NOT_SUPPORTED = 2
    ACCESS_DENIED = 3

class MiracastReceiverWiFiStatus(enum.IntEnum):
    MIRACAST_SUPPORT_UNDETERMINED = 0
    MIRACAST_NOT_SUPPORTED = 1
    MIRACAST_SUPPORT_NOT_OPTIMIZED = 2
    MIRACAST_SUPPORTED = 3

class MiracastTransmitterAuthorizationStatus(enum.IntEnum):
    UNDECIDED = 0
    ALLOWED = 1
    ALWAYS_PROMPT = 2
    BLOCKED = 3

class MiracastReceiver(_winrt.winrt_base):
    ...
    def clear_known_transmitters() -> None:
        ...
    def create_session(view: winrt.windows.applicationmodel.core.CoreApplicationView) -> MiracastReceiverSession:
        ...
    def create_session_async(view: winrt.windows.applicationmodel.core.CoreApplicationView) -> winrt.windows.foundation.IAsyncOperation[MiracastReceiverSession]:
        ...
    def disconnect_all_and_apply_settings(settings: MiracastReceiverSettings) -> MiracastReceiverApplySettingsResult:
        ...
    def disconnect_all_and_apply_settings_async(settings: MiracastReceiverSettings) -> winrt.windows.foundation.IAsyncOperation[MiracastReceiverApplySettingsResult]:
        ...
    def get_current_settings() -> MiracastReceiverSettings:
        ...
    def get_current_settings_async() -> winrt.windows.foundation.IAsyncOperation[MiracastReceiverSettings]:
        ...
    def get_default_settings() -> MiracastReceiverSettings:
        ...
    def get_status() -> MiracastReceiverStatus:
        ...
    def get_status_async() -> winrt.windows.foundation.IAsyncOperation[MiracastReceiverStatus]:
        ...
    def remove_known_transmitter(transmitter: MiracastTransmitter) -> None:
        ...
    def add_status_changed(handler: winrt.windows.foundation.TypedEventHandler[MiracastReceiver, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_status_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class MiracastReceiverApplySettingsResult(_winrt.winrt_base):
    ...
    extended_error: winrt.windows.foundation.HResult
    status: MiracastReceiverApplySettingsStatus

class MiracastReceiverConnection(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    cursor_image_channel: MiracastReceiverCursorImageChannel
    input_devices: MiracastReceiverInputDevices
    stream_control: MiracastReceiverStreamControl
    transmitter: MiracastTransmitter
    def close() -> None:
        ...
    def disconnect(reason: MiracastReceiverDisconnectReason) -> None:
        ...
    def disconnect(reason: MiracastReceiverDisconnectReason, message: str) -> None:
        ...
    def pause() -> None:
        ...
    def pause_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def resume() -> None:
        ...
    def resume_async() -> winrt.windows.foundation.IAsyncAction:
        ...

class MiracastReceiverConnectionCreatedEventArgs(_winrt.winrt_base):
    ...
    connection: MiracastReceiverConnection
    pin: str
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...

class MiracastReceiverCursorImageChannel(_winrt.winrt_base):
    ...
    image_stream: winrt.windows.storage.streams.IRandomAccessStreamWithContentType
    is_enabled: bool
    max_image_size: winrt.windows.graphics.SizeInt32
    position: winrt.windows.graphics.PointInt32
    def add_image_stream_changed(handler: winrt.windows.foundation.TypedEventHandler[MiracastReceiverCursorImageChannel, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_image_stream_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_position_changed(handler: winrt.windows.foundation.TypedEventHandler[MiracastReceiverCursorImageChannel, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_position_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class MiracastReceiverCursorImageChannelSettings(_winrt.winrt_base):
    ...
    max_image_size: winrt.windows.graphics.SizeInt32
    is_enabled: bool

class MiracastReceiverDisconnectedEventArgs(_winrt.winrt_base):
    ...
    connection: MiracastReceiverConnection

class MiracastReceiverGameControllerDevice(_winrt.winrt_base):
    ...
    transmit_input: bool
    mode: MiracastReceiverGameControllerDeviceUsageMode
    is_requested_by_transmitter: bool
    is_transmitting_input: bool
    def add_changed(handler: winrt.windows.foundation.TypedEventHandler[MiracastReceiverGameControllerDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class MiracastReceiverInputDevices(_winrt.winrt_base):
    ...
    game_controller: MiracastReceiverGameControllerDevice
    keyboard: MiracastReceiverKeyboardDevice

class MiracastReceiverKeyboardDevice(_winrt.winrt_base):
    ...
    transmit_input: bool
    is_requested_by_transmitter: bool
    is_transmitting_input: bool
    def add_changed(handler: winrt.windows.foundation.TypedEventHandler[MiracastReceiverKeyboardDevice, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class MiracastReceiverMediaSourceCreatedEventArgs(_winrt.winrt_base):
    ...
    connection: MiracastReceiverConnection
    cursor_image_channel_settings: MiracastReceiverCursorImageChannelSettings
    media_source: winrt.windows.media.core.MediaSource
    def get_deferral() -> winrt.windows.foundation.Deferral:
        ...

class MiracastReceiverSession(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    max_simultaneous_connections: int
    allow_connection_takeover: bool
    def close() -> None:
        ...
    def start() -> MiracastReceiverSessionStartResult:
        ...
    def start_async() -> winrt.windows.foundation.IAsyncOperation[MiracastReceiverSessionStartResult]:
        ...
    def add_connection_created(handler: winrt.windows.foundation.TypedEventHandler[MiracastReceiverSession, MiracastReceiverConnectionCreatedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_connection_created(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_disconnected(handler: winrt.windows.foundation.TypedEventHandler[MiracastReceiverSession, MiracastReceiverDisconnectedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_disconnected(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_media_source_created(handler: winrt.windows.foundation.TypedEventHandler[MiracastReceiverSession, MiracastReceiverMediaSourceCreatedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_media_source_created(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class MiracastReceiverSessionStartResult(_winrt.winrt_base):
    ...
    extended_error: winrt.windows.foundation.HResult
    status: MiracastReceiverSessionStartStatus

class MiracastReceiverSettings(_winrt.winrt_base):
    ...
    require_authorization_from_known_transmitters: bool
    model_number: str
    model_name: str
    friendly_name: str
    authorization_method: MiracastReceiverAuthorizationMethod

class MiracastReceiverStatus(_winrt.winrt_base):
    ...
    is_connection_takeover_supported: bool
    known_transmitters: winrt.windows.foundation.collections.IVectorView[MiracastTransmitter]
    listening_status: MiracastReceiverListeningStatus
    max_simultaneous_connections: int
    wi_fi_status: MiracastReceiverWiFiStatus

class MiracastReceiverStreamControl(_winrt.winrt_base):
    ...
    mute_audio: bool
    def get_video_stream_settings() -> MiracastReceiverVideoStreamSettings:
        ...
    def get_video_stream_settings_async() -> winrt.windows.foundation.IAsyncOperation[MiracastReceiverVideoStreamSettings]:
        ...
    def suggest_video_stream_settings(settings: MiracastReceiverVideoStreamSettings) -> None:
        ...
    def suggest_video_stream_settings_async(settings: MiracastReceiverVideoStreamSettings) -> winrt.windows.foundation.IAsyncAction:
        ...

class MiracastReceiverVideoStreamSettings(_winrt.winrt_base):
    ...
    size: winrt.windows.graphics.SizeInt32
    bitrate: int

class MiracastTransmitter(_winrt.winrt_base):
    ...
    name: str
    authorization_status: MiracastTransmitterAuthorizationStatus
    last_connection_time: winrt.windows.foundation.DateTime
    mac_address: str
    def get_connections() -> winrt.windows.foundation.collections.IVectorView[MiracastReceiverConnection]:
        ...

