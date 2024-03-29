# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.media.capture
except Exception:
    pass

class VariablePhotoCapturedEventArgs(_winrt.winrt_base):
    ...
    capture_time_offset: winrt.windows.foundation.TimeSpan
    captured_frame_control_values: winrt.windows.media.capture.CapturedFrameControlValues
    frame: winrt.windows.media.capture.CapturedFrame
    used_frame_controller_index: typing.Optional[int]

class VariablePhotoSequenceCapture(_winrt.winrt_base):
    ...
    def finish_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def start_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def stop_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def update_settings_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def add_photo_captured(handler: winrt.windows.foundation.TypedEventHandler[VariablePhotoSequenceCapture, VariablePhotoCapturedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_photo_captured(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_stopped(handler: winrt.windows.foundation.TypedEventHandler[VariablePhotoSequenceCapture, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_stopped(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

