# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum
import typing
import uuid

import winrt._winrt as _winrt
try:
    import winrt.windows.devices.enumeration
except Exception:
    pass

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class MidiMessageType(enum.IntEnum):
    NONE = 0
    NOTE_OFF = 128
    NOTE_ON = 144
    POLYPHONIC_KEY_PRESSURE = 160
    CONTROL_CHANGE = 176
    PROGRAM_CHANGE = 192
    CHANNEL_PRESSURE = 208
    PITCH_BEND_CHANGE = 224
    SYSTEM_EXCLUSIVE = 240
    MIDI_TIME_CODE = 241
    SONG_POSITION_POINTER = 242
    SONG_SELECT = 243
    TUNE_REQUEST = 246
    END_SYSTEM_EXCLUSIVE = 247
    TIMING_CLOCK = 248
    START = 250
    CONTINUE = 251
    STOP = 252
    ACTIVE_SENSING = 254
    SYSTEM_RESET = 255

class MidiActiveSensingMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class MidiChannelPressureMessage(IMidiMessage, _winrt.winrt_base):
    ...
    channel: int
    pressure: int
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class MidiContinueMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class MidiControlChangeMessage(IMidiMessage, _winrt.winrt_base):
    ...
    channel: int
    control_value: int
    controller: int
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class MidiInPort(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    device_id: str
    def close() -> None:
        ...
    def from_id_async(device_id: str) -> winrt.windows.foundation.IAsyncOperation[MidiInPort]:
        ...
    def get_device_selector() -> str:
        ...
    def add_message_received(handler: winrt.windows.foundation.TypedEventHandler[MidiInPort, MidiMessageReceivedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_message_received(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class MidiMessageReceivedEventArgs(_winrt.winrt_base):
    ...
    message: IMidiMessage

class MidiNoteOffMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType
    channel: int
    note: int
    velocity: int

class MidiNoteOnMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType
    channel: int
    note: int
    velocity: int

class MidiOutPort(IMidiOutPort, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    device_id: str
    def close() -> None:
        ...
    def from_id_async(device_id: str) -> winrt.windows.foundation.IAsyncOperation[IMidiOutPort]:
        ...
    def get_device_selector() -> str:
        ...
    def send_buffer(midi_data: winrt.windows.storage.streams.IBuffer) -> None:
        ...
    def send_message(midi_message: IMidiMessage) -> None:
        ...

class MidiPitchBendChangeMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType
    bend: int
    channel: int

class MidiPolyphonicKeyPressureMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType
    channel: int
    note: int
    pressure: int

class MidiProgramChangeMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType
    channel: int
    program: int

class MidiSongPositionPointerMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType
    beats: int

class MidiSongSelectMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType
    song: int

class MidiStartMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class MidiStopMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class MidiSynthesizer(IMidiOutPort, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    device_id: str
    volume: float
    audio_device: winrt.windows.devices.enumeration.DeviceInformation
    def close() -> None:
        ...
    def create_async() -> winrt.windows.foundation.IAsyncOperation[MidiSynthesizer]:
        ...
    def create_async(audio_device: winrt.windows.devices.enumeration.DeviceInformation) -> winrt.windows.foundation.IAsyncOperation[MidiSynthesizer]:
        ...
    def is_synthesizer(midi_device: winrt.windows.devices.enumeration.DeviceInformation) -> bool:
        ...
    def send_buffer(midi_data: winrt.windows.storage.streams.IBuffer) -> None:
        ...
    def send_message(midi_message: IMidiMessage) -> None:
        ...

class MidiSystemExclusiveMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class MidiSystemResetMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class MidiTimeCodeMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType
    frame_type: int
    values: int

class MidiTimingClockMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class MidiTuneRequestMessage(IMidiMessage, _winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class IMidiMessage(_winrt.winrt_base):
    ...
    raw_data: winrt.windows.storage.streams.IBuffer
    timestamp: winrt.windows.foundation.TimeSpan
    type: MidiMessageType

class IMidiOutPort(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    device_id: str
    def send_buffer(midi_data: winrt.windows.storage.streams.IBuffer) -> None:
        ...
    def send_message(midi_message: IMidiMessage) -> None:
        ...
    def close() -> None:
        ...
