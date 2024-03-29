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
    import winrt.windows.media.core
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class SpeechAppendedSilence(enum.IntEnum):
    DEFAULT = 0
    MIN = 1

class SpeechPunctuationSilence(enum.IntEnum):
    DEFAULT = 0
    MIN = 1

class VoiceGender(enum.IntEnum):
    MALE = 0
    FEMALE = 1

class SpeechSynthesisStream(winrt.windows.storage.streams.IRandomAccessStreamWithContentType, winrt.windows.storage.streams.IContentTypeProvider, winrt.windows.storage.streams.IRandomAccessStream, winrt.windows.storage.streams.IOutputStream, winrt.windows.foundation.IClosable, winrt.windows.storage.streams.IInputStream, winrt.windows.media.core.ITimedMetadataTrackProvider, _winrt.winrt_base):
    ...
    timed_metadata_tracks: winrt.windows.foundation.collections.IVectorView[winrt.windows.media.core.TimedMetadataTrack]
    markers: winrt.windows.foundation.collections.IVectorView[winrt.windows.media.IMediaMarker]
    content_type: str
    size: int
    can_read: bool
    can_write: bool
    position: int
    def clone_stream() -> winrt.windows.storage.streams.IRandomAccessStream:
        ...
    def close() -> None:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def get_input_stream_at(position: int) -> winrt.windows.storage.streams.IInputStream:
        ...
    def get_output_stream_at(position: int) -> winrt.windows.storage.streams.IOutputStream:
        ...
    def read_async(buffer: winrt.windows.storage.streams.IBuffer, count: int, options: winrt.windows.storage.streams.InputStreamOptions) -> winrt.windows.foundation.IAsyncOperationWithProgress[winrt.windows.storage.streams.IBuffer, int]:
        ...
    def seek(position: int) -> None:
        ...
    def write_async(buffer: winrt.windows.storage.streams.IBuffer) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...

class SpeechSynthesizer(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    voice: VoiceInformation
    options: SpeechSynthesizerOptions
    all_voices: winrt.windows.foundation.collections.IVectorView[VoiceInformation]
    default_voice: VoiceInformation
    def close() -> None:
        ...
    def synthesize_ssml_to_stream_async(ssml: str) -> winrt.windows.foundation.IAsyncOperation[SpeechSynthesisStream]:
        ...
    def synthesize_text_to_stream_async(text: str) -> winrt.windows.foundation.IAsyncOperation[SpeechSynthesisStream]:
        ...
    def try_set_default_voice_async(voice: VoiceInformation) -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...

class SpeechSynthesizerOptions(_winrt.winrt_base):
    ...
    include_word_boundary_metadata: bool
    include_sentence_boundary_metadata: bool
    speaking_rate: float
    audio_volume: float
    audio_pitch: float
    punctuation_silence: SpeechPunctuationSilence
    appended_silence: SpeechAppendedSilence

class VoiceInformation(_winrt.winrt_base):
    ...
    description: str
    display_name: str
    gender: VoiceGender
    id: str
    language: str

