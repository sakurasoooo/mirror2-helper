# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Media.Effects")

try:
    import winrt.windows.foundation
except Exception:
    pass

try:
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.foundation.numerics
except Exception:
    pass

try:
    import winrt.windows.graphics.directx.direct3d11
except Exception:
    pass

try:
    import winrt.windows.media
except Exception:
    pass

try:
    import winrt.windows.media.capture
except Exception:
    pass

try:
    import winrt.windows.media.editing
except Exception:
    pass

try:
    import winrt.windows.media.mediaproperties
except Exception:
    pass

try:
    import winrt.windows.media.playback
except Exception:
    pass

try:
    import winrt.windows.media.render
except Exception:
    pass

try:
    import winrt.windows.media.transcoding
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

try:
    import winrt.windows.ui
except Exception:
    pass

class AudioEffectType(enum.IntEnum):
    OTHER = 0
    ACOUSTIC_ECHO_CANCELLATION = 1
    NOISE_SUPPRESSION = 2
    AUTOMATIC_GAIN_CONTROL = 3
    BEAM_FORMING = 4
    CONSTANT_TONE_REMOVAL = 5
    EQUALIZER = 6
    LOUDNESS_EQUALIZER = 7
    BASS_BOOST = 8
    VIRTUAL_SURROUND = 9
    VIRTUAL_HEADPHONES = 10
    SPEAKER_FILL = 11
    ROOM_CORRECTION = 12
    BASS_MANAGEMENT = 13
    ENVIRONMENTAL_EFFECTS = 14
    SPEAKER_PROTECTION = 15
    SPEAKER_COMPENSATION = 16
    DYNAMIC_RANGE_COMPRESSION = 17
    FAR_FIELD_BEAM_FORMING = 18
    DEEP_NOISE_SUPPRESSION = 19

class MediaEffectClosedReason(enum.IntEnum):
    DONE = 0
    UNKNOWN_ERROR = 1
    UNSUPPORTED_ENCODING_FORMAT = 2
    EFFECT_CURRENTLY_UNLOADED = 3

class MediaMemoryTypes(enum.IntEnum):
    GPU = 0
    CPU = 1
    GPU_AND_CPU = 2

AudioCaptureEffectsManager = _ns_module.AudioCaptureEffectsManager
AudioEffect = _ns_module.AudioEffect
AudioEffectDefinition = _ns_module.AudioEffectDefinition
AudioEffectsManager = _ns_module.AudioEffectsManager
AudioRenderEffectsManager = _ns_module.AudioRenderEffectsManager
CompositeVideoFrameContext = _ns_module.CompositeVideoFrameContext
ProcessAudioFrameContext = _ns_module.ProcessAudioFrameContext
ProcessVideoFrameContext = _ns_module.ProcessVideoFrameContext
VideoCompositorDefinition = _ns_module.VideoCompositorDefinition
VideoEffectDefinition = _ns_module.VideoEffectDefinition
VideoTransformEffectDefinition = _ns_module.VideoTransformEffectDefinition
VideoTransformSphericalProjection = _ns_module.VideoTransformSphericalProjection
IAudioEffectDefinition = _ns_module.IAudioEffectDefinition
IBasicAudioEffect = _ns_module.IBasicAudioEffect
IBasicVideoEffect = _ns_module.IBasicVideoEffect
IVideoCompositor = _ns_module.IVideoCompositor
IVideoCompositorDefinition = _ns_module.IVideoCompositorDefinition
IVideoEffectDefinition = _ns_module.IVideoEffectDefinition