# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Media.Core")

try:
    import winrt.windows.applicationmodel.appservice
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
    import winrt.windows.graphics.directx.direct3d11
except Exception:
    pass

try:
    import winrt.windows.graphics.imaging
except Exception:
    pass

try:
    import winrt.windows.media.capture
except Exception:
    pass

try:
    import winrt.windows.media.capture.frames
except Exception:
    pass

try:
    import winrt.windows.media.devices
except Exception:
    pass

try:
    import winrt.windows.media.devices.core
except Exception:
    pass

try:
    import winrt.windows.media.faceanalysis
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
    import winrt.windows.media.protection
except Exception:
    pass

try:
    import winrt.windows.media.streaming.adaptive
except Exception:
    pass

try:
    import winrt.windows.networking.backgroundtransfer
except Exception:
    pass

try:
    import winrt.windows.storage
except Exception:
    pass

try:
    import winrt.windows.storage.fileproperties
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

class AudioDecoderDegradation(enum.IntEnum):
    NONE = 0
    DOWNMIX_TO2_CHANNELS = 1
    DOWNMIX_TO6_CHANNELS = 2
    DOWNMIX_TO8_CHANNELS = 3

class AudioDecoderDegradationReason(enum.IntEnum):
    NONE = 0
    LICENSING_REQUIREMENT = 1
    SPATIAL_AUDIO_NOT_SUPPORTED = 2

class CodecCategory(enum.IntEnum):
    ENCODER = 0
    DECODER = 1

class CodecKind(enum.IntEnum):
    AUDIO = 0
    VIDEO = 1

class FaceDetectionMode(enum.IntEnum):
    HIGH_PERFORMANCE = 0
    BALANCED = 1
    HIGH_QUALITY = 2

class MediaDecoderStatus(enum.IntEnum):
    FULLY_SUPPORTED = 0
    UNSUPPORTED_SUBTYPE = 1
    UNSUPPORTED_ENCODER_PROPERTIES = 2
    DEGRADED = 3

class MediaSourceState(enum.IntEnum):
    INITIAL = 0
    OPENING = 1
    OPENED = 2
    FAILED = 3
    CLOSED = 4

class MediaSourceStatus(enum.IntEnum):
    FULLY_SUPPORTED = 0
    UNKNOWN = 1

class MediaStreamSourceClosedReason(enum.IntEnum):
    DONE = 0
    UNKNOWN_ERROR = 1
    APP_REPORTED_ERROR = 2
    UNSUPPORTED_PROTECTION_SYSTEM = 3
    PROTECTION_SYSTEM_FAILURE = 4
    UNSUPPORTED_ENCODING_FORMAT = 5
    MISSING_SAMPLE_REQUESTED_EVENT_HANDLER = 6

class MediaStreamSourceErrorStatus(enum.IntEnum):
    OTHER = 0
    OUT_OF_MEMORY = 1
    FAILED_TO_OPEN_FILE = 2
    FAILED_TO_CONNECT_TO_SERVER = 3
    CONNECTION_TO_SERVER_LOST = 4
    UNSPECIFIED_NETWORK_ERROR = 5
    DECODE_ERROR = 6
    UNSUPPORTED_MEDIA_FORMAT = 7

class MediaTrackKind(enum.IntEnum):
    AUDIO = 0
    VIDEO = 1
    TIMED_METADATA = 2

class MseAppendMode(enum.IntEnum):
    SEGMENTS = 0
    SEQUENCE = 1

class MseEndOfStreamStatus(enum.IntEnum):
    SUCCESS = 0
    NETWORK_ERROR = 1
    DECODE_ERROR = 2
    UNKNOWN_ERROR = 3

class MseReadyState(enum.IntEnum):
    CLOSED = 0
    OPEN = 1
    ENDED = 2

class SceneAnalysisRecommendation(enum.IntEnum):
    STANDARD = 0
    HDR = 1
    LOW_LIGHT = 2

class TimedMetadataKind(enum.IntEnum):
    CAPTION = 0
    CHAPTER = 1
    CUSTOM = 2
    DATA = 3
    DESCRIPTION = 4
    SUBTITLE = 5
    IMAGE_SUBTITLE = 6
    SPEECH = 7

class TimedMetadataTrackErrorCode(enum.IntEnum):
    NONE = 0
    DATA_FORMAT_ERROR = 1
    NETWORK_ERROR = 2
    INTERNAL_ERROR = 3

class TimedTextBoutenPosition(enum.IntEnum):
    BEFORE = 0
    AFTER = 1
    OUTSIDE = 2

class TimedTextBoutenType(enum.IntEnum):
    NONE = 0
    AUTO = 1
    FILLED_CIRCLE = 2
    OPEN_CIRCLE = 3
    FILLED_DOT = 4
    OPEN_DOT = 5
    FILLED_SESAME = 6
    OPEN_SESAME = 7

class TimedTextDisplayAlignment(enum.IntEnum):
    BEFORE = 0
    AFTER = 1
    CENTER = 2

class TimedTextFlowDirection(enum.IntEnum):
    LEFT_TO_RIGHT = 0
    RIGHT_TO_LEFT = 1

class TimedTextFontStyle(enum.IntEnum):
    NORMAL = 0
    OBLIQUE = 1
    ITALIC = 2

class TimedTextLineAlignment(enum.IntEnum):
    START = 0
    END = 1
    CENTER = 2

class TimedTextRubyAlign(enum.IntEnum):
    CENTER = 0
    START = 1
    END = 2
    SPACE_AROUND = 3
    SPACE_BETWEEN = 4
    WITH_BASE = 5

class TimedTextRubyPosition(enum.IntEnum):
    BEFORE = 0
    AFTER = 1
    OUTSIDE = 2

class TimedTextRubyReserve(enum.IntEnum):
    NONE = 0
    BEFORE = 1
    AFTER = 2
    BOTH = 3
    OUTSIDE = 4

class TimedTextScrollMode(enum.IntEnum):
    POPON = 0
    ROLLUP = 1

class TimedTextUnit(enum.IntEnum):
    PIXELS = 0
    PERCENTAGE = 1

class TimedTextWeight(enum.IntEnum):
    NORMAL = 400
    BOLD = 700

class TimedTextWrapping(enum.IntEnum):
    NO_WRAP = 0
    WRAP = 1

class TimedTextWritingMode(enum.IntEnum):
    LEFT_RIGHT_TOP_BOTTOM = 0
    RIGHT_LEFT_TOP_BOTTOM = 1
    TOP_BOTTOM_RIGHT_LEFT = 2
    TOP_BOTTOM_LEFT_RIGHT = 3
    LEFT_RIGHT = 4
    RIGHT_LEFT = 5
    TOP_BOTTOM = 6

class VideoStabilizationEffectEnabledChangedReason(enum.IntEnum):
    PROGRAMMATIC = 0
    PIXEL_RATE_TOO_HIGH = 1
    RUNNING_SLOWLY = 2

MseTimeRange = _ns_module.MseTimeRange
TimedTextDouble = _ns_module.TimedTextDouble
TimedTextPadding = _ns_module.TimedTextPadding
TimedTextPoint = _ns_module.TimedTextPoint
TimedTextSize = _ns_module.TimedTextSize
AudioStreamDescriptor = _ns_module.AudioStreamDescriptor
AudioTrack = _ns_module.AudioTrack
AudioTrackOpenFailedEventArgs = _ns_module.AudioTrackOpenFailedEventArgs
AudioTrackSupportInfo = _ns_module.AudioTrackSupportInfo
ChapterCue = _ns_module.ChapterCue
CodecInfo = _ns_module.CodecInfo
CodecQuery = _ns_module.CodecQuery
CodecSubtypes = _ns_module.CodecSubtypes
DataCue = _ns_module.DataCue
FaceDetectedEventArgs = _ns_module.FaceDetectedEventArgs
FaceDetectionEffect = _ns_module.FaceDetectionEffect
FaceDetectionEffectDefinition = _ns_module.FaceDetectionEffectDefinition
FaceDetectionEffectFrame = _ns_module.FaceDetectionEffectFrame
HighDynamicRangeControl = _ns_module.HighDynamicRangeControl
HighDynamicRangeOutput = _ns_module.HighDynamicRangeOutput
ImageCue = _ns_module.ImageCue
InitializeMediaStreamSourceRequestedEventArgs = _ns_module.InitializeMediaStreamSourceRequestedEventArgs
LowLightFusion = _ns_module.LowLightFusion
LowLightFusionResult = _ns_module.LowLightFusionResult
MediaBinder = _ns_module.MediaBinder
MediaBindingEventArgs = _ns_module.MediaBindingEventArgs
MediaCueEventArgs = _ns_module.MediaCueEventArgs
MediaSource = _ns_module.MediaSource
MediaSourceAppServiceConnection = _ns_module.MediaSourceAppServiceConnection
MediaSourceError = _ns_module.MediaSourceError
MediaSourceOpenOperationCompletedEventArgs = _ns_module.MediaSourceOpenOperationCompletedEventArgs
MediaSourceStateChangedEventArgs = _ns_module.MediaSourceStateChangedEventArgs
MediaStreamSample = _ns_module.MediaStreamSample
MediaStreamSamplePropertySet = _ns_module.MediaStreamSamplePropertySet
MediaStreamSampleProtectionProperties = _ns_module.MediaStreamSampleProtectionProperties
MediaStreamSource = _ns_module.MediaStreamSource
MediaStreamSourceClosedEventArgs = _ns_module.MediaStreamSourceClosedEventArgs
MediaStreamSourceClosedRequest = _ns_module.MediaStreamSourceClosedRequest
MediaStreamSourceSampleRenderedEventArgs = _ns_module.MediaStreamSourceSampleRenderedEventArgs
MediaStreamSourceSampleRequest = _ns_module.MediaStreamSourceSampleRequest
MediaStreamSourceSampleRequestDeferral = _ns_module.MediaStreamSourceSampleRequestDeferral
MediaStreamSourceSampleRequestedEventArgs = _ns_module.MediaStreamSourceSampleRequestedEventArgs
MediaStreamSourceStartingEventArgs = _ns_module.MediaStreamSourceStartingEventArgs
MediaStreamSourceStartingRequest = _ns_module.MediaStreamSourceStartingRequest
MediaStreamSourceStartingRequestDeferral = _ns_module.MediaStreamSourceStartingRequestDeferral
MediaStreamSourceSwitchStreamsRequest = _ns_module.MediaStreamSourceSwitchStreamsRequest
MediaStreamSourceSwitchStreamsRequestDeferral = _ns_module.MediaStreamSourceSwitchStreamsRequestDeferral
MediaStreamSourceSwitchStreamsRequestedEventArgs = _ns_module.MediaStreamSourceSwitchStreamsRequestedEventArgs
MseSourceBuffer = _ns_module.MseSourceBuffer
MseSourceBufferList = _ns_module.MseSourceBufferList
MseStreamSource = _ns_module.MseStreamSource
SceneAnalysisEffect = _ns_module.SceneAnalysisEffect
SceneAnalysisEffectDefinition = _ns_module.SceneAnalysisEffectDefinition
SceneAnalysisEffectFrame = _ns_module.SceneAnalysisEffectFrame
SceneAnalyzedEventArgs = _ns_module.SceneAnalyzedEventArgs
SpeechCue = _ns_module.SpeechCue
TimedMetadataStreamDescriptor = _ns_module.TimedMetadataStreamDescriptor
TimedMetadataTrack = _ns_module.TimedMetadataTrack
TimedMetadataTrackError = _ns_module.TimedMetadataTrackError
TimedMetadataTrackFailedEventArgs = _ns_module.TimedMetadataTrackFailedEventArgs
TimedTextBouten = _ns_module.TimedTextBouten
TimedTextCue = _ns_module.TimedTextCue
TimedTextLine = _ns_module.TimedTextLine
TimedTextRegion = _ns_module.TimedTextRegion
TimedTextRuby = _ns_module.TimedTextRuby
TimedTextSource = _ns_module.TimedTextSource
TimedTextSourceResolveResultEventArgs = _ns_module.TimedTextSourceResolveResultEventArgs
TimedTextStyle = _ns_module.TimedTextStyle
TimedTextSubformat = _ns_module.TimedTextSubformat
VideoStabilizationEffect = _ns_module.VideoStabilizationEffect
VideoStabilizationEffectDefinition = _ns_module.VideoStabilizationEffectDefinition
VideoStabilizationEffectEnabledChangedEventArgs = _ns_module.VideoStabilizationEffectEnabledChangedEventArgs
VideoStreamDescriptor = _ns_module.VideoStreamDescriptor
VideoTrack = _ns_module.VideoTrack
VideoTrackOpenFailedEventArgs = _ns_module.VideoTrackOpenFailedEventArgs
VideoTrackSupportInfo = _ns_module.VideoTrackSupportInfo
IMediaCue = _ns_module.IMediaCue
IMediaSource = _ns_module.IMediaSource
IMediaStreamDescriptor = _ns_module.IMediaStreamDescriptor
IMediaStreamDescriptor2 = _ns_module.IMediaStreamDescriptor2
IMediaTrack = _ns_module.IMediaTrack
ISingleSelectMediaTrackList = _ns_module.ISingleSelectMediaTrackList
ITimedMetadataTrackProvider = _ns_module.ITimedMetadataTrackProvider
