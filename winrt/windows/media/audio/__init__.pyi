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
    import winrt.windows.foundation.collections
except Exception:
    pass

try:
    import winrt.windows.foundation.numerics
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
    import winrt.windows.media.core
except Exception:
    pass

try:
    import winrt.windows.media.devices
except Exception:
    pass

try:
    import winrt.windows.media.effects
except Exception:
    pass

try:
    import winrt.windows.media.mediaproperties
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
    import winrt.windows.storage
except Exception:
    pass

class AudioDeviceNodeCreationStatus(enum.IntEnum):
    SUCCESS = 0
    DEVICE_NOT_AVAILABLE = 1
    FORMAT_NOT_SUPPORTED = 2
    UNKNOWN_FAILURE = 3
    ACCESS_DENIED = 4

class AudioFileNodeCreationStatus(enum.IntEnum):
    SUCCESS = 0
    FILE_NOT_FOUND = 1
    INVALID_FILE_TYPE = 2
    FORMAT_NOT_SUPPORTED = 3
    UNKNOWN_FAILURE = 4

class AudioGraphCreationStatus(enum.IntEnum):
    SUCCESS = 0
    DEVICE_NOT_AVAILABLE = 1
    FORMAT_NOT_SUPPORTED = 2
    UNKNOWN_FAILURE = 3

class AudioGraphUnrecoverableError(enum.IntEnum):
    NONE = 0
    AUDIO_DEVICE_LOST = 1
    AUDIO_SESSION_DISCONNECTED = 2
    UNKNOWN_FAILURE = 3

class AudioNodeEmitterDecayKind(enum.IntEnum):
    NATURAL = 0
    CUSTOM = 1

class AudioNodeEmitterSettings(enum.IntFlag):
    NONE = 0
    DISABLE_DOPPLER = 0x1

class AudioNodeEmitterShapeKind(enum.IntEnum):
    OMNIDIRECTIONAL = 0
    CONE = 1

class AudioPlaybackConnectionOpenResultStatus(enum.IntEnum):
    SUCCESS = 0
    REQUEST_TIMED_OUT = 1
    DENIED_BY_SYSTEM = 2
    UNKNOWN_FAILURE = 3

class AudioPlaybackConnectionState(enum.IntEnum):
    CLOSED = 0
    OPENED = 1

class MediaSourceAudioInputNodeCreationStatus(enum.IntEnum):
    SUCCESS = 0
    FORMAT_NOT_SUPPORTED = 1
    NETWORK_ERROR = 2
    UNKNOWN_FAILURE = 3

class MixedRealitySpatialAudioFormatPolicy(enum.IntEnum):
    USE_MIXED_REALITY_DEFAULT_SPATIAL_AUDIO_FORMAT = 0
    USE_DEVICE_CONFIGURATION_DEFAULT_SPATIAL_AUDIO_FORMAT = 1

class QuantumSizeSelectionMode(enum.IntEnum):
    SYSTEM_DEFAULT = 0
    LOWEST_LATENCY = 1
    CLOSEST_TO_DESIRED = 2

class SetDefaultSpatialAudioFormatStatus(enum.IntEnum):
    SUCCEEDED = 0
    ACCESS_DENIED = 1
    LICENSE_EXPIRED = 2
    LICENSE_NOT_VALID_FOR_AUDIO_ENDPOINT = 3
    NOT_SUPPORTED_ON_AUDIO_ENDPOINT = 4
    UNKNOWN_ERROR = 5

class SpatialAudioModel(enum.IntEnum):
    OBJECT_BASED = 0
    FOLD_DOWN = 1

class AudioDeviceInputNode(IAudioInputNode, IAudioNode, winrt.windows.foundation.IClosable, IAudioInputNode2, _winrt.winrt_base):
    ...
    device: winrt.windows.devices.enumeration.DeviceInformation
    outgoing_connections: winrt.windows.foundation.collections.IVectorView[AudioGraphConnection]
    emitter: AudioNodeEmitter
    outgoing_gain: float
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    def add_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def add_outgoing_connection(destination: IAudioNode, gain: float) -> None:
        ...
    def close() -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def remove_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...

class AudioDeviceOutputNode(IAudioNode, winrt.windows.foundation.IClosable, IAudioNodeWithListener, _winrt.winrt_base):
    ...
    device: winrt.windows.devices.enumeration.DeviceInformation
    outgoing_gain: float
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    listener: AudioNodeListener
    def close() -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...

class AudioFileInputNode(IAudioInputNode, IAudioNode, winrt.windows.foundation.IClosable, IAudioInputNode2, _winrt.winrt_base):
    ...
    playback_speed_factor: float
    loop_count: typing.Optional[int]
    end_time: typing.Optional[winrt.windows.foundation.TimeSpan]
    start_time: typing.Optional[winrt.windows.foundation.TimeSpan]
    position: winrt.windows.foundation.TimeSpan
    source_file: winrt.windows.storage.StorageFile
    duration: winrt.windows.foundation.TimeSpan
    outgoing_connections: winrt.windows.foundation.collections.IVectorView[AudioGraphConnection]
    emitter: AudioNodeEmitter
    outgoing_gain: float
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    def add_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def add_outgoing_connection(destination: IAudioNode, gain: float) -> None:
        ...
    def close() -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def remove_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def reset() -> None:
        ...
    def seek(position: winrt.windows.foundation.TimeSpan) -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...
    def add_file_completed(handler: winrt.windows.foundation.TypedEventHandler[AudioFileInputNode, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_file_completed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class AudioFileOutputNode(IAudioNode, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    file: winrt.windows.storage.IStorageFile
    file_encoding_profile: winrt.windows.media.mediaproperties.MediaEncodingProfile
    outgoing_gain: float
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    def close() -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def finalize_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.media.transcoding.TranscodeFailureReason]:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...

class AudioFrameCompletedEventArgs(_winrt.winrt_base):
    ...
    frame: winrt.windows.media.AudioFrame

class AudioFrameInputNode(IAudioInputNode, IAudioNode, winrt.windows.foundation.IClosable, IAudioInputNode2, _winrt.winrt_base):
    ...
    playback_speed_factor: float
    queued_sample_count: int
    outgoing_connections: winrt.windows.foundation.collections.IVectorView[AudioGraphConnection]
    emitter: AudioNodeEmitter
    outgoing_gain: float
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    def add_frame(frame: winrt.windows.media.AudioFrame) -> None:
        ...
    def add_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def add_outgoing_connection(destination: IAudioNode, gain: float) -> None:
        ...
    def close() -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def discard_queued_frames() -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def remove_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...
    def add_audio_frame_completed(handler: winrt.windows.foundation.TypedEventHandler[AudioFrameInputNode, AudioFrameCompletedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_audio_frame_completed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_quantum_started(handler: winrt.windows.foundation.TypedEventHandler[AudioFrameInputNode, FrameInputNodeQuantumStartedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_quantum_started(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class AudioFrameOutputNode(IAudioNode, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    outgoing_gain: float
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    def close() -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def get_frame() -> winrt.windows.media.AudioFrame:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...

class AudioGraph(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    completed_quantum_count: int
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    latency_in_samples: int
    primary_render_device: winrt.windows.devices.enumeration.DeviceInformation
    render_device_audio_processing: winrt.windows.media.AudioProcessing
    samples_per_quantum: int
    def close() -> None:
        ...
    def create_async(settings: AudioGraphSettings) -> winrt.windows.foundation.IAsyncOperation[CreateAudioGraphResult]:
        ...
    def create_batch_updater() -> AudioGraphBatchUpdater:
        ...
    def create_device_input_node_async(category: winrt.windows.media.capture.MediaCategory) -> winrt.windows.foundation.IAsyncOperation[CreateAudioDeviceInputNodeResult]:
        ...
    def create_device_input_node_async(category: winrt.windows.media.capture.MediaCategory, encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties) -> winrt.windows.foundation.IAsyncOperation[CreateAudioDeviceInputNodeResult]:
        ...
    def create_device_input_node_async(category: winrt.windows.media.capture.MediaCategory, encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties, device: winrt.windows.devices.enumeration.DeviceInformation) -> winrt.windows.foundation.IAsyncOperation[CreateAudioDeviceInputNodeResult]:
        ...
    def create_device_input_node_async(category: winrt.windows.media.capture.MediaCategory, encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties, device: winrt.windows.devices.enumeration.DeviceInformation, emitter: AudioNodeEmitter) -> winrt.windows.foundation.IAsyncOperation[CreateAudioDeviceInputNodeResult]:
        ...
    def create_device_output_node_async() -> winrt.windows.foundation.IAsyncOperation[CreateAudioDeviceOutputNodeResult]:
        ...
    def create_file_input_node_async(file: winrt.windows.storage.IStorageFile) -> winrt.windows.foundation.IAsyncOperation[CreateAudioFileInputNodeResult]:
        ...
    def create_file_input_node_async(file: winrt.windows.storage.IStorageFile, emitter: AudioNodeEmitter) -> winrt.windows.foundation.IAsyncOperation[CreateAudioFileInputNodeResult]:
        ...
    def create_file_output_node_async(file: winrt.windows.storage.IStorageFile) -> winrt.windows.foundation.IAsyncOperation[CreateAudioFileOutputNodeResult]:
        ...
    def create_file_output_node_async(file: winrt.windows.storage.IStorageFile, file_encoding_profile: winrt.windows.media.mediaproperties.MediaEncodingProfile) -> winrt.windows.foundation.IAsyncOperation[CreateAudioFileOutputNodeResult]:
        ...
    def create_frame_input_node() -> AudioFrameInputNode:
        ...
    def create_frame_input_node(encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties) -> AudioFrameInputNode:
        ...
    def create_frame_input_node(encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties, emitter: AudioNodeEmitter) -> AudioFrameInputNode:
        ...
    def create_frame_output_node() -> AudioFrameOutputNode:
        ...
    def create_frame_output_node(encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties) -> AudioFrameOutputNode:
        ...
    def create_media_source_audio_input_node_async(media_source: winrt.windows.media.core.MediaSource) -> winrt.windows.foundation.IAsyncOperation[CreateMediaSourceAudioInputNodeResult]:
        ...
    def create_media_source_audio_input_node_async(media_source: winrt.windows.media.core.MediaSource, emitter: AudioNodeEmitter) -> winrt.windows.foundation.IAsyncOperation[CreateMediaSourceAudioInputNodeResult]:
        ...
    def create_submix_node() -> AudioSubmixNode:
        ...
    def create_submix_node(encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties) -> AudioSubmixNode:
        ...
    def create_submix_node(encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties, emitter: AudioNodeEmitter) -> AudioSubmixNode:
        ...
    def reset_all_nodes() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...
    def add_quantum_processed(handler: winrt.windows.foundation.TypedEventHandler[AudioGraph, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_quantum_processed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_quantum_started(handler: winrt.windows.foundation.TypedEventHandler[AudioGraph, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_quantum_started(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_unrecoverable_error_occurred(handler: winrt.windows.foundation.TypedEventHandler[AudioGraph, AudioGraphUnrecoverableErrorOccurredEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_unrecoverable_error_occurred(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class AudioGraphBatchUpdater(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    def close() -> None:
        ...

class AudioGraphConnection(_winrt.winrt_base):
    ...
    gain: float
    destination: IAudioNode

class AudioGraphSettings(_winrt.winrt_base):
    ...
    quantum_size_selection_mode: QuantumSizeSelectionMode
    primary_render_device: winrt.windows.devices.enumeration.DeviceInformation
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    desired_samples_per_quantum: int
    desired_render_device_audio_processing: winrt.windows.media.AudioProcessing
    audio_render_category: winrt.windows.media.render.AudioRenderCategory
    max_playback_speed_factor: float

class AudioGraphUnrecoverableErrorOccurredEventArgs(_winrt.winrt_base):
    ...
    error: AudioGraphUnrecoverableError

class AudioNodeEmitter(_winrt.winrt_base):
    ...
    position: winrt.windows.foundation.numerics.Vector3
    gain: float
    doppler_velocity: winrt.windows.foundation.numerics.Vector3
    doppler_scale: float
    distance_scale: float
    direction: winrt.windows.foundation.numerics.Vector3
    decay_model: AudioNodeEmitterDecayModel
    is_doppler_disabled: bool
    shape: AudioNodeEmitterShape
    spatial_audio_model: SpatialAudioModel

class AudioNodeEmitterConeProperties(_winrt.winrt_base):
    ...
    inner_angle: float
    outer_angle: float
    outer_angle_gain: float

class AudioNodeEmitterDecayModel(_winrt.winrt_base):
    ...
    kind: AudioNodeEmitterDecayKind
    max_gain: float
    min_gain: float
    natural_properties: AudioNodeEmitterNaturalDecayModelProperties
    def create_custom(min_gain: float, max_gain: float) -> AudioNodeEmitterDecayModel:
        ...
    def create_natural(min_gain: float, max_gain: float, unity_gain_distance: float, cutoff_distance: float) -> AudioNodeEmitterDecayModel:
        ...

class AudioNodeEmitterNaturalDecayModelProperties(_winrt.winrt_base):
    ...
    cutoff_distance: float
    unity_gain_distance: float

class AudioNodeEmitterShape(_winrt.winrt_base):
    ...
    cone_properties: AudioNodeEmitterConeProperties
    kind: AudioNodeEmitterShapeKind
    def create_cone(inner_angle: float, outer_angle: float, outer_angle_gain: float) -> AudioNodeEmitterShape:
        ...
    def create_omnidirectional() -> AudioNodeEmitterShape:
        ...

class AudioNodeListener(_winrt.winrt_base):
    ...
    speed_of_sound: float
    position: winrt.windows.foundation.numerics.Vector3
    orientation: winrt.windows.foundation.numerics.Quaternion
    doppler_velocity: winrt.windows.foundation.numerics.Vector3

class AudioPlaybackConnection(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    device_id: str
    state: AudioPlaybackConnectionState
    def close() -> None:
        ...
    def get_device_selector() -> str:
        ...
    def open() -> AudioPlaybackConnectionOpenResult:
        ...
    def open_async() -> winrt.windows.foundation.IAsyncOperation[AudioPlaybackConnectionOpenResult]:
        ...
    def start() -> None:
        ...
    def start_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def try_create_from_id(id: str) -> AudioPlaybackConnection:
        ...
    def add_state_changed(handler: winrt.windows.foundation.TypedEventHandler[AudioPlaybackConnection, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_state_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class AudioPlaybackConnectionOpenResult(_winrt.winrt_base):
    ...
    extended_error: winrt.windows.foundation.HResult
    status: AudioPlaybackConnectionOpenResultStatus

class AudioStateMonitor(_winrt.winrt_base):
    ...
    sound_level: winrt.windows.media.SoundLevel
    def create_for_capture_monitoring() -> AudioStateMonitor:
        ...
    def create_for_capture_monitoring(category: winrt.windows.media.capture.MediaCategory) -> AudioStateMonitor:
        ...
    def create_for_capture_monitoring(category: winrt.windows.media.capture.MediaCategory, role: winrt.windows.media.devices.AudioDeviceRole) -> AudioStateMonitor:
        ...
    def create_for_capture_monitoring_with_category_and_device_id(category: winrt.windows.media.capture.MediaCategory, device_id: str) -> AudioStateMonitor:
        ...
    def create_for_render_monitoring() -> AudioStateMonitor:
        ...
    def create_for_render_monitoring(category: winrt.windows.media.render.AudioRenderCategory) -> AudioStateMonitor:
        ...
    def create_for_render_monitoring(category: winrt.windows.media.render.AudioRenderCategory, role: winrt.windows.media.devices.AudioDeviceRole) -> AudioStateMonitor:
        ...
    def create_for_render_monitoring_with_category_and_device_id(category: winrt.windows.media.render.AudioRenderCategory, device_id: str) -> AudioStateMonitor:
        ...
    def add_sound_level_changed(handler: winrt.windows.foundation.TypedEventHandler[AudioStateMonitor, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_sound_level_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class AudioSubmixNode(IAudioInputNode, IAudioNode, winrt.windows.foundation.IClosable, IAudioInputNode2, _winrt.winrt_base):
    ...
    outgoing_connections: winrt.windows.foundation.collections.IVectorView[AudioGraphConnection]
    emitter: AudioNodeEmitter
    outgoing_gain: float
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    def add_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def add_outgoing_connection(destination: IAudioNode, gain: float) -> None:
        ...
    def close() -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def remove_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...

class CreateAudioDeviceInputNodeResult(_winrt.winrt_base):
    ...
    device_input_node: AudioDeviceInputNode
    status: AudioDeviceNodeCreationStatus
    extended_error: winrt.windows.foundation.HResult

class CreateAudioDeviceOutputNodeResult(_winrt.winrt_base):
    ...
    device_output_node: AudioDeviceOutputNode
    status: AudioDeviceNodeCreationStatus
    extended_error: winrt.windows.foundation.HResult

class CreateAudioFileInputNodeResult(_winrt.winrt_base):
    ...
    file_input_node: AudioFileInputNode
    status: AudioFileNodeCreationStatus
    extended_error: winrt.windows.foundation.HResult

class CreateAudioFileOutputNodeResult(_winrt.winrt_base):
    ...
    file_output_node: AudioFileOutputNode
    status: AudioFileNodeCreationStatus
    extended_error: winrt.windows.foundation.HResult

class CreateAudioGraphResult(_winrt.winrt_base):
    ...
    graph: AudioGraph
    status: AudioGraphCreationStatus
    extended_error: winrt.windows.foundation.HResult

class CreateMediaSourceAudioInputNodeResult(_winrt.winrt_base):
    ...
    node: MediaSourceAudioInputNode
    status: MediaSourceAudioInputNodeCreationStatus
    extended_error: winrt.windows.foundation.HResult

class EchoEffectDefinition(winrt.windows.media.effects.IAudioEffectDefinition, _winrt.winrt_base):
    ...
    wet_dry_mix: float
    feedback: float
    delay: float
    activatable_class_id: str
    properties: winrt.windows.foundation.collections.IPropertySet

class EqualizerBand(_winrt.winrt_base):
    ...
    gain: float
    frequency_center: float
    bandwidth: float

class EqualizerEffectDefinition(winrt.windows.media.effects.IAudioEffectDefinition, _winrt.winrt_base):
    ...
    bands: winrt.windows.foundation.collections.IVectorView[EqualizerBand]
    activatable_class_id: str
    properties: winrt.windows.foundation.collections.IPropertySet

class FrameInputNodeQuantumStartedEventArgs(_winrt.winrt_base):
    ...
    required_samples: int

class LimiterEffectDefinition(winrt.windows.media.effects.IAudioEffectDefinition, _winrt.winrt_base):
    ...
    release: int
    loudness: int
    activatable_class_id: str
    properties: winrt.windows.foundation.collections.IPropertySet

class MediaSourceAudioInputNode(IAudioInputNode2, IAudioInputNode, IAudioNode, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    outgoing_connections: winrt.windows.foundation.collections.IVectorView[AudioGraphConnection]
    emitter: AudioNodeEmitter
    outgoing_gain: float
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    start_time: typing.Optional[winrt.windows.foundation.TimeSpan]
    playback_speed_factor: float
    loop_count: typing.Optional[int]
    end_time: typing.Optional[winrt.windows.foundation.TimeSpan]
    duration: winrt.windows.foundation.TimeSpan
    media_source: winrt.windows.media.core.MediaSource
    position: winrt.windows.foundation.TimeSpan
    def add_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def add_outgoing_connection(destination: IAudioNode, gain: float) -> None:
        ...
    def close() -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def remove_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def reset() -> None:
        ...
    def seek(position: winrt.windows.foundation.TimeSpan) -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...
    def add_media_source_completed(handler: winrt.windows.foundation.TypedEventHandler[MediaSourceAudioInputNode, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_media_source_completed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class ReverbEffectDefinition(winrt.windows.media.effects.IAudioEffectDefinition, _winrt.winrt_base):
    ...
    high_e_q_gain: int
    high_e_q_cutoff: int
    disable_late_field: bool
    density: float
    position_right: int
    decay_time: float
    late_diffusion: int
    position_matrix_right: int
    position_matrix_left: int
    position_left: int
    low_e_q_gain: int
    low_e_q_cutoff: int
    room_filter_freq: float
    reverb_gain: float
    reverb_delay: int
    reflections_gain: float
    reflections_delay: int
    rear_delay: int
    wet_dry_mix: float
    early_diffusion: int
    room_size: float
    room_filter_main: float
    room_filter_h_f: float
    activatable_class_id: str
    properties: winrt.windows.foundation.collections.IPropertySet

class SetDefaultSpatialAudioFormatResult(_winrt.winrt_base):
    ...
    status: SetDefaultSpatialAudioFormatStatus

class SpatialAudioDeviceConfiguration(_winrt.winrt_base):
    ...
    active_spatial_audio_format: str
    default_spatial_audio_format: str
    device_id: str
    is_spatial_audio_supported: bool
    def get_for_device_id(device_id: str) -> SpatialAudioDeviceConfiguration:
        ...
    def is_spatial_audio_format_supported(subtype: str) -> bool:
        ...
    def set_default_spatial_audio_format_async(subtype: str) -> winrt.windows.foundation.IAsyncOperation[SetDefaultSpatialAudioFormatResult]:
        ...
    def add_configuration_changed(handler: winrt.windows.foundation.TypedEventHandler[SpatialAudioDeviceConfiguration, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_configuration_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class SpatialAudioFormatConfiguration(_winrt.winrt_base):
    ...
    mixed_reality_exclusive_mode_policy: MixedRealitySpatialAudioFormatPolicy
    def get_default() -> SpatialAudioFormatConfiguration:
        ...
    def report_configuration_changed_async(subtype: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def report_license_changed_async(subtype: str) -> winrt.windows.foundation.IAsyncAction:
        ...

class SpatialAudioFormatSubtype(_winrt.winrt_base):
    ...
    d_t_s_headphone_x: str
    d_t_s_x_ultra: str
    dolby_atmos_for_headphones: str
    dolby_atmos_for_home_theater: str
    dolby_atmos_for_speakers: str
    windows_sonic: str
    d_t_s_x_for_home_theater: str

class IAudioInputNode(IAudioNode, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    outgoing_connections: winrt.windows.foundation.collections.IVectorView[AudioGraphConnection]
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    outgoing_gain: float
    def add_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def add_outgoing_connection(destination: IAudioNode, gain: float) -> None:
        ...
    def remove_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...
    def close() -> None:
        ...

class IAudioInputNode2(IAudioNode, winrt.windows.foundation.IClosable, IAudioInputNode, _winrt.winrt_base):
    ...
    emitter: AudioNodeEmitter
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    outgoing_gain: float
    outgoing_connections: winrt.windows.foundation.collections.IVectorView[AudioGraphConnection]
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...
    def close() -> None:
        ...
    def add_outgoing_connection(destination: IAudioNode) -> None:
        ...
    def add_outgoing_connection(destination: IAudioNode, gain: float) -> None:
        ...
    def remove_outgoing_connection(destination: IAudioNode) -> None:
        ...

class IAudioNode(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    outgoing_gain: float
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...
    def close() -> None:
        ...

class IAudioNodeWithListener(winrt.windows.foundation.IClosable, IAudioNode, _winrt.winrt_base):
    ...
    listener: AudioNodeListener
    consume_input: bool
    effect_definitions: winrt.windows.foundation.collections.IVector[winrt.windows.media.effects.IAudioEffectDefinition]
    encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    outgoing_gain: float
    def close() -> None:
        ...
    def disable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def enable_effects_by_definition(definition: winrt.windows.media.effects.IAudioEffectDefinition) -> None:
        ...
    def reset() -> None:
        ...
    def start() -> None:
        ...
    def stop() -> None:
        ...
