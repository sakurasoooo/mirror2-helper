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
    import winrt.windows.graphics.directx.direct3d11
except Exception:
    pass

try:
    import winrt.windows.graphics.imaging
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
    import winrt.windows.media.devices
except Exception:
    pass

try:
    import winrt.windows.media.devices.core
except Exception:
    pass

try:
    import winrt.windows.media.mediaproperties
except Exception:
    pass

try:
    import winrt.windows.perception.spatial
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

try:
    import winrt.windows.ui.windowmanagement
except Exception:
    pass

class MediaFrameReaderAcquisitionMode(enum.IntEnum):
    REALTIME = 0
    BUFFERED = 1

class MediaFrameReaderStartStatus(enum.IntEnum):
    SUCCESS = 0
    UNKNOWN_FAILURE = 1
    DEVICE_NOT_AVAILABLE = 2
    OUTPUT_FORMAT_NOT_SUPPORTED = 3
    EXCLUSIVE_CONTROL_NOT_AVAILABLE = 4

class MediaFrameSourceGetPropertyStatus(enum.IntEnum):
    SUCCESS = 0
    UNKNOWN_FAILURE = 1
    NOT_SUPPORTED = 2
    DEVICE_NOT_AVAILABLE = 3
    MAX_PROPERTY_VALUE_SIZE_TOO_SMALL = 4
    MAX_PROPERTY_VALUE_SIZE_REQUIRED = 5

class MediaFrameSourceKind(enum.IntEnum):
    CUSTOM = 0
    COLOR = 1
    INFRARED = 2
    DEPTH = 3
    AUDIO = 4
    IMAGE = 5
    METADATA = 6

class MediaFrameSourceSetPropertyStatus(enum.IntEnum):
    SUCCESS = 0
    UNKNOWN_FAILURE = 1
    NOT_SUPPORTED = 2
    INVALID_VALUE = 3
    DEVICE_NOT_AVAILABLE = 4
    NOT_IN_CONTROL = 5

class MultiSourceMediaFrameReaderStartStatus(enum.IntEnum):
    SUCCESS = 0
    NOT_SUPPORTED = 1
    INSUFFICIENT_RESOURCES = 2
    DEVICE_NOT_AVAILABLE = 3
    UNKNOWN_FAILURE = 4

class AudioMediaFrame(_winrt.winrt_base):
    ...
    audio_encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties
    frame_reference: MediaFrameReference
    def get_audio_frame() -> winrt.windows.media.AudioFrame:
        ...

class BufferMediaFrame(_winrt.winrt_base):
    ...
    buffer: winrt.windows.storage.streams.IBuffer
    frame_reference: MediaFrameReference

class DepthMediaFrame(_winrt.winrt_base):
    ...
    depth_format: DepthMediaFrameFormat
    frame_reference: MediaFrameReference
    video_media_frame: VideoMediaFrame
    max_reliable_depth: int
    min_reliable_depth: int
    def try_create_coordinate_mapper(camera_intrinsics: winrt.windows.media.devices.core.CameraIntrinsics, coordinate_system: winrt.windows.perception.spatial.SpatialCoordinateSystem) -> winrt.windows.media.devices.core.DepthCorrelatedCoordinateMapper:
        ...

class DepthMediaFrameFormat(_winrt.winrt_base):
    ...
    depth_scale_in_meters: float
    video_format: VideoMediaFrameFormat

class InfraredMediaFrame(_winrt.winrt_base):
    ...
    frame_reference: MediaFrameReference
    is_illuminated: bool
    video_media_frame: VideoMediaFrame

class MediaFrameArrivedEventArgs(_winrt.winrt_base):
    ...

class MediaFrameFormat(_winrt.winrt_base):
    ...
    frame_rate: winrt.windows.media.mediaproperties.MediaRatio
    major_type: str
    properties: winrt.windows.foundation.collections.IMapView[uuid.UUID, _winrt.winrt_base]
    subtype: str
    video_format: VideoMediaFrameFormat
    audio_encoding_properties: winrt.windows.media.mediaproperties.AudioEncodingProperties

class MediaFrameReader(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    acquisition_mode: MediaFrameReaderAcquisitionMode
    def close() -> None:
        ...
    def start_async() -> winrt.windows.foundation.IAsyncOperation[MediaFrameReaderStartStatus]:
        ...
    def stop_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def try_acquire_latest_frame() -> MediaFrameReference:
        ...
    def add_frame_arrived(handler: winrt.windows.foundation.TypedEventHandler[MediaFrameReader, MediaFrameArrivedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_frame_arrived(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class MediaFrameReference(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    buffer_media_frame: BufferMediaFrame
    coordinate_system: winrt.windows.perception.spatial.SpatialCoordinateSystem
    duration: winrt.windows.foundation.TimeSpan
    format: MediaFrameFormat
    properties: winrt.windows.foundation.collections.IMapView[uuid.UUID, _winrt.winrt_base]
    source_kind: MediaFrameSourceKind
    system_relative_time: typing.Optional[winrt.windows.foundation.TimeSpan]
    video_media_frame: VideoMediaFrame
    audio_media_frame: AudioMediaFrame
    def close() -> None:
        ...

class MediaFrameSource(_winrt.winrt_base):
    ...
    controller: MediaFrameSourceController
    current_format: MediaFrameFormat
    info: MediaFrameSourceInfo
    supported_formats: winrt.windows.foundation.collections.IVectorView[MediaFrameFormat]
    def set_format_async(format: MediaFrameFormat) -> winrt.windows.foundation.IAsyncAction:
        ...
    def try_get_camera_intrinsics(format: MediaFrameFormat) -> winrt.windows.media.devices.core.CameraIntrinsics:
        ...
    def add_format_changed(handler: winrt.windows.foundation.TypedEventHandler[MediaFrameSource, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_format_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class MediaFrameSourceController(_winrt.winrt_base):
    ...
    video_device_controller: winrt.windows.media.devices.VideoDeviceController
    audio_device_controller: winrt.windows.media.devices.AudioDeviceController
    def get_property_async(property_id: str) -> winrt.windows.foundation.IAsyncOperation[MediaFrameSourceGetPropertyResult]:
        ...
    def get_property_by_extended_id_async(extended_property_id: typing.Sequence[int], max_property_value_size: typing.Optional[int]) -> winrt.windows.foundation.IAsyncOperation[MediaFrameSourceGetPropertyResult]:
        ...
    def set_property_async(property_id: str, property_value: _winrt.winrt_base) -> winrt.windows.foundation.IAsyncOperation[MediaFrameSourceSetPropertyStatus]:
        ...
    def set_property_by_extended_id_async(extended_property_id: typing.Sequence[int], property_value: typing.Sequence[int]) -> winrt.windows.foundation.IAsyncOperation[MediaFrameSourceSetPropertyStatus]:
        ...

class MediaFrameSourceGetPropertyResult(_winrt.winrt_base):
    ...
    status: MediaFrameSourceGetPropertyStatus
    value: _winrt.winrt_base

class MediaFrameSourceGroup(_winrt.winrt_base):
    ...
    display_name: str
    id: str
    source_infos: winrt.windows.foundation.collections.IVectorView[MediaFrameSourceInfo]
    def find_all_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[MediaFrameSourceGroup]]:
        ...
    def from_id_async(id: str) -> winrt.windows.foundation.IAsyncOperation[MediaFrameSourceGroup]:
        ...
    def get_device_selector() -> str:
        ...

class MediaFrameSourceInfo(_winrt.winrt_base):
    ...
    coordinate_system: winrt.windows.perception.spatial.SpatialCoordinateSystem
    device_information: winrt.windows.devices.enumeration.DeviceInformation
    id: str
    media_stream_type: winrt.windows.media.capture.MediaStreamType
    properties: winrt.windows.foundation.collections.IMapView[uuid.UUID, _winrt.winrt_base]
    source_group: MediaFrameSourceGroup
    source_kind: MediaFrameSourceKind
    profile_id: str
    video_profile_media_description: winrt.windows.foundation.collections.IVectorView[winrt.windows.media.capture.MediaCaptureVideoProfileMediaDescription]
    def get_relative_panel(display_region: winrt.windows.ui.windowmanagement.DisplayRegion) -> winrt.windows.devices.enumeration.Panel:
        ...

class MultiSourceMediaFrameArrivedEventArgs(_winrt.winrt_base):
    ...

class MultiSourceMediaFrameReader(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    acquisition_mode: MediaFrameReaderAcquisitionMode
    def close() -> None:
        ...
    def start_async() -> winrt.windows.foundation.IAsyncOperation[MultiSourceMediaFrameReaderStartStatus]:
        ...
    def stop_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def try_acquire_latest_frame() -> MultiSourceMediaFrameReference:
        ...
    def add_frame_arrived(handler: winrt.windows.foundation.TypedEventHandler[MultiSourceMediaFrameReader, MultiSourceMediaFrameArrivedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_frame_arrived(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class MultiSourceMediaFrameReference(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    def close() -> None:
        ...
    def try_get_frame_reference_by_source_id(source_id: str) -> MediaFrameReference:
        ...

class VideoMediaFrame(_winrt.winrt_base):
    ...
    camera_intrinsics: winrt.windows.media.devices.core.CameraIntrinsics
    depth_media_frame: DepthMediaFrame
    direct3_d_surface: winrt.windows.graphics.directx.direct3d11.IDirect3DSurface
    frame_reference: MediaFrameReference
    infrared_media_frame: InfraredMediaFrame
    software_bitmap: winrt.windows.graphics.imaging.SoftwareBitmap
    video_format: VideoMediaFrameFormat
    def get_video_frame() -> winrt.windows.media.VideoFrame:
        ...

class VideoMediaFrameFormat(_winrt.winrt_base):
    ...
    depth_format: DepthMediaFrameFormat
    height: int
    media_frame_format: MediaFrameFormat
    width: int
