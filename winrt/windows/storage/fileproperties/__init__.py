# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

import enum

import winrt

_ns_module = winrt._import_ns_module("Windows.Storage.FileProperties")

try:
    import winrt.windows.devices.geolocation
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
    import winrt.windows.storage
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class PhotoOrientation(enum.IntEnum):
    UNSPECIFIED = 0
    NORMAL = 1
    FLIP_HORIZONTAL = 2
    ROTATE180 = 3
    FLIP_VERTICAL = 4
    TRANSPOSE = 5
    ROTATE270 = 6
    TRANSVERSE = 7
    ROTATE90 = 8

class PropertyPrefetchOptions(enum.IntFlag):
    NONE = 0
    MUSIC_PROPERTIES = 0x1
    VIDEO_PROPERTIES = 0x2
    IMAGE_PROPERTIES = 0x4
    DOCUMENT_PROPERTIES = 0x8
    BASIC_PROPERTIES = 0x10

class ThumbnailMode(enum.IntEnum):
    PICTURES_VIEW = 0
    VIDEOS_VIEW = 1
    MUSIC_VIEW = 2
    DOCUMENTS_VIEW = 3
    LIST_VIEW = 4
    SINGLE_ITEM = 5

class ThumbnailOptions(enum.IntFlag):
    NONE = 0
    RETURN_ONLY_IF_CACHED = 0x1
    RESIZE_THUMBNAIL = 0x2
    USE_CURRENT_SCALE = 0x4

class ThumbnailType(enum.IntEnum):
    IMAGE = 0
    ICON = 1

class VideoOrientation(enum.IntEnum):
    NORMAL = 0
    ROTATE90 = 90
    ROTATE180 = 180
    ROTATE270 = 270

BasicProperties = _ns_module.BasicProperties
DocumentProperties = _ns_module.DocumentProperties
GeotagHelper = _ns_module.GeotagHelper
ImageProperties = _ns_module.ImageProperties
MusicProperties = _ns_module.MusicProperties
StorageItemContentProperties = _ns_module.StorageItemContentProperties
StorageItemThumbnail = _ns_module.StorageItemThumbnail
VideoProperties = _ns_module.VideoProperties
IStorageItemExtraProperties = _ns_module.IStorageItemExtraProperties