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
    import winrt.windows.storage
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class PhotoImportAccessMode(enum.IntEnum):
    READ_WRITE = 0
    READ_ONLY = 1
    READ_AND_DELETE = 2

class PhotoImportConnectionTransport(enum.IntEnum):
    UNKNOWN = 0
    USB = 1
    I_P = 2
    BLUETOOTH = 3

class PhotoImportContentType(enum.IntEnum):
    UNKNOWN = 0
    IMAGE = 1
    VIDEO = 2

class PhotoImportContentTypeFilter(enum.IntEnum):
    ONLY_IMAGES = 0
    ONLY_VIDEOS = 1
    IMAGES_AND_VIDEOS = 2
    IMAGES_AND_VIDEOS_FROM_CAMERA_ROLL = 3

class PhotoImportImportMode(enum.IntEnum):
    IMPORT_EVERYTHING = 0
    IGNORE_SIDECARS = 1
    IGNORE_SIBLINGS = 2
    IGNORE_SIDECARS_AND_SIBLINGS = 3

class PhotoImportItemSelectionMode(enum.IntEnum):
    SELECT_ALL = 0
    SELECT_NONE = 1
    SELECT_NEW = 2

class PhotoImportPowerSource(enum.IntEnum):
    UNKNOWN = 0
    BATTERY = 1
    EXTERNAL = 2

class PhotoImportSourceType(enum.IntEnum):
    GENERIC = 0
    CAMERA = 1
    MEDIA_PLAYER = 2
    PHONE = 3
    VIDEO = 4
    PERSONAL_INFO_MANAGER = 5
    AUDIO_RECORDER = 6

class PhotoImportStage(enum.IntEnum):
    NOT_STARTED = 0
    FINDING_ITEMS = 1
    IMPORTING_ITEMS = 2
    DELETING_IMPORTED_ITEMS_FROM_SOURCE = 3

class PhotoImportStorageMediumType(enum.IntEnum):
    UNDEFINED = 0
    FIXED = 1
    REMOVABLE = 2

class PhotoImportSubfolderCreationMode(enum.IntEnum):
    DO_NOT_CREATE_SUBFOLDERS = 0
    CREATE_SUBFOLDERS_FROM_FILE_DATE = 1
    CREATE_SUBFOLDERS_FROM_EXIF_DATE = 2
    KEEP_ORIGINAL_FOLDER_STRUCTURE = 3

class PhotoImportSubfolderDateFormat(enum.IntEnum):
    YEAR = 0
    YEAR_MONTH = 1
    YEAR_MONTH_DAY = 2

class PhotoImportProgress(_winrt.winrt_base):
    ...

class PhotoImportDeleteImportedItemsFromSourceResult(_winrt.winrt_base):
    ...
    deleted_items: winrt.windows.foundation.collections.IVectorView[PhotoImportItem]
    has_succeeded: bool
    photos_count: int
    photos_size_in_bytes: int
    session: PhotoImportSession
    siblings_count: int
    siblings_size_in_bytes: int
    sidecars_count: int
    sidecars_size_in_bytes: int
    total_count: int
    total_size_in_bytes: int
    videos_count: int
    videos_size_in_bytes: int

class PhotoImportFindItemsResult(_winrt.winrt_base):
    ...
    found_items: winrt.windows.foundation.collections.IVectorView[PhotoImportItem]
    has_succeeded: bool
    import_mode: PhotoImportImportMode
    photos_count: int
    photos_size_in_bytes: int
    selected_photos_count: int
    selected_photos_size_in_bytes: int
    selected_siblings_count: int
    selected_siblings_size_in_bytes: int
    selected_sidecars_count: int
    selected_sidecars_size_in_bytes: int
    selected_total_count: int
    selected_total_size_in_bytes: int
    selected_videos_count: int
    selected_videos_size_in_bytes: int
    session: PhotoImportSession
    siblings_count: int
    siblings_size_in_bytes: int
    sidecars_count: int
    sidecars_size_in_bytes: int
    total_count: int
    total_size_in_bytes: int
    videos_count: int
    videos_size_in_bytes: int
    def add_items_in_date_range_to_selection(range_start: winrt.windows.foundation.DateTime, range_length: winrt.windows.foundation.TimeSpan) -> None:
        ...
    def import_items_async() -> winrt.windows.foundation.IAsyncOperationWithProgress[PhotoImportImportItemsResult, PhotoImportProgress]:
        ...
    def select_all() -> None:
        ...
    def select_new_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def select_none() -> None:
        ...
    def set_import_mode(value: PhotoImportImportMode) -> None:
        ...
    def add_item_imported(value: winrt.windows.foundation.TypedEventHandler[PhotoImportFindItemsResult, PhotoImportItemImportedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_item_imported(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_selection_changed(value: winrt.windows.foundation.TypedEventHandler[PhotoImportFindItemsResult, PhotoImportSelectionChangedEventArgs]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_selection_changed(token: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class PhotoImportImportItemsResult(_winrt.winrt_base):
    ...
    has_succeeded: bool
    imported_items: winrt.windows.foundation.collections.IVectorView[PhotoImportItem]
    photos_count: int
    photos_size_in_bytes: int
    session: PhotoImportSession
    siblings_count: int
    siblings_size_in_bytes: int
    sidecars_count: int
    sidecars_size_in_bytes: int
    total_count: int
    total_size_in_bytes: int
    videos_count: int
    videos_size_in_bytes: int
    def delete_imported_items_from_source_async() -> winrt.windows.foundation.IAsyncOperationWithProgress[PhotoImportDeleteImportedItemsFromSourceResult, float]:
        ...

class PhotoImportItem(_winrt.winrt_base):
    ...
    is_selected: bool
    content_type: PhotoImportContentType
    date: winrt.windows.foundation.DateTime
    deleted_file_names: winrt.windows.foundation.collections.IVectorView[str]
    imported_file_names: winrt.windows.foundation.collections.IVectorView[str]
    item_key: int
    name: str
    sibling: PhotoImportSidecar
    sidecars: winrt.windows.foundation.collections.IVectorView[PhotoImportSidecar]
    size_in_bytes: int
    thumbnail: winrt.windows.storage.streams.IRandomAccessStreamReference
    video_segments: winrt.windows.foundation.collections.IVectorView[PhotoImportVideoSegment]
    path: str

class PhotoImportItemImportedEventArgs(_winrt.winrt_base):
    ...
    imported_item: PhotoImportItem

class PhotoImportManager(_winrt.winrt_base):
    ...
    def find_all_sources_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[PhotoImportSource]]:
        ...
    def get_pending_operations() -> winrt.windows.foundation.collections.IVectorView[PhotoImportOperation]:
        ...
    def is_supported_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...

class PhotoImportOperation(_winrt.winrt_base):
    ...
    continue_deleting_imported_items_from_source_async: winrt.windows.foundation.IAsyncOperationWithProgress[PhotoImportDeleteImportedItemsFromSourceResult, float]
    continue_finding_items_async: winrt.windows.foundation.IAsyncOperationWithProgress[PhotoImportFindItemsResult, int]
    continue_importing_items_async: winrt.windows.foundation.IAsyncOperationWithProgress[PhotoImportImportItemsResult, PhotoImportProgress]
    session: PhotoImportSession
    stage: PhotoImportStage

class PhotoImportSelectionChangedEventArgs(_winrt.winrt_base):
    ...
    is_selection_empty: bool

class PhotoImportSession(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    subfolder_creation_mode: PhotoImportSubfolderCreationMode
    destination_folder: winrt.windows.storage.IStorageFolder
    destination_file_name_prefix: str
    append_session_date_to_destination_folder: bool
    session_id: uuid.UUID
    source: PhotoImportSource
    subfolder_date_format: PhotoImportSubfolderDateFormat
    remember_deselected_items: bool
    def close() -> None:
        ...
    def find_items_async(content_type_filter: PhotoImportContentTypeFilter, item_selection_mode: PhotoImportItemSelectionMode) -> winrt.windows.foundation.IAsyncOperationWithProgress[PhotoImportFindItemsResult, int]:
        ...

class PhotoImportSidecar(_winrt.winrt_base):
    ...
    date: winrt.windows.foundation.DateTime
    name: str
    size_in_bytes: int

class PhotoImportSource(_winrt.winrt_base):
    ...
    battery_level_percent: typing.Optional[int]
    connection_protocol: str
    connection_transport: PhotoImportConnectionTransport
    date_time: typing.Optional[winrt.windows.foundation.DateTime]
    description: str
    display_name: str
    id: str
    is_locked: typing.Optional[bool]
    is_mass_storage: bool
    manufacturer: str
    model: str
    power_source: PhotoImportPowerSource
    serial_number: str
    storage_media: winrt.windows.foundation.collections.IVectorView[PhotoImportStorageMedium]
    thumbnail: winrt.windows.storage.streams.IRandomAccessStreamReference
    type: PhotoImportSourceType
    def create_import_session() -> PhotoImportSession:
        ...
    def from_folder_async(source_root_folder: winrt.windows.storage.IStorageFolder) -> winrt.windows.foundation.IAsyncOperation[PhotoImportSource]:
        ...
    def from_id_async(source_id: str) -> winrt.windows.foundation.IAsyncOperation[PhotoImportSource]:
        ...

class PhotoImportStorageMedium(_winrt.winrt_base):
    ...
    available_space_in_bytes: int
    capacity_in_bytes: int
    description: str
    name: str
    serial_number: str
    storage_medium_type: PhotoImportStorageMediumType
    supported_access_mode: PhotoImportAccessMode
    def refresh() -> None:
        ...

class PhotoImportVideoSegment(_winrt.winrt_base):
    ...
    date: winrt.windows.foundation.DateTime
    name: str
    sibling: PhotoImportSidecar
    sidecars: winrt.windows.foundation.collections.IVectorView[PhotoImportSidecar]
    size_in_bytes: int

