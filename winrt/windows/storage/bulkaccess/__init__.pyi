# WARNING: Please don't edit this file. It was generated by Python/WinRT v1.2.3.4

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
    import winrt.windows.storage.fileproperties
except Exception:
    pass

try:
    import winrt.windows.storage.search
except Exception:
    pass

try:
    import winrt.windows.storage.streams
except Exception:
    pass

class FileInformation(IStorageItemInformation, winrt.windows.storage.IStorageFile, winrt.windows.storage.streams.IInputStreamReference, winrt.windows.storage.streams.IRandomAccessStreamReference, winrt.windows.storage.IStorageItem, winrt.windows.storage.IStorageItemProperties, winrt.windows.storage.IStorageItem2, winrt.windows.storage.IStorageItemPropertiesWithProvider, winrt.windows.storage.IStorageFilePropertiesWithAvailability, winrt.windows.storage.IStorageFile2, _winrt.winrt_base):
    ...
    basic_properties: winrt.windows.storage.fileproperties.BasicProperties
    document_properties: winrt.windows.storage.fileproperties.DocumentProperties
    image_properties: winrt.windows.storage.fileproperties.ImageProperties
    music_properties: winrt.windows.storage.fileproperties.MusicProperties
    thumbnail: winrt.windows.storage.fileproperties.StorageItemThumbnail
    video_properties: winrt.windows.storage.fileproperties.VideoProperties
    content_type: str
    file_type: str
    is_available: bool
    attributes: winrt.windows.storage.FileAttributes
    date_created: winrt.windows.foundation.DateTime
    name: str
    path: str
    display_name: str
    display_type: str
    folder_relative_id: str
    properties: winrt.windows.storage.fileproperties.StorageItemContentProperties
    provider: winrt.windows.storage.StorageProvider
    def copy_and_replace_async(file_to_replace: winrt.windows.storage.IStorageFile) -> winrt.windows.foundation.IAsyncAction:
        ...
    def copy_async(destination_folder: winrt.windows.storage.IStorageFolder) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFile]:
        ...
    def copy_async(destination_folder: winrt.windows.storage.IStorageFolder, desired_new_name: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFile]:
        ...
    def copy_async(destination_folder: winrt.windows.storage.IStorageFolder, desired_new_name: str, option: winrt.windows.storage.NameCollisionOption) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFile]:
        ...
    def delete_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def delete_async(option: winrt.windows.storage.StorageDeleteOption) -> winrt.windows.foundation.IAsyncAction:
        ...
    def get_basic_properties_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.fileproperties.BasicProperties]:
        ...
    def get_parent_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFolder]:
        ...
    def get_thumbnail_async(mode: winrt.windows.storage.fileproperties.ThumbnailMode) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.fileproperties.StorageItemThumbnail]:
        ...
    def get_thumbnail_async(mode: winrt.windows.storage.fileproperties.ThumbnailMode, requested_size: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.fileproperties.StorageItemThumbnail]:
        ...
    def get_thumbnail_async(mode: winrt.windows.storage.fileproperties.ThumbnailMode, requested_size: int, options: winrt.windows.storage.fileproperties.ThumbnailOptions) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.fileproperties.StorageItemThumbnail]:
        ...
    def is_equal(item: winrt.windows.storage.IStorageItem) -> bool:
        ...
    def is_of_type(type: winrt.windows.storage.StorageItemTypes) -> bool:
        ...
    def move_and_replace_async(file_to_replace: winrt.windows.storage.IStorageFile) -> winrt.windows.foundation.IAsyncAction:
        ...
    def move_async(destination_folder: winrt.windows.storage.IStorageFolder) -> winrt.windows.foundation.IAsyncAction:
        ...
    def move_async(destination_folder: winrt.windows.storage.IStorageFolder, desired_new_name: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def move_async(destination_folder: winrt.windows.storage.IStorageFolder, desired_new_name: str, option: winrt.windows.storage.NameCollisionOption) -> winrt.windows.foundation.IAsyncAction:
        ...
    def open_async(access_mode: winrt.windows.storage.FileAccessMode) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.IRandomAccessStream]:
        ...
    def open_async(access_mode: winrt.windows.storage.FileAccessMode, options: winrt.windows.storage.StorageOpenOptions) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.IRandomAccessStream]:
        ...
    def open_read_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.IRandomAccessStreamWithContentType]:
        ...
    def open_sequential_read_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.streams.IInputStream]:
        ...
    def open_transacted_write_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageStreamTransaction]:
        ...
    def open_transacted_write_async(options: winrt.windows.storage.StorageOpenOptions) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageStreamTransaction]:
        ...
    def rename_async(desired_name: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def rename_async(desired_name: str, option: winrt.windows.storage.NameCollisionOption) -> winrt.windows.foundation.IAsyncAction:
        ...
    def add_properties_updated(changed_handler: winrt.windows.foundation.TypedEventHandler[IStorageItemInformation, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_properties_updated(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_thumbnail_updated(changed_handler: winrt.windows.foundation.TypedEventHandler[IStorageItemInformation, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_thumbnail_updated(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class FileInformationFactory(_winrt.winrt_base):
    ...
    def get_files_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[FileInformation]]:
        ...
    def get_files_async(start_index: int, max_items_to_retrieve: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[FileInformation]]:
        ...
    def get_folders_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[FolderInformation]]:
        ...
    def get_folders_async(start_index: int, max_items_to_retrieve: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[FolderInformation]]:
        ...
    def get_items_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[IStorageItemInformation]]:
        ...
    def get_items_async(start_index: int, max_items_to_retrieve: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[IStorageItemInformation]]:
        ...
    def get_virtualized_files_vector() -> _winrt.winrt_base:
        ...
    def get_virtualized_folders_vector() -> _winrt.winrt_base:
        ...
    def get_virtualized_items_vector() -> _winrt.winrt_base:
        ...

class FolderInformation(IStorageItemInformation, winrt.windows.storage.IStorageFolder, winrt.windows.storage.IStorageItem, winrt.windows.storage.IStorageItemProperties, winrt.windows.storage.search.IStorageFolderQueryOperations, winrt.windows.storage.IStorageItem2, winrt.windows.storage.IStorageFolder2, winrt.windows.storage.IStorageItemPropertiesWithProvider, _winrt.winrt_base):
    ...
    basic_properties: winrt.windows.storage.fileproperties.BasicProperties
    document_properties: winrt.windows.storage.fileproperties.DocumentProperties
    image_properties: winrt.windows.storage.fileproperties.ImageProperties
    music_properties: winrt.windows.storage.fileproperties.MusicProperties
    thumbnail: winrt.windows.storage.fileproperties.StorageItemThumbnail
    video_properties: winrt.windows.storage.fileproperties.VideoProperties
    attributes: winrt.windows.storage.FileAttributes
    date_created: winrt.windows.foundation.DateTime
    name: str
    path: str
    display_name: str
    display_type: str
    folder_relative_id: str
    properties: winrt.windows.storage.fileproperties.StorageItemContentProperties
    provider: winrt.windows.storage.StorageProvider
    def are_query_options_supported(query_options: winrt.windows.storage.search.QueryOptions) -> bool:
        ...
    def create_file_async(desired_name: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFile]:
        ...
    def create_file_async(desired_name: str, options: winrt.windows.storage.CreationCollisionOption) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFile]:
        ...
    def create_file_query() -> winrt.windows.storage.search.StorageFileQueryResult:
        ...
    def create_file_query(query: winrt.windows.storage.search.CommonFileQuery) -> winrt.windows.storage.search.StorageFileQueryResult:
        ...
    def create_file_query_with_options(query_options: winrt.windows.storage.search.QueryOptions) -> winrt.windows.storage.search.StorageFileQueryResult:
        ...
    def create_folder_async(desired_name: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFolder]:
        ...
    def create_folder_async(desired_name: str, options: winrt.windows.storage.CreationCollisionOption) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFolder]:
        ...
    def create_folder_query() -> winrt.windows.storage.search.StorageFolderQueryResult:
        ...
    def create_folder_query(query: winrt.windows.storage.search.CommonFolderQuery) -> winrt.windows.storage.search.StorageFolderQueryResult:
        ...
    def create_folder_query_with_options(query_options: winrt.windows.storage.search.QueryOptions) -> winrt.windows.storage.search.StorageFolderQueryResult:
        ...
    def create_item_query() -> winrt.windows.storage.search.StorageItemQueryResult:
        ...
    def create_item_query_with_options(query_options: winrt.windows.storage.search.QueryOptions) -> winrt.windows.storage.search.StorageItemQueryResult:
        ...
    def delete_async() -> winrt.windows.foundation.IAsyncAction:
        ...
    def delete_async(option: winrt.windows.storage.StorageDeleteOption) -> winrt.windows.foundation.IAsyncAction:
        ...
    def get_basic_properties_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.fileproperties.BasicProperties]:
        ...
    def get_file_async(name: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFile]:
        ...
    def get_files_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.storage.StorageFile]]:
        ...
    def get_files_async(query: winrt.windows.storage.search.CommonFileQuery) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.storage.StorageFile]]:
        ...
    def get_files_async(query: winrt.windows.storage.search.CommonFileQuery, start_index: int, max_items_to_retrieve: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.storage.StorageFile]]:
        ...
    def get_folder_async(name: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFolder]:
        ...
    def get_folders_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.storage.StorageFolder]]:
        ...
    def get_folders_async(query: winrt.windows.storage.search.CommonFolderQuery) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.storage.StorageFolder]]:
        ...
    def get_folders_async(query: winrt.windows.storage.search.CommonFolderQuery, start_index: int, max_items_to_retrieve: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.storage.StorageFolder]]:
        ...
    def get_indexed_state_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.search.IndexedState]:
        ...
    def get_item_async(name: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.IStorageItem]:
        ...
    def get_items_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.storage.IStorageItem]]:
        ...
    def get_items_async(start_index: int, max_items_to_retrieve: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[winrt.windows.storage.IStorageItem]]:
        ...
    def get_parent_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageFolder]:
        ...
    def get_thumbnail_async(mode: winrt.windows.storage.fileproperties.ThumbnailMode) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.fileproperties.StorageItemThumbnail]:
        ...
    def get_thumbnail_async(mode: winrt.windows.storage.fileproperties.ThumbnailMode, requested_size: int) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.fileproperties.StorageItemThumbnail]:
        ...
    def get_thumbnail_async(mode: winrt.windows.storage.fileproperties.ThumbnailMode, requested_size: int, options: winrt.windows.storage.fileproperties.ThumbnailOptions) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.fileproperties.StorageItemThumbnail]:
        ...
    def is_common_file_query_supported(query: winrt.windows.storage.search.CommonFileQuery) -> bool:
        ...
    def is_common_folder_query_supported(query: winrt.windows.storage.search.CommonFolderQuery) -> bool:
        ...
    def is_equal(item: winrt.windows.storage.IStorageItem) -> bool:
        ...
    def is_of_type(type: winrt.windows.storage.StorageItemTypes) -> bool:
        ...
    def rename_async(desired_name: str) -> winrt.windows.foundation.IAsyncAction:
        ...
    def rename_async(desired_name: str, option: winrt.windows.storage.NameCollisionOption) -> winrt.windows.foundation.IAsyncAction:
        ...
    def try_get_item_async(name: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.IStorageItem]:
        ...
    def add_properties_updated(changed_handler: winrt.windows.foundation.TypedEventHandler[IStorageItemInformation, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_properties_updated(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_thumbnail_updated(changed_handler: winrt.windows.foundation.TypedEventHandler[IStorageItemInformation, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_thumbnail_updated(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...

class IStorageItemInformation(_winrt.winrt_base):
    ...
    basic_properties: winrt.windows.storage.fileproperties.BasicProperties
    document_properties: winrt.windows.storage.fileproperties.DocumentProperties
    image_properties: winrt.windows.storage.fileproperties.ImageProperties
    music_properties: winrt.windows.storage.fileproperties.MusicProperties
    thumbnail: winrt.windows.storage.fileproperties.StorageItemThumbnail
    video_properties: winrt.windows.storage.fileproperties.VideoProperties
    def add_properties_updated(changed_handler: winrt.windows.foundation.TypedEventHandler[IStorageItemInformation, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_properties_updated(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
    def add_thumbnail_updated(changed_handler: winrt.windows.foundation.TypedEventHandler[IStorageItemInformation, _winrt.winrt_base]) -> winrt.windows.foundation.EventRegistrationToken:
        ...
    def remove_thumbnail_updated(event_cookie: winrt.windows.foundation.EventRegistrationToken) -> None:
        ...
