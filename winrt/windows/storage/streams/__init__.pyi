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
    import winrt.windows.system
except Exception:
    pass

class ByteOrder(enum.IntEnum):
    LITTLE_ENDIAN = 0
    BIG_ENDIAN = 1

class FileOpenDisposition(enum.IntEnum):
    OPEN_EXISTING = 0
    OPEN_ALWAYS = 1
    CREATE_NEW = 2
    CREATE_ALWAYS = 3
    TRUNCATE_EXISTING = 4

class InputStreamOptions(enum.IntFlag):
    NONE = 0
    PARTIAL = 0x1
    READ_AHEAD = 0x2

class UnicodeEncoding(enum.IntEnum):
    UTF8 = 0
    UTF16_L_E = 1
    UTF16_B_E = 2

class Buffer(IBuffer, _winrt.winrt_base):
    ...
    length: int
    capacity: int
    def create_copy_from_memory_buffer(input: winrt.windows.foundation.IMemoryBuffer) -> Buffer:
        ...
    def create_memory_buffer_over_i_buffer(input: IBuffer) -> winrt.windows.foundation.MemoryBuffer:
        ...

class DataReader(IDataReader, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    unicode_encoding: UnicodeEncoding
    input_stream_options: InputStreamOptions
    byte_order: ByteOrder
    unconsumed_buffer_length: int
    def close() -> None:
        ...
    def detach_buffer() -> IBuffer:
        ...
    def detach_stream() -> IInputStream:
        ...
    def from_buffer(buffer: IBuffer) -> DataReader:
        ...
    def load_async(count: int) -> DataReaderLoadOperation:
        ...
    def read_boolean() -> bool:
        ...
    def read_buffer(length: int) -> IBuffer:
        ...
    def read_byte() -> int:
        ...
    def read_bytes(value_size: int) -> typing.List[int]:
        ...
    def read_date_time() -> winrt.windows.foundation.DateTime:
        ...
    def read_double() -> float:
        ...
    def read_guid() -> uuid.UUID:
        ...
    def read_int16() -> int:
        ...
    def read_int32() -> int:
        ...
    def read_int64() -> int:
        ...
    def read_single() -> float:
        ...
    def read_string(code_unit_count: int) -> str:
        ...
    def read_time_span() -> winrt.windows.foundation.TimeSpan:
        ...
    def read_uint16() -> int:
        ...
    def read_uint32() -> int:
        ...
    def read_uint64() -> int:
        ...

class DataReaderLoadOperation(winrt.windows.foundation.IAsyncOperation[int], winrt.windows.foundation.IAsyncInfo, _winrt.winrt_base):
    ...
    error_code: winrt.windows.foundation.HResult
    id: int
    status: winrt.windows.foundation.AsyncStatus
    completed: winrt.windows.foundation.AsyncOperationCompletedHandler[int]
    def cancel() -> None:
        ...
    def close() -> None:
        ...
    def get_results() -> int:
        ...

class DataWriter(IDataWriter, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    unicode_encoding: UnicodeEncoding
    byte_order: ByteOrder
    unstored_buffer_length: int
    def close() -> None:
        ...
    def detach_buffer() -> IBuffer:
        ...
    def detach_stream() -> IOutputStream:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def measure_string(value: str) -> int:
        ...
    def store_async() -> DataWriterStoreOperation:
        ...
    def write_boolean(value: bool) -> None:
        ...
    def write_buffer(buffer: IBuffer) -> None:
        ...
    def write_buffer(buffer: IBuffer, start: int, count: int) -> None:
        ...
    def write_byte(value: int) -> None:
        ...
    def write_bytes(value: typing.Sequence[int]) -> None:
        ...
    def write_date_time(value: winrt.windows.foundation.DateTime) -> None:
        ...
    def write_double(value: float) -> None:
        ...
    def write_guid(value: uuid.UUID) -> None:
        ...
    def write_int16(value: int) -> None:
        ...
    def write_int32(value: int) -> None:
        ...
    def write_int64(value: int) -> None:
        ...
    def write_single(value: float) -> None:
        ...
    def write_string(value: str) -> int:
        ...
    def write_time_span(value: winrt.windows.foundation.TimeSpan) -> None:
        ...
    def write_uint16(value: int) -> None:
        ...
    def write_uint32(value: int) -> None:
        ...
    def write_uint64(value: int) -> None:
        ...

class DataWriterStoreOperation(winrt.windows.foundation.IAsyncOperation[int], winrt.windows.foundation.IAsyncInfo, _winrt.winrt_base):
    ...
    error_code: winrt.windows.foundation.HResult
    id: int
    status: winrt.windows.foundation.AsyncStatus
    completed: winrt.windows.foundation.AsyncOperationCompletedHandler[int]
    def cancel() -> None:
        ...
    def close() -> None:
        ...
    def get_results() -> int:
        ...

class FileInputStream(IInputStream, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    def close() -> None:
        ...
    def read_async(buffer: IBuffer, count: int, options: InputStreamOptions) -> winrt.windows.foundation.IAsyncOperationWithProgress[IBuffer, int]:
        ...

class FileOutputStream(IOutputStream, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    def close() -> None:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def write_async(buffer: IBuffer) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...

class FileRandomAccessStream(IRandomAccessStream, IOutputStream, winrt.windows.foundation.IClosable, IInputStream, _winrt.winrt_base):
    ...
    size: int
    can_read: bool
    can_write: bool
    position: int
    def clone_stream() -> IRandomAccessStream:
        ...
    def close() -> None:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def get_input_stream_at(position: int) -> IInputStream:
        ...
    def get_output_stream_at(position: int) -> IOutputStream:
        ...
    def open_async(file_path: str, access_mode: winrt.windows.storage.FileAccessMode) -> winrt.windows.foundation.IAsyncOperation[IRandomAccessStream]:
        ...
    def open_async(file_path: str, access_mode: winrt.windows.storage.FileAccessMode, sharing_options: winrt.windows.storage.StorageOpenOptions, open_disposition: FileOpenDisposition) -> winrt.windows.foundation.IAsyncOperation[IRandomAccessStream]:
        ...
    def open_for_user_async(user: winrt.windows.system.User, file_path: str, access_mode: winrt.windows.storage.FileAccessMode) -> winrt.windows.foundation.IAsyncOperation[IRandomAccessStream]:
        ...
    def open_for_user_async(user: winrt.windows.system.User, file_path: str, access_mode: winrt.windows.storage.FileAccessMode, sharing_options: winrt.windows.storage.StorageOpenOptions, open_disposition: FileOpenDisposition) -> winrt.windows.foundation.IAsyncOperation[IRandomAccessStream]:
        ...
    def open_transacted_write_async(file_path: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageStreamTransaction]:
        ...
    def open_transacted_write_async(file_path: str, open_options: winrt.windows.storage.StorageOpenOptions, open_disposition: FileOpenDisposition) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageStreamTransaction]:
        ...
    def open_transacted_write_for_user_async(user: winrt.windows.system.User, file_path: str) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageStreamTransaction]:
        ...
    def open_transacted_write_for_user_async(user: winrt.windows.system.User, file_path: str, open_options: winrt.windows.storage.StorageOpenOptions, open_disposition: FileOpenDisposition) -> winrt.windows.foundation.IAsyncOperation[winrt.windows.storage.StorageStreamTransaction]:
        ...
    def read_async(buffer: IBuffer, count: int, options: InputStreamOptions) -> winrt.windows.foundation.IAsyncOperationWithProgress[IBuffer, int]:
        ...
    def seek(position: int) -> None:
        ...
    def write_async(buffer: IBuffer) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...

class InMemoryRandomAccessStream(IRandomAccessStream, IOutputStream, winrt.windows.foundation.IClosable, IInputStream, _winrt.winrt_base):
    ...
    size: int
    can_read: bool
    can_write: bool
    position: int
    def clone_stream() -> IRandomAccessStream:
        ...
    def close() -> None:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def get_input_stream_at(position: int) -> IInputStream:
        ...
    def get_output_stream_at(position: int) -> IOutputStream:
        ...
    def read_async(buffer: IBuffer, count: int, options: InputStreamOptions) -> winrt.windows.foundation.IAsyncOperationWithProgress[IBuffer, int]:
        ...
    def seek(position: int) -> None:
        ...
    def write_async(buffer: IBuffer) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...

class InputStreamOverStream(IInputStream, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    def close() -> None:
        ...
    def read_async(buffer: IBuffer, count: int, options: InputStreamOptions) -> winrt.windows.foundation.IAsyncOperationWithProgress[IBuffer, int]:
        ...

class OutputStreamOverStream(IOutputStream, winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    def close() -> None:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def write_async(buffer: IBuffer) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...

class RandomAccessStream(_winrt.winrt_base):
    ...
    def copy_and_close_async(source: IInputStream, destination: IOutputStream) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...
    def copy_async(source: IInputStream, destination: IOutputStream) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...
    def copy_async(source: IInputStream, destination: IOutputStream, bytes_to_copy: int) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...

class RandomAccessStreamOverStream(IRandomAccessStream, IOutputStream, winrt.windows.foundation.IClosable, IInputStream, _winrt.winrt_base):
    ...
    size: int
    can_read: bool
    can_write: bool
    position: int
    def clone_stream() -> IRandomAccessStream:
        ...
    def close() -> None:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def get_input_stream_at(position: int) -> IInputStream:
        ...
    def get_output_stream_at(position: int) -> IOutputStream:
        ...
    def read_async(buffer: IBuffer, count: int, options: InputStreamOptions) -> winrt.windows.foundation.IAsyncOperationWithProgress[IBuffer, int]:
        ...
    def seek(position: int) -> None:
        ...
    def write_async(buffer: IBuffer) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...

class RandomAccessStreamReference(IRandomAccessStreamReference, _winrt.winrt_base):
    ...
    def create_from_file(file: winrt.windows.storage.IStorageFile) -> RandomAccessStreamReference:
        ...
    def create_from_stream(stream: IRandomAccessStream) -> RandomAccessStreamReference:
        ...
    def create_from_uri(uri: winrt.windows.foundation.Uri) -> RandomAccessStreamReference:
        ...
    def open_read_async() -> winrt.windows.foundation.IAsyncOperation[IRandomAccessStreamWithContentType]:
        ...

class IBuffer(_winrt.winrt_base):
    ...
    capacity: int
    length: int

class IContentTypeProvider(_winrt.winrt_base):
    ...
    content_type: str

class IDataReader(_winrt.winrt_base):
    ...
    byte_order: ByteOrder
    input_stream_options: InputStreamOptions
    unconsumed_buffer_length: int
    unicode_encoding: UnicodeEncoding
    def detach_buffer() -> IBuffer:
        ...
    def detach_stream() -> IInputStream:
        ...
    def load_async(count: int) -> DataReaderLoadOperation:
        ...
    def read_boolean() -> bool:
        ...
    def read_buffer(length: int) -> IBuffer:
        ...
    def read_byte() -> int:
        ...
    def read_bytes(value_size: int) -> typing.List[int]:
        ...
    def read_date_time() -> winrt.windows.foundation.DateTime:
        ...
    def read_double() -> float:
        ...
    def read_guid() -> uuid.UUID:
        ...
    def read_int16() -> int:
        ...
    def read_int32() -> int:
        ...
    def read_int64() -> int:
        ...
    def read_single() -> float:
        ...
    def read_string(code_unit_count: int) -> str:
        ...
    def read_time_span() -> winrt.windows.foundation.TimeSpan:
        ...
    def read_uint16() -> int:
        ...
    def read_uint32() -> int:
        ...
    def read_uint64() -> int:
        ...

class IDataWriter(_winrt.winrt_base):
    ...
    byte_order: ByteOrder
    unicode_encoding: UnicodeEncoding
    unstored_buffer_length: int
    def detach_buffer() -> IBuffer:
        ...
    def detach_stream() -> IOutputStream:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def measure_string(value: str) -> int:
        ...
    def store_async() -> DataWriterStoreOperation:
        ...
    def write_boolean(value: bool) -> None:
        ...
    def write_buffer(buffer: IBuffer) -> None:
        ...
    def write_buffer(buffer: IBuffer, start: int, count: int) -> None:
        ...
    def write_byte(value: int) -> None:
        ...
    def write_bytes(value: typing.Sequence[int]) -> None:
        ...
    def write_date_time(value: winrt.windows.foundation.DateTime) -> None:
        ...
    def write_double(value: float) -> None:
        ...
    def write_guid(value: uuid.UUID) -> None:
        ...
    def write_int16(value: int) -> None:
        ...
    def write_int32(value: int) -> None:
        ...
    def write_int64(value: int) -> None:
        ...
    def write_single(value: float) -> None:
        ...
    def write_string(value: str) -> int:
        ...
    def write_time_span(value: winrt.windows.foundation.TimeSpan) -> None:
        ...
    def write_uint16(value: int) -> None:
        ...
    def write_uint32(value: int) -> None:
        ...
    def write_uint64(value: int) -> None:
        ...

class IInputStream(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    def read_async(buffer: IBuffer, count: int, options: InputStreamOptions) -> winrt.windows.foundation.IAsyncOperationWithProgress[IBuffer, int]:
        ...
    def close() -> None:
        ...

class IInputStreamReference(_winrt.winrt_base):
    ...
    def open_sequential_read_async() -> winrt.windows.foundation.IAsyncOperation[IInputStream]:
        ...

class IOutputStream(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def write_async(buffer: IBuffer) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...
    def close() -> None:
        ...

class IPropertySetSerializer(_winrt.winrt_base):
    ...
    def deserialize(property_set: winrt.windows.foundation.collections.IPropertySet, buffer: IBuffer) -> None:
        ...
    def serialize(property_set: winrt.windows.foundation.collections.IPropertySet) -> IBuffer:
        ...

class IRandomAccessStream(winrt.windows.foundation.IClosable, IInputStream, IOutputStream, _winrt.winrt_base):
    ...
    can_read: bool
    can_write: bool
    position: int
    size: int
    def clone_stream() -> IRandomAccessStream:
        ...
    def get_input_stream_at(position: int) -> IInputStream:
        ...
    def get_output_stream_at(position: int) -> IOutputStream:
        ...
    def seek(position: int) -> None:
        ...
    def close() -> None:
        ...
    def read_async(buffer: IBuffer, count: int, options: InputStreamOptions) -> winrt.windows.foundation.IAsyncOperationWithProgress[IBuffer, int]:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def write_async(buffer: IBuffer) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...

class IRandomAccessStreamReference(_winrt.winrt_base):
    ...
    def open_read_async() -> winrt.windows.foundation.IAsyncOperation[IRandomAccessStreamWithContentType]:
        ...

class IRandomAccessStreamWithContentType(IRandomAccessStream, winrt.windows.foundation.IClosable, IInputStream, IOutputStream, IContentTypeProvider, _winrt.winrt_base):
    ...
    can_read: bool
    can_write: bool
    position: int
    size: int
    content_type: str
    def clone_stream() -> IRandomAccessStream:
        ...
    def get_input_stream_at(position: int) -> IInputStream:
        ...
    def get_output_stream_at(position: int) -> IOutputStream:
        ...
    def seek(position: int) -> None:
        ...
    def close() -> None:
        ...
    def read_async(buffer: IBuffer, count: int, options: InputStreamOptions) -> winrt.windows.foundation.IAsyncOperationWithProgress[IBuffer, int]:
        ...
    def flush_async() -> winrt.windows.foundation.IAsyncOperation[bool]:
        ...
    def write_async(buffer: IBuffer) -> winrt.windows.foundation.IAsyncOperationWithProgress[int, int]:
        ...
