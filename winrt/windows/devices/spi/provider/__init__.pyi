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

class ProviderSpiMode(enum.IntEnum):
    MODE0 = 0
    MODE1 = 1
    MODE2 = 2
    MODE3 = 3

class ProviderSpiSharingMode(enum.IntEnum):
    EXCLUSIVE = 0
    SHARED = 1

class ProviderSpiConnectionSettings(_winrt.winrt_base):
    ...
    sharing_mode: ProviderSpiSharingMode
    mode: ProviderSpiMode
    data_bit_length: int
    clock_frequency: int
    chip_select_line: int

class ISpiControllerProvider(_winrt.winrt_base):
    ...
    def get_device_provider(settings: ProviderSpiConnectionSettings) -> ISpiDeviceProvider:
        ...

class ISpiDeviceProvider(winrt.windows.foundation.IClosable, _winrt.winrt_base):
    ...
    connection_settings: ProviderSpiConnectionSettings
    device_id: str
    def read(buffer_size: int) -> typing.List[int]:
        ...
    def transfer_full_duplex(write_buffer: typing.Sequence[int], read_buffer_size: int) -> typing.List[int]:
        ...
    def transfer_sequential(write_buffer: typing.Sequence[int], read_buffer_size: int) -> typing.List[int]:
        ...
    def write(buffer: typing.Sequence[int]) -> None:
        ...
    def close() -> None:
        ...

class ISpiProvider(_winrt.winrt_base):
    ...
    def get_controllers_async() -> winrt.windows.foundation.IAsyncOperation[winrt.windows.foundation.collections.IVectorView[ISpiControllerProvider]]:
        ...
